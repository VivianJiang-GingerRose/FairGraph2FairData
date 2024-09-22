#!/usr/bin/env python
import itertools
from collections import deque
from itertools import permutations

import networkx as nx
from tqdm.auto import trange

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from FairnessAwareHC.base import DAG
from FairnessAwareHC.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)
from FairnessAwareHC.global_vars import SHOW_PROGRESS
from FairnessAwareHC.models import BayesianNetwork
from FairnessAwareHC.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from FairnessAwareHC.sampling import BayesianModelSampling

from fairlearn.metrics import demographic_parity_difference
from math import log


def generate_linear_combinations(node_list_1, node_list_2):
    """
    Function that generates all possible combinations between two sets of nodes
    and returns a list of tuples with all possible combinations.
    """

    all_combinations = list(itertools.product(node_list_1, node_list_2))
    all_combinations = set(all_combinations)

    # Remove the combinations where the two elements are the same
    all_combinations = [x for x in all_combinations if x[0] != x[1]]
    
    return all_combinations


def one_hot_encoding_dataframe_columns(df):
    """Function that one-hot encodes given columns"""
    ohc = OneHotEncoder()
    for col in df.columns:
        if (df[col].dtype == 'object') or (df[col].dtype.name == 'category'):
            df_ohc = pd.DataFrame(ohc.fit_transform(df[[col]]).toarray(), columns=ohc.categories_)

            df_ohc = df_ohc.add_prefix(col+'_')

            # Drop the original column and concatenate the one-hot encoded data
            df = pd.concat([df.drop([col], axis=1), df_ohc], axis=1)
    return df


def preprocess_data(df, target_name):

    """Function that pre-processes the data, ready for modelling"""
    # Parsed target_name is a set, convert to string.
    oc = list(target_name)[0]

    # Columns to be one-hot encoded. 
    # Right now we will one-hot encode all columns with object or category data types except the target column.
    cols_ohe = df.select_dtypes(include=['object', 'category']).columns
    cols_ohe = cols_ohe[cols_ohe != oc]
    
    df1 = one_hot_encoding_dataframe_columns(df[cols_ohe])
    df2 = pd.concat([df1, df.drop(cols_ohe, axis=1)], axis=1)

    df2.columns = df2.columns.map(str)

    # Also split data for modelling
    X = df2.drop([oc], axis=1)
    y = df2[oc]

    return df2, X, y

def build_xgboost_model(X, y, X_train,  X_test, y_train, y_test):
    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    
    # Convert data to DMatrix object, which is optimized for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dall = xgb.DMatrix(X, label=y)

    # Set up parameters for xgboost
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 4,  # Depth of the trees
        'alpha': 10,  # L1 regularization
        'learning_rate': 0.01  # Learning rate
        # 'n_estimators': 100  # Number of trees
    }

    # Train the model
    xgb_model = xgb.train(params, dtrain, num_boost_round=10)
    
    # Predict the target variable on the testing data
    y_pred_all = xgb_model.predict(dall)

    return xgb_model, y_pred_all

def calc_demographic_parity_difference(dag, df, outcome_node, protected_node):


    df_fairness = df.copy()

    # Learn parameters to turn it into a Bayesian Network
    bn_fairness = BayesianNetwork(ebunch=dag.edges())
    bn_fairness.fit(data=df_fairness, estimator=BayesianEstimator)

    # Generate data and build a XGboost model
    inference = BayesianModelSampling(bn_fairness)
    sample_fairness = inference.forward_sample(size=df_fairness.shape[0], include_latents=True)

    # If the outcome node is not in the sample, then we randomly assign 1 or 0 as the outcome
    oc = list(outcome_node)[0]
    # # Get the proportion of 1 and 0 in the outcome node
    # oc_prop = df_fairness[oc].value_counts(normalize=True)
    # oc_prop_1 = oc_prop.get(1, 0)  # Proportion of 1s
    # oc_prop_0 = oc_prop.get(0, 0)  # Proportion of 0s
    # if oc not in sample_fairness.columns:
    #     # Generate the outcome randomly, but to the proportion of the outcome in the original data
    #     sample_fairness[oc] = np.random.choice([0, 1], size=len(sample_fairness), p=[oc_prop_0, oc_prop_1])

    if oc not in sample_fairness.columns:
        # Generate the outcome randomly, with equal probability
        sample_fairness[oc] = np.random.choice([0, 1], size=len(sample_fairness), p=[0.9, 0.1])
    
    # One-hot-encoding the data
    df_all, X, y = preprocess_data(sample_fairness,  outcome_node)

    # Split the data into training and testing data, 70-30 split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)

    # Feed the data trough XGBoost model and get predictions back
    xgboost_model, pred_all = build_xgboost_model(X, y, X_train,  X_test, y_train, y_test)

    # Adding predictions to the dataframe B
    sample_fairness['pred_prob'] = pred_all
    sample_fairness['pred_label'] = sample_fairness['pred_prob'].apply(lambda x: 1 if x > 0.5 else 0)

    # Calculate Demographic Parity Difference using fairlearn
    group_col = sample_fairness[protected_node]
    dpd = demographic_parity_difference(y_true=y, y_pred=sample_fairness['pred_label'], sensitive_features=group_col)

    return dpd

def calc_fairness_metric_from_dag(model_new, model_new_edges_list, protected_attribute, outcome_nodes, Y, fairness_score_baseline, fairness_sample):
    if protected_attribute in model_new_edges_list:

        fairness_sample_new = fairness_sample[model_new_edges_list]
        fairness_score_new = calc_demographic_parity_difference(dag=model_new, df=fairness_sample_new, outcome_node=outcome_nodes, protected_node=protected_attribute)
    else:
        print("The protected attribute is not in new DAG")
        fairness_score_new = fairness_score_baseline

    return fairness_score_new


def find_all_paths(graph, start, end, path=[]):
    """
    Find all paths from start node to end node using DFS in a DAG.
    """
    path = path + [start]

    # If start is the same as end, we've found a path
    if start == end:
        return [path]

    # If the start node is not in graph, return empty list
    if start not in graph:
        return []

    paths = []  # A list to store all paths
    for node in graph[start]:
        if node not in path:  # Check to avoid cycles (not necessary for DAG)
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)

    return paths


class FairnessAwareHillClimbeSearch(StructureEstimator):
    def __init__(self, data, use_cache=True, **kwargs):
        """
        Class for heuristic hill climb searches for DAGs, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.

        Parameters
        ----------
        data: pandas DataFrame object
            dataframe object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        use_caching: boolean
            If True, uses caching of score for faster computation.
            Note: Caching only works for scoring methods which are decomposable. Can
            give wrong results in case of custom scoring methods.

        References
        ----------
        Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.4.3 (page 811ff)
        """
        self.use_cache = use_cache

        super(FairnessAwareHillClimbeSearch, self).__init__(data, **kwargs)

    def _legal_operations(
        self,
        model,
        score,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
        root_nodes,
        temporal_order,
        outcome_nodes,
        protected_attribute,
        fairness_sample,
        lambda_fairness=0,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        tabu_list = set(tabu_list)

        sample_size = len(self.data)

        # Generate edges between root nodes, they will not be allowed to be added (blacklist)
        root_nodes_edges = generate_linear_combinations(root_nodes, root_nodes)

        # Generate edges between differen temporal tiers, the added edges must be in this list (whitelist)
        temporal_order_edges = []
        for i in range(len(temporal_order)-1):
            edges_temp = generate_linear_combinations(temporal_order[i], temporal_order[i+1])
            temporal_order_edges.extend(edges_temp)

        # Generate temporal_order_edges with with reversed order - these edges will not be searched
        temporal_order_edges_rev = [(x[1], x[0]) for x in temporal_order_edges]


        print("=====Model edges: ", model.edges(), "=====")

        # Calculate the baseline fairness score. Recalculate the baseline after each iteration.
        # Fairness baseline score can only be calculated if the protected attribute is in the model.
        if protected_attribute is not None:
            # Get the edges from current model as a list.
            model_edges = list(model.edges())
            # Extract all elements frmo dag0_edges into a list
            model_edges_list = [item for t in model_edges for item in t]
            # Only keep distinct elements in dag0_edges_list
            model_edges_list = list(set(model_edges_list))
            # Protected attribute should always be in the model, as we start from a Chow-Liu tree
            if protected_attribute in model_edges_list:
                if len(list(model.edges())) > 0:
                    # Only progress if the model has edges
                    model_new = model.copy()
                    model_new_edges = list(model_new.edges())
                    # Extract all elements frmo dag0_edges into a list
                    model_new_edges_list = [item for t in model_new_edges for item in t]
                    # Only keep distinct elements in dag0_edges_list
                    model_new_edges_list = list(set(model_new_edges_list))

                    # If the outcome node is not in the existing edges, add it to the new graph.
                    if not outcome_nodes.issubset(model_new_edges_list):
                        # Add an edge between a random node in model_new and the outcome node
                        model_new.add_edge(list(model_new.nodes())[0], list(outcome_nodes)[0])
                        # Also add it to the list of nodes in the new graph
                        model_new_edges_list.append(list(outcome_nodes)[0])

                    fairness_sample_new = fairness_sample[model_new_edges_list]
                    fairness_score_baseline = calc_demographic_parity_difference(dag=model_new, df=fairness_sample_new, outcome_node=outcome_nodes, protected_node=protected_attribute)

            print("fairness_score_baseline: ", fairness_score_baseline)
    
        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
            - set(temporal_order_edges_rev)
        )

        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                    (operation not in tabu_list)
                    and ((X, Y) not in black_list)
                    and ((X, Y) in white_list)
                    and ((X, Y) not in root_nodes_edges)
                    and (Y not in root_nodes)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta_nofairness = score(Y, new_parents) - score(Y, old_parents)

                        print("----Processing edge addition of: ", X, "and", Y, "----")
                        score_old = score(Y, old_parents)
                        score_new = score(Y, new_parents)
                        print("Old structure score before fairness: ", score_old)

                        if protected_attribute is not None:

                            # Get the edges from current model as a list.
                            model_edges = list(model.edges())
                            # Extract all elements frmo dag0_edges into a list
                            model_edges_list = [item for t in model_edges for item in t]
                            # Only keep distinct elements in dag0_edges_list
                            model_edges_list = list(set(model_edges_list))

                            # Initialize the fairness score for the new graph.
                            fairness_score_new = fairness_score_baseline

                            # Get the fairness matrix for the new graph
                            model_new = model.copy()
                            model_new.add_edge(X, Y)

                            # Append the edge X -> Y to the list of nodes in the new graph.
                            model_new_edges_list = model_edges_list + [X, Y]

                            # Deduplicate the list
                            model_new_edges_list = list(set(model_new_edges_list))

                            # Add the fairness baseline score to the old structure score
                            if fairness_score_baseline > 0:
                                score_old = score_old - sample_size*lambda_fairness*log(fairness_score_baseline)

                            fairness_score_new = calc_fairness_metric_from_dag(model_new, model_new_edges_list, protected_attribute, outcome_nodes, Y, fairness_score_baseline, fairness_sample)

                            print("fairness_score_baseline: ", fairness_score_baseline)
                            print("fairness_score_new: ", fairness_score_new)
                            print("Old structure score after fairness: ", score_old)
                            print("New structure score before fairness: ", score_new)
                            if fairness_score_new > 0:
                                score_new = score_new - sample_size*lambda_fairness*log(fairness_score_new)
                            print("New structure score after fairness: ", score_new)
                        else:
                            print("Adding the edge does not create new paths from root nodes to outcome nodes, and hence does not affect fairness metric.")

                        score_delta = score_new - score_old
                        print("Delta structure score with fairness: ", score_delta)
                        print("Delta structure score without fairness: ", score_delta_nofairness)
                        score_delta += structure_score("+")
                        yield (operation, score_delta)
            
        # Step 2: Get all legal operations for removing edges
        for X, Y in model.edges():
            operation = ("-", (X, Y))
                
            if (operation not in tabu_list) and ((X, Y) not in fixed_edges) and (Y not in root_nodes): 
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)

                print("----Processing edge removal of: ", X, "and", Y, "----")
                score_delta_nofairness = score(Y, new_parents) - score(Y, old_parents)

                score_old = score(Y, old_parents)
                score_new = score(Y, new_parents)
                print("Old structure score before fairness: ", score_old)

                if protected_attribute is not None:

                    # Get the fairness matrix for the new graph
                    model_new = model.copy()

                    model_new.remove_edge(X, Y)
                    # Check the nodes in the new model, remove from fairness_sample if the node is not in the new model
                    # First convert the elements in the edges in model_new into a list 
                    model_new_edges = list(model_new.edges())
                    # Extract all elements frmo dag0_edges into a list
                    model_new_edges_list = [item for t in model_new_edges for item in t]
                    # Only keep distinct elements in dag0_edges_list
                    model_new_edges_list = list(set(model_new_edges_list))

                    # Add the fairness baseline score to the old structure score
                    if fairness_score_baseline > 0:
                        score_old = score_old - sample_size*lambda_fairness*log(fairness_score_baseline)

                    fairness_score_new = calc_fairness_metric_from_dag(model_new, model_new_edges_list, protected_attribute, outcome_nodes, Y, fairness_score_baseline, fairness_sample)
                    if fairness_score_new > 0:
                        score_new = score_new - sample_size*lambda_fairness*log(fairness_score_new)
                    print("fairness_score_baseline: ", fairness_score_baseline)
                    print("fairness_score_new: ", fairness_score_new)
                    print("Old structure score after fairness: ", score_old)
                    print("New structure score before fairness: ", score_new)
                    print("New structure score after fairness: ", score_new)

                score_delta = score_new - score_old
                print("Delta structure score: ", score_delta)
                print("Delta structure score without fairness: ", score_delta_nofairness)
                score_delta += structure_score("-")
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in model.edges():

            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((X, Y) not in fixed_edges)
                    and ((Y, X) not in black_list)
                    and ((Y, X) in white_list)
                    and (X not in root_nodes)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if len(new_X_parents) <= max_indegree:
                        score_delta_nofairness = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )

                        print("----Processing edge reversal of: ", X, "and", Y, "----")

                        score_old = score(X, old_X_parents) + score(Y, old_Y_parents)
                        score_new = score(X, new_X_parents) + score(Y, new_Y_parents)

                        if protected_attribute is not None:

                            model_new = model.copy()
                            model_new.remove_edge(X, Y)
                            model_new.add_edge(Y, X)

                            model_new_edges = list(model_new.edges())
                            # Extract all elements frmo dag0_edges into a list
                            model_new_edges_list = [item for t in model_new_edges for item in t]
                            # Only keep distinct elements in dag0_edges_list
                            model_new_edges_list = list(set(model_new_edges_list))

                            # Add the fairness baseline score to the old structure score
                            if fairness_score_baseline > 0:
                                score_old = score_old - sample_size*lambda_fairness*log(fairness_score_baseline)

                            fairness_score_new = calc_fairness_metric_from_dag(model_new, model_new_edges_list, protected_attribute, outcome_nodes, Y, fairness_score_baseline, fairness_sample)
                            if fairness_score_new > 0:
                                score_new = score_new - sample_size*lambda_fairness*log(fairness_score_new)

                            print("fairness_score_baseline: ", fairness_score_baseline)
                            print("fairness_score_new: ", fairness_score_new)
                            print("Old structure score after fairness: ", score_old)
                            print("New structure score before fairness: ", score_new)
                            print("New structure score after fairness: ", score_new)
  
                        score_delta = score_new - score_old
                        print("Delta structure score: ", score_delta)
                        print("Delta structure score without fairness: ", score_delta_nofairness)

                        score_delta += structure_score("flip")
                        yield (operation, score_delta)

    def estimate(
        self,
        scoring_method="k2score",
        start_dag=None,
        fixed_edges=set(),
        tabu_length=100,
        max_indegree=None,
        black_list=None,
        white_list=None,
        root_nodes=None,
        temporal_order=None,
        outcome_nodes=None,
        protected_attribute=None,
        lambda_fairness=0,
        epsilon=1e-4,
        max_iter=1e6,
        show_progress=True,
    ):
        """
        Performs local hill climb search to estimates the `DAG` structure that
        has optimal score, according to the scoring method supplied. Starts at
        model `start_dag` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no
        parametrization.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2score, bdeuscore, bdsscore, bicscore, aicscore. Also accepts a
            custom score, but it should be an instance of `StructureScore`.

        start_dag: DAG instance
            The starting point for the local search. By default, a completely
            disconnected network is used.

        fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.
        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None
        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model when the
            number of iterations is greater than `max_iter`.

        Returns
        -------
        Estimated model: FairnessAwareHC.base.DAG
            A `DAG` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from FairnessAwareHC.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data)
        >>> best_model = est.estimate(scoring_method=BicScore(data))
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        OutEdgeView([('B', 'J'), ('A', 'J')])
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        OutEdgeView([('J', 'A'), ('B', 'J')])
        """

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        supported_methods = {
            "k2score": K2Score,
            "bdeuscore": BDeuScore,
            "bdsscore": BDsScore,
            "bicscore": BicScore,
            "aicscore": AICScore,
        }
        if (
            (
                isinstance(scoring_method, str)
                and (scoring_method.lower() not in supported_methods)
            )
        ) and (not isinstance(scoring_method, StructureScore)):
            raise ValueError(
                "scoring_method should either be one of k2score, bdeuscore, bicscore, bdsscore, aicscore, or an instance of StructureScore"
            )

        if isinstance(scoring_method, str):
            score = supported_methods[scoring_method.lower()](data=self.data)
        else:
            score = scoring_method

        if self.use_cache:
            score_fn = ScoreCache.ScoreCache(score, self.data).local_score
        else:
            score_fn = score.local_score


        # Step 1.2: Initiate a start_dag. If the user did not provide any, we will develop a chow-liu tree.
        if start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(self.variables)
            # # Add the root nodes to the start_dag
            # # Note elements in the root node cannot have edges between them unless it is in the white list.
            # if root_nodes is not None:
            #     start_dag.add_nodes_from(root_nodes)
        elif not isinstance(start_dag, DAG) or not set(start_dag.nodes()) == set(
            self.variables
        ):
            raise ValueError(
                "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
            )
        

        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            start_dag.add_edges_from(fixed_edges)
            if not nx.is_directed_acyclic_graph(start_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )

        # Step 1.4a: Check root nodes and outcome nodes
        root_nodes = set() if root_nodes is None else set(root_nodes)
        outcome_nodes = set() if outcome_nodes is None else set(outcome_nodes)
        temporal_order = set() if temporal_order is None else temporal_order

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        if max_indegree is None:
            max_indegree = float("inf")

        tabu_list = deque(maxlen=tabu_length)
        current_model = start_dag

        # Step 1.6: Sample from input data, stratified by protected attribute, for the fairness learning part.
        # We take the entire dataset for now.
        if protected_attribute is not None:
            # fairness_sample = self.data.groupby(protected_attribute, group_keys=False).apply(lambda x: x.sample(frac=0.2))
            fairness_sample = self.data.copy()
        else:
            fairness_sample = set()


        if show_progress and SHOW_PROGRESS:
            iteration = trange(int(max_iter))
        else:
            iteration = range(int(max_iter))

        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.
        for _ in iteration:
            best_operation, best_score_delta = max(
                self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
                    root_nodes,
                    temporal_order,
                    outcome_nodes,
                    protected_attribute,
                    fairness_sample,
                    lambda_fairness,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == "+":
                current_model.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1])
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == "flip":
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list.append(best_operation)
            
        # Step 3: Return if no more improvements or maximum iterations reached.
        return current_model
