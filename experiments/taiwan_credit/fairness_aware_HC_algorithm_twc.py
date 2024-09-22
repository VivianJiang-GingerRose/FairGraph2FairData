import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import io

from FairnessAwareHC.estimators import BicScore
from FairnessAwareHC.estimators import FairnessAwareHillClimbSearch

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Set seeds
seed_bn = 123
seed_ml = 123

# 
def create_edges(edge_list):
    for edge in edge_list:
        print(f"d.add_edge('{edge[0]}', '{edge[1]}')")

# *******************
# Test with Taiwan data
# *******************
df_twc = pd.read_csv('data/data_taiwan_credit/twc_processed.csv')


# Only keep a few columns for testing
df_twc_1 = df_twc[['default_f', 'interest_cat', 'gender_cat', 'education_cat', 'marriage_cat']]


# Define the outcome nodes, that is what we are using for downstream ML prediction tasks.
# Note BicScoreWithOutcome is currently hard-coded and expect a parameter called outcome_node to be defined here.
outcome_node = ['default_f']

protected_node = 'gender_cat'



# *******************
# Build a initial graph with Chow-liu tree
# *******************
from FairnessAwareHC.estimators import TreeSearch

est = TreeSearch(df_twc_1, root_node=protected_node)
dag0 = est.estimate(estimator_type="chow-liu")
create_edges(dag0.edges())

# From Chow-liu:
# d.add_edge('gender_cat', 'interest_cat')
# d.add_edge('interest_cat', 'default_f')
# d.add_edge('default_f', 'education_cat')
# d.add_edge('education_cat', 'marriage_cat')

# We edit the graph with a causal structure knowledge
# Produce an empty DAG
from FairnessAwareHC.models import BayesianNetwork
dag1 = BayesianNetwork()
dag1.add_nodes_from(dag0.nodes())

# Add edges
dag1.add_edge('gender_cat', 'interest_cat')
dag1.add_edge('gender_cat', 'education_cat')
dag1.add_edge('interest_cat', 'default_f')
dag1.add_edge('education_cat', 'marriage_cat')


# *******************
# Run hill climb search with initial DAG and fairness constraint
# Note: lambda_fairness is set to 0.5
# Note: Prediction score cut-off: 0.6
# *******************
with open("experiments/taiwan_credit/outputs/HC_withFairness_log.txt","w") as f:
    with redirect_stdout(f):
        scoring_method = BicScore(data=df_twc_1)
        hc = FairnessAwareHillClimbSearch(data=df_twc_1)
        bn_twc_1 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag1, protected_attribute=protected_node)
        create_edges(bn_twc_1.edges())


# d.add_edge('gender_cat', 'interest_cat')
# d.add_edge('gender_cat', 'marriage_cat')
# d.add_edge('interest_cat', 'default_f')
# d.add_edge('education_cat', 'marriage_cat')
# d.add_edge('marriage_cat', 'default_f')



# *******************
# Run hill climb search with initial DAG and !!WITHOUT!! fairness constraint
# *******************
# The initial DAG dag1 gets over-written by the algorithm, so we reset it before running the algorithm again.
dag1 = BayesianNetwork()
dag1.add_nodes_from(dag0.nodes())
# Add edges
dag1.add_edge('gender_cat', 'interest_cat')
dag1.add_edge('gender_cat', 'education_cat')
dag1.add_edge('interest_cat', 'default_f')
dag1.add_edge('education_cat', 'marriage_cat')        

with open("experiments/taiwan_credit/outputs/HC_noFairness_log.txt","w") as f:
    with redirect_stdout(f):       
        scoring_method = BicScore(data=df_twc_1)
        hc = FairnessAwareHillClimbSearch(data=df_twc_1)
        bn_twc_2 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag1)
        create_edges(bn_twc_2.edges())

# Results:
# d.add_edge('gender_cat', 'interest_cat')
# d.add_edge('gender_cat', 'marriage_cat')
# d.add_edge('gender_cat', 'default_f')
# d.add_edge('interest_cat', 'default_f')
# d.add_edge('education_cat', 'marriage_cat')
