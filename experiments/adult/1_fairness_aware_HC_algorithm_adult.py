import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import io

from FairnessAwareHC.estimators import K2Score, BicScore
from FairnessAwareHC.estimators import FairnessAwareHillClimbSearch

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

## Set seeds
seed_bn = 123
seed_ml = 123

folder_name = 'dexperiments/adult'

## Specify the folder name where the logs are saved.
log_folder_name = 'outputs'


def create_edges(edge_list):
    for edge in edge_list:
        print(f"d.add_edge('{edge[0]}', '{edge[1]}')")

## *******************
## Test Adult data
## *******************
df_adult = pd.read_csv('data/data_adult/adult_final.csv')


## Define the outcome nodes, that is what we are using for downstream ML prediction tasks.
## Note BicScoreWithOutcome is currently hard-coded and expect a parameter called outcome_node to be defined here.
outcome_node = ['income_f']

protected_node = 'sex'


## *******************
## Build a initial graph with Chow-liu tree
## *******************
from FairnessAwareHC.estimators import TreeSearch

est = TreeSearch(df_adult, root_node=protected_node)
dag0 = est.estimate(estimator_type="chow-liu")
create_edges(dag0.edges())

# d.add_edge('sex', 'relationship_cat')
# d.add_edge('relationship_cat', 'marital_status_cat')
# d.add_edge('relationship_cat', 'income_f')
# d.add_edge('relationship_cat', 'race_cat')
# d.add_edge('marital_status_cat', 'age_cat')
# d.add_edge('income_f', 'education_cat')
# d.add_edge('income_f', 'capital_gain_cat')
# d.add_edge('income_f', 'capital_loss_cat')
# d.add_edge('race_cat', 'native_country_cat')
# d.add_edge('age_cat', 'hours_per_week_cat')
# d.add_edge('education_cat', 'occupation_cat')
# d.add_edge('occupation_cat', 'workclass_cat')


## *******************
## Run hill climb search with initial DAG with fairness constraints
# Note: lambda_fairness is set to 0.1
# Note: Prediction score cut-off: 0.5
## *******************
from FairnessAwareHC.models import BayesianNetwork

lambda_fairness_vals = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

## Iterate over lambda_fairness_vals
for i, lambda_fairness in enumerate(lambda_fairness_vals):

    ## Reset the initial DAG for each run
    dag_initial = BayesianNetwork()
    dag_initial.add_nodes_from(dag0.nodes())

    ## Add edges
    dag_initial.add_edge('sex', 'relationship_cat')
    dag_initial.add_edge('relationship_cat', 'marital_status_cat')
    dag_initial.add_edge('relationship_cat', 'income_f')
    dag_initial.add_edge('race_cat', 'relationship_cat')
    dag_initial.add_edge('age_cat', 'marital_status_cat')
    dag_initial.add_edge('education_cat', 'income_f')
    dag_initial.add_edge('capital_gain_cat', 'income_f')
    dag_initial.add_edge('capital_loss_cat', 'income_f')
    dag_initial.add_edge('race_cat', 'native_country_cat')
    dag_initial.add_edge('age_cat', 'hours_per_week_cat')
    dag_initial.add_edge('education_cat', 'occupation_cat')
    dag_initial.add_edge('occupation_cat', 'workclass_cat')
    
    print("Processing lambda fairness value = ", lambda_fairness)
    file_name = f'{folder_name}/{log_folder_name}/HC_withFairness_lambda_{lambda_fairness}.txt'

    with open(file_name,"w") as f:
        with redirect_stdout(f):
            scoring_method = BicScore(data=df_adult)
            hc = FairnessAwareHillClimbSearch(data=df_adult)
            bn_adult_1 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), start_dag=dag_initial,  outcome_nodes=outcome_node, protected_attribute=protected_node, lambda_fairness=lambda_fairness)
            create_edges(bn_adult_1.edges())



## *******************
## Run hill climb search with initial DAG WITHOUT fairness constraints
## *******************
## dag_initial gets over-written by the algorithm, so we reset it before running the algorithm again.
dag_initial = BayesianNetwork()
dag_initial.add_nodes_from(dag0.nodes())

## Add edges
dag_initial.add_edge('sex', 'relationship_cat')
dag_initial.add_edge('relationship_cat', 'marital_status_cat')
dag_initial.add_edge('relationship_cat', 'income_f')
dag_initial.add_edge('race_cat', 'relationship_cat')
dag_initial.add_edge('age_cat', 'marital_status_cat')
dag_initial.add_edge('education_cat', 'income_f')
dag_initial.add_edge('capital_gain_cat', 'income_f')
dag_initial.add_edge('capital_loss_cat', 'income_f')
dag_initial.add_edge('race_cat', 'native_country_cat')
dag_initial.add_edge('age_cat', 'hours_per_week_cat')
dag_initial.add_edge('education_cat', 'occupation_cat')
dag_initial.add_edge('occupation_cat', 'workclass_cat')

with open(f'{folder_name}/{log_folder_name}/HC_without_Fairness.txt',"w") as f:
    with redirect_stdout(f):
        scoring_method = BicScore(data=df_adult)
        hc = FairnessAwareHillClimbSearch(data=df_adult)
        bn_adult_2 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag_initial)
        create_edges(bn_adult_2.edges())
