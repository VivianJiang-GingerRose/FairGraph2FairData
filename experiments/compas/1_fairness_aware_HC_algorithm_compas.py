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

folder_name = 'experiments/compas'

## Specify the folder name where the logs are saved.
log_folder_name = 'outputs'

## 
def create_edges(edge_list):
    for edge in edge_list:
        print(f"d.add_edge('{edge[0]}', '{edge[1]}')")

## *******************
## Test Compas data
## *******************
df_compas = pd.read_csv('data/data_compas/compas_final.csv')

print(df_compas.shape)
# (6172, 23)

## Switch the label as not having recidivism is the disable class
df_compas['two_year_recid'] = df_compas['two_year_recid'].apply(lambda x: 1 if x == 0 else 0)

## Define the outcome nodes, that is what we are using for downstream ML prediction tasks.
## Note BicScoreWithOutcome is currently hard-coded and expect a parameter called outcome_node to be defined here.
outcome_node = ['two_year_recid']

protected_node = 'race_cat'


## *******************
## Build a initial graph with Chow-liu tree
## *******************
from FairnessAwareHC.estimators import TreeSearch

est = TreeSearch(df_compas, root_node=protected_node)
dag0 = est.estimate(estimator_type="chow-liu")
create_edges(dag0.edges())

# d.add_edge('race_cat', 'age_cat')
# d.add_edge('age_cat', 'priors_count_cat')
# d.add_edge('age_cat', 'juv_other_count_cat')
# d.add_edge('priors_count_cat', 'two_year_recid')
# d.add_edge('priors_count_cat', 'c_days_from_compas_cat')
# d.add_edge('priors_count_cat', 'juv_misd_count_cat')
# d.add_edge('priors_count_cat', 'juv_fel_count_cat')
# d.add_edge('priors_count_cat', 'c_charge_degree')
# d.add_edge('priors_count_cat', 'sex')
# d.add_edge('c_days_from_compas_cat', 'days_b_screening_arrest_cat')

## *******************
## Edit the initial DAG 
## *******************

from FairnessAwareHC.models import BayesianNetwork
dag1 = BayesianNetwork()
dag1.add_nodes_from(dag0.nodes())

## Add edges
dag1.add_edge('race_cat', 'age_cat')
dag1.add_edge('age_cat', 'priors_count_cat')
dag1.add_edge('age_cat', 'juv_other_count_cat')
dag1.add_edge('priors_count_cat', 'two_year_recid')
dag1.add_edge('priors_count_cat', 'c_days_from_compas_cat')
dag1.add_edge('priors_count_cat', 'juv_misd_count_cat')
dag1.add_edge('priors_count_cat', 'juv_fel_count_cat')
dag1.add_edge('priors_count_cat', 'c_charge_degree')
dag1.add_edge('sex', 'priors_count_cat')
dag1.add_edge('c_days_from_compas_cat', 'days_b_screening_arrest_cat')


## *******************
## Run hill climb search with initial DAG with fairness constraints
# Note: lambda_fairness is set to 16
# Note: Prediction score cut-off: 0.5
## *******************
lambda_fairness_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0]

## Iterate over lambda_fairness_vals
for i, lambda_fairness in enumerate(lambda_fairness_vals):

    ## Reset the initial DAG
    dag1 = BayesianNetwork()
    dag1.add_nodes_from(dag0.nodes())

    ## Add edges
    dag1.add_edge('race_cat', 'age_cat')
    dag1.add_edge('age_cat', 'priors_count_cat')
    dag1.add_edge('age_cat', 'juv_other_count_cat')
    dag1.add_edge('priors_count_cat', 'two_year_recid')
    dag1.add_edge('priors_count_cat', 'c_days_from_compas_cat')
    dag1.add_edge('priors_count_cat', 'juv_misd_count_cat')
    dag1.add_edge('priors_count_cat', 'juv_fel_count_cat')
    dag1.add_edge('priors_count_cat', 'c_charge_degree')
    dag1.add_edge('sex', 'priors_count_cat')
    dag1.add_edge('c_days_from_compas_cat', 'days_b_screening_arrest_cat')

    print("Processing lambda fairness value = ", lambda_fairness)
    file_name = f'{folder_name}/{log_folder_name}/HC_withFairness_log_lambda_fairness_{lambda_fairness}.txt'
    with open(file_name,"w") as f:
        with redirect_stdout(f):
            scoring_method = BicScore(data=df_compas)
            hc = FairnessAwareHillClimbSearch(data=df_compas)
            bn_compa_1 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag1, protected_attribute=protected_node, lambda_fairness=lambda_fairness)
            create_edges(bn_compa_1.edges())


## *******************
## Run hill climb search with initial DAG WITHOUT fairness constraints
## *******************
## Reset the initial DAG
dag1 = BayesianNetwork()
dag1.add_nodes_from(dag0.nodes())

## Add edges
dag1.add_edge('race_cat', 'age_cat')
dag1.add_edge('age_cat', 'priors_count_cat')
dag1.add_edge('age_cat', 'juv_other_count_cat')
dag1.add_edge('priors_count_cat', 'two_year_recid')
dag1.add_edge('priors_count_cat', 'c_days_from_compas_cat')
dag1.add_edge('priors_count_cat', 'juv_misd_count_cat')
dag1.add_edge('priors_count_cat', 'juv_fel_count_cat')
dag1.add_edge('priors_count_cat', 'c_charge_degree')
dag1.add_edge('sex', 'priors_count_cat')
dag1.add_edge('c_days_from_compas_cat', 'days_b_screening_arrest_cat')

with open(f'{folder_name}/{log_folder_name}/HC_withoutFairness.txt',"w") as f:
    with redirect_stdout(f):
        scoring_method = BicScore(data=df_compas)
        hc = FairnessAwareHillClimbSearch(data=df_compas)
        bn_compa_1 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag1)
        create_edges(bn_compa_1.edges())

