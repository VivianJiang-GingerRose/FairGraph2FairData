import pandas as pd
from contextlib import redirect_stdout

from FairnessAwareHC.estimators import K2Score, BicScore, AICScore
from FairnessAwareHC.estimators import FairnessAwareHillClimbSearch

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

## Set seeds
seed_bn = 123
seed_ml = 123

folder_name = 'experiments/law_school'

## Specify the folder name where the logs are saved.
log_folder_name = 'outputs'


## *******************
## Test law data
## *******************
df_law = pd.read_csv('data/data_law_school/law_school_clean_processed.csv')

print(df_law.shape)
# (20527, 6)

## Admission rate by race
df_law.groupby(by='race').admit.mean()
# Non-White    0.003951
# White        0.010211

## Make sure all the columns are in the correct data type
df_law['fam_inc'] = df_law['fam_inc'].astype('object')
df_law['male'] = df_law['male'].astype('object')


## Define the outcome nodes, that is what we are using for downstream ML prediction tasks.
## Note BicScoreWithOutcome is currently hard-coded and expect a parameter called outcome_node to be defined here.
outcome_node = ['admit']

protected_node = 'race'


## *******************
## Build a initial graph with Chow-liu tree
## *******************
from FairnessAwareHC.estimators import TreeSearch

est = TreeSearch(df_law, root_node=protected_node)
dag0 = est.estimate(estimator_type="chow-liu")
create_edges(dag0.edges())

# d.add_edge('race', 'lsat_bin')
# d.add_edge('race', 'fam_inc')
# d.add_edge('lsat_bin', 'ugpa_bin')
# d.add_edge('lsat_bin', 'admit')
# d.add_edge('ugpa_bin', 'male')

## *******************
## Edit the initial DAG 
## *******************

from FairnessAwareHC.models import BayesianNetwork
dag1 = BayesianNetwork()
dag1.add_nodes_from(dag0.nodes())

## Add edges
dag1.add_edge('race', 'lsat_bin')
dag1.add_edge('race', 'fam_inc')
dag1.add_edge('lsat_bin', 'ugpa_bin')
dag1.add_edge('lsat_bin', 'admit')
dag1.add_edge('ugpa_bin', 'male')


## *******************
## Run hill climb search with initial DAG with fairness constraints
# Note: lambda_fairness is set to 16
# Note: Prediction score cut-off: 0.5
## *******************
lambda_fairness_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

## Iterate over lambda_fairness_vals
for i, lambda_fairness in enumerate(lambda_fairness_vals):

    ## Reset the initial DAG
    dag1 = BayesianNetwork()
    dag1.add_nodes_from(dag0.nodes())

    ## Add edges
    dag1.add_edge('race', 'lsat_bin')
    dag1.add_edge('race', 'fam_inc')
    dag1.add_edge('lsat_bin', 'ugpa_bin')
    dag1.add_edge('lsat_bin', 'admit')
    dag1.add_edge('ugpa_bin', 'male')

    print("Processing lambda fairness value = ", lambda_fairness)
    file_name = f'{folder_name}/{log_folder_name}/HC_withFairness_log_lambda_fairness_{lambda_fairness}.txt'
    with open(file_name,"w") as f:
        with redirect_stdout(f):
            scoring_method = BicScore(data=df_law)
            hc = FairnessAwareHillClimbSearch(data=df_law)
            bn_law_1 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag1, protected_attribute=protected_node, lambda_fairness=lambda_fairness)
            create_edges(bn_law_1.edges())


## *******************
## Run hill climb search with initial DAG WITHOUT fairness constraints
## *******************
with open(f'{folder_name}/{log_folder_name}/HC_withoutFairness.txt',"w") as f:
    with redirect_stdout(f):
        scoring_method = BicScore(data=df_law)
        hc = FairnessAwareHillClimbSearch(data=df_law)
        bn_law_1 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag1)
        create_edges(bn_law_1.edges())