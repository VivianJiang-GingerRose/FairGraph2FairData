import pandas as pd
from contextlib import redirect_stdout

from FairnessAwareHC.estimators import K2Score, BicScore
from FairnessAwareHC.estimators import FairnessAwareHillClimbSearch

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

## Set seeds
seed_bn = 123
seed_ml = 123

folder_name = 'experiments/dutch_census'

## Specify the folder name where the logs are saved.
log_folder_name = 'outputs'

def create_edges(edge_list):
    for edge in edge_list:
        print(f"d.add_edge('{edge[0]}', '{edge[1]}')")

## *******************
## Test Adult data
## *******************
df_dutch = pd.read_csv('data/data_dutch_census/dutch.csv')

print(df_dutch.shape)
# (60420, 12)

## Make sure all the columns are in the correct data type
df_dutch['age'] = df_dutch['age'].astype('object')
df_dutch['household_position'] = df_dutch['household_position'].astype('object')
df_dutch['household_size'] = df_dutch['household_size'].astype('object')
df_dutch['prev_residence_place'] = df_dutch['prev_residence_place'].astype('object')
df_dutch['citizenship'] = df_dutch['citizenship'].astype('object')
df_dutch['country_birth'] = df_dutch['country_birth'].astype('object')
df_dutch['edu_level'] = df_dutch['edu_level'].astype('object')
df_dutch['economic_status'] = df_dutch['economic_status'].astype('object')
df_dutch['cur_eco_activity'] = df_dutch['cur_eco_activity'].astype('object')
df_dutch['marital_status'] = df_dutch['marital_status'].astype('object')


## Define the outcome nodes, that is what we are using for downstream ML prediction tasks.
## Note BicScoreWithOutcome is currently hard-coded and expect a parameter called outcome_node to be defined here.
outcome_node = ['occupation']

protected_node = 'sex'



## *******************
## Build a initial graph with Chow-liu tree
## *******************
from FairnessAwareHC.estimators import TreeSearch

est = TreeSearch(df_dutch, root_node=protected_node)
dag0 = est.estimate(estimator_type="chow-liu")
create_edges(dag0.edges())

# d.add_edge('sex', 'cur_eco_activity')
# d.add_edge('cur_eco_activity', 'edu_level')
# d.add_edge('cur_eco_activity', 'economic_status')
# d.add_edge('edu_level', 'occupation')
# d.add_edge('edu_level', 'country_birth')
# d.add_edge('economic_status', 'age')
# d.add_edge('country_birth', 'citizenship')
# d.add_edge('age', 'household_position')
# d.add_edge('household_position', 'household_size')
# d.add_edge('household_position', 'marital_status')
# d.add_edge('household_position', 'prev_residence_place')


## *******************
## Run hill climb search with initial DAG with fairness constraints
## *******************
from FairnessAwareHC.models import BayesianNetwork

lambda_fairness_vals = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0]

## Iterate over lambda_fairness_vals
for i, lambda_fairness in enumerate(lambda_fairness_vals):
    
    ## Reset the initial DAG
    dag1 = BayesianNetwork()
    dag1.add_nodes_from(dag0.nodes())

    ## Add edges
    dag1.add_edge('sex', 'cur_eco_activity')
    dag1.add_edge('edu_level', 'cur_eco_activity')
    dag1.add_edge('cur_eco_activity', 'economic_status')
    dag1.add_edge('edu_level', 'occupation')
    dag1.add_edge('country_birth', 'citizenship')
    dag1.add_edge('country_birth', 'edu_level')
    dag1.add_edge('age', 'household_position')
    dag1.add_edge('age', 'economic_status')
    dag1.add_edge('household_position', 'household_size')
    dag1.add_edge('household_position', 'marital_status')
    dag1.add_edge('household_position', 'prev_residence_place')

    print("Processing lambda fairness value = ", lambda_fairness)
    file_name = f'{folder_name}/{log_folder_name}/HC_withFairness_lambda_fairness_{lambda_fairness}.txt'

    with open(file_name,"w") as f:
        with redirect_stdout(f):
            scoring_method = BicScore(data=df_dutch)
            hc = FairnessAwareHillClimbSearch(data=df_dutch)
            bn_dutch_1 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag1, protected_attribute=protected_node, lambda_fairness=lambda_fairness)
            create_edges(bn_dutch_1.edges())


## *******************
## Run hill climb search with initial DAG WITHOUT fairness constraints
## *******************
dag2 = BayesianNetwork()
dag2.add_nodes_from(dag0.nodes())

## Add edges
dag2.add_edge('sex', 'cur_eco_activity')
dag2.add_edge('edu_level', 'cur_eco_activity')
dag2.add_edge('cur_eco_activity', 'economic_status')
dag2.add_edge('edu_level', 'occupation')
dag2.add_edge('country_birth', 'citizenship')
dag2.add_edge('country_birth', 'edu_level')
dag2.add_edge('age', 'household_position')
dag2.add_edge('age', 'economic_status')
dag2.add_edge('household_position', 'household_size')
dag2.add_edge('household_position', 'marital_status')
dag2.add_edge('household_position', 'prev_residence_place')        

with open(f'{folder_name}/{log_folder_name}/HC_withoutFairness.txt',"w") as f:
    with redirect_stdout(f):
        scoring_method = BicScore(data=df_dutch)
        hc = FairnessAwareHillClimbSearch(data=df_dutch)
        bn_dutch_2 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag2)
        create_edges(bn_dutch_2.edges())