import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import io

from pgmpyVJ.estimators import K2Score, BicScore, AICScore
from pgmpyVJ.estimators import HillClimbSearchVJ

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

## Set seeds
seed_bn = 123
seed_ml = 123

folder_name = 'data_adult'

## Specify the folder name where the logs are saved.
log_folder_name = 'log_20240603'

## 
def create_edges(edge_list):
    for edge in edge_list:
        print(f"d.add_edge('{edge[0]}', '{edge[1]}')")

## *******************
## Test Adult data
## *******************
df_adult = pd.read_csv('data_adult/adult_final.csv')


# ## Only keep a subset of rows for testing
# df_adult_1 = df_adult.sample(n=5000, random_state=seed_bn)


## Define the outcome nodes, that is what we are using for downstream ML prediction tasks.
## Note BicScoreWithOutcome is currently hard-coded and expect a parameter called outcome_node to be defined here.
outcome_node = ['income_f']

protected_node = 'sex'



## *******************
## Build a initial graph with Chow-liu tree
## *******************
from pgmpyVJ.estimators import TreeSearch

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
## Edit the initial DAG 
## *******************

from pgmpyVJ.models import BayesianNetwork
dag_initial = BayesianNetwork()
dag_initial.add_nodes_from(dag0.nodes())

## Add edges
dag_initial.add_edge('sex', 'relationship_cat')
dag_initial.add_edge('relationship_cat', 'marital_status_cat')
dag_initial.add_edge('relationship_cat', 'income_f')
dag_initial.add_edge('relationship_cat', 'race_cat')
dag_initial.add_edge('marital_status_cat', 'age_cat')
dag_initial.add_edge('income_f', 'education_cat')
dag_initial.add_edge('income_f', 'capital_gain_cat')
dag_initial.add_edge('income_f', 'capital_loss_cat')
dag_initial.add_edge('race_cat', 'native_country_cat')
dag_initial.add_edge('age_cat', 'hours_per_week_cat')
dag_initial.add_edge('education_cat', 'occupation_cat')
dag_initial.add_edge('occupation_cat', 'workclass_cat')


# replacement_dict = {
#     'sex': 'Gender',
#     'relationship_cat': 'Relationship Status',
#     'marital_status_cat': 'Marital Status',
#     'race_cat':'Race',
#     'age_cat': 'Age',
#     'income_f': 'Income',
#     'native_country_cat': 'Native Country',
#     'education_cat': 'Education',
#     'capital_gain_cat': 'Capital Gain',
#     'capital_loss_cat': 'Capital Loss',
#     'workclass_cat': 'Workclass',
#     'occupation_cat': 'Occupation',
#     'hours_per_week_cat': 'Hours per Week',
# }

# code_block = """
# 'sex', 'relationship_cat', 'marital_status_cat', 'income_f', 'race_cat', 'age_cat', 'education_cat', 
# 'capital_gain_cat', 'capital_loss_cat', 'native_country_cat', 'hours_per_week_cat', 'occupation_cat', 'workclass_cat'
# d.add_edge('sex', 'relationship_cat')
# d.add_edge('sex', 'occupation_cat')
# d.add_edge('sex', 'hours_per_week_cat')
# d.add_edge('sex', 'workclass_cat')
# d.add_edge('sex', 'marital_status_cat')
# d.add_edge('sex', 'race_cat')
# d.add_edge('relationship_cat', 'marital_status_cat')
# d.add_edge('relationship_cat', 'income_f')
# d.add_edge('relationship_cat', 'race_cat')
# d.add_edge('relationship_cat', 'age_cat')
# d.add_edge('marital_status_cat', 'age_cat')
# d.add_edge('income_f', 'education_cat')
# d.add_edge('income_f', 'capital_gain_cat')
# d.add_edge('income_f', 'capital_loss_cat')
# d.add_edge('race_cat', 'native_country_cat')
# d.add_edge('age_cat', 'hours_per_week_cat')
# d.add_edge('age_cat', 'education_cat')
# d.add_edge('age_cat', 'workclass_cat')
# d.add_edge('age_cat', 'income_f')
# d.add_edge('education_cat', 'occupation_cat')
# d.add_edge('education_cat', 'native_country_cat')
# d.add_edge('capital_loss_cat', 'capital_gain_cat')
# d.add_edge('workclass_cat', 'occupation_cat')
# """

# # Applying the replacements
# for old_value, new_value in replacement_dict.items():
#     code_block = code_block.replace(old_value, new_value)

# # Print the updated code block
# print(code_block)

# # 'Gender', 'Relationship Status', 'Marital Status', 'Income', 'Race', 'Age', 'Education',
# # 'Capital Gain', 'Capital Loss', 'Native Country', 'Hours per Week', 'Occupation', 'Workclass'
# # dag_initial.add_edge('Gender', 'Relationship Status')
# # dag_initial.add_edge('Relationship Status', 'Marital Status')
# # dag_initial.add_edge('Relationship Status', 'Income')
# # dag_initial.add_edge('Relationship Status', 'Race')
# # dag_initial.add_edge('Marital Status', 'Age')
# # dag_initial.add_edge('Income', 'Education')
# # dag_initial.add_edge('Income', 'Capital Gain')
# # dag_initial.add_edge('Income', 'Capital Loss')
# # dag_initial.add_edge('Race', 'Native Country')
# # dag_initial.add_edge('Age', 'Hours per Week')
# # dag_initial.add_edge('Education', 'Occupation')
# # dag_initial.add_edge('Occupation', 'Workclass')

## *******************
## Run hill climb search with initial DAG with fairness constraints
# Note: lambda_fairness is set to 0.1
# Note: Prediction score cut-off: 0.5
## *******************
# lambda_fairness_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
lambda_fairness_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

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
            # scoring_method = BicScore(data=df_adult)
            scoring_method = AICScore(data=df_adult)
            hc = HillClimbSearchVJ(data=df_adult)
            bn_adult_1 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), start_dag=dag_initial,  outcome_nodes=outcome_node, protected_attribute=protected_node, lambda_fairness=lambda_fairness)
            create_edges(bn_adult_1.edges())


# code_block = """
# 'sex', 'relationship_cat', 'marital_status_cat', 'income_f', 'race_cat', 'age_cat', 'education_cat', 
# 'capital_gain_cat', 'capital_loss_cat', 'native_country_cat', 'hours_per_week_cat', 'occupation_cat', 'workclass_cat'
# d.add_edge('sex', 'relationship_cat')
# d.add_edge('sex', 'occupation_cat')
# d.add_edge('sex', 'hours_per_week_cat')
# d.add_edge('sex', 'race_cat')
# d.add_edge('sex', 'workclass_cat')
# d.add_edge('sex', 'age_cat')
# d.add_edge('relationship_cat', 'marital_status_cat')
# d.add_edge('relationship_cat', 'income_f')
# d.add_edge('relationship_cat', 'age_cat')
# d.add_edge('relationship_cat', 'capital_gain_cat')
# d.add_edge('relationship_cat', 'capital_loss_cat')
# d.add_edge('race_cat', 'relationship_cat')
# d.add_edge('race_cat', 'native_country_cat')
# d.add_edge('age_cat', 'marital_status_cat')
# d.add_edge('age_cat', 'hours_per_week_cat')
# d.add_edge('age_cat', 'education_cat')
# d.add_edge('age_cat', 'workclass_cat')
# d.add_edge('education_cat', 'native_country_cat')
# d.add_edge('capital_gain_cat', 'capital_loss_cat')
# d.add_edge('capital_loss_cat', 'income_f')
# d.add_edge('hours_per_week_cat', 'income_f')
# d.add_edge('occupation_cat', 'education_cat')
# d.add_edge('workclass_cat', 'occupation_cat')
# """

# # Applying the replacements
# for old_value, new_value in replacement_dict.items():
#     code_block = code_block.replace(old_value, new_value)

# # Print the updated code block
# print(code_block)


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
dag_initial.add_edge('relationship_cat', 'race_cat')
dag_initial.add_edge('marital_status_cat', 'age_cat')
dag_initial.add_edge('income_f', 'education_cat')
dag_initial.add_edge('income_f', 'capital_gain_cat')
dag_initial.add_edge('income_f', 'capital_loss_cat')
dag_initial.add_edge('race_cat', 'native_country_cat')
dag_initial.add_edge('age_cat', 'hours_per_week_cat')
dag_initial.add_edge('education_cat', 'occupation_cat')
dag_initial.add_edge('occupation_cat', 'workclass_cat')

with open(f'{folder_name}/{log_folder_name}/HC_without_Fairness.txt',"w") as f:
    with redirect_stdout(f):
        # scoring_method = BicScore(data=df_adult)
        scoring_method = AICScore(data=df_adult)
        hc = HillClimbSearchVJ(data=df_adult)
        bn_adult_2 = hc.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4), outcome_nodes=outcome_node, start_dag=dag_initial)
        create_edges(bn_adult_2.edges())

# # code_block = """
# # d.add_edge('sex', 'relationship_cat')
# # d.add_edge('sex', 'occupation_cat')
# # d.add_edge('sex', 'hours_per_week_cat')
# # d.add_edge('sex', 'race_cat')
# # d.add_edge('sex', 'workclass_cat')
# # d.add_edge('sex', 'age_cat')
# # d.add_edge('age_cat', 'marital_status_cat')
# # d.add_edge('age_cat', 'hours_per_week_cat')
# # d.add_edge('age_cat', 'education_cat')
# # d.add_edge('age_cat', 'workclass_cat')
# # d.add_edge('capital_gain_cat', 'income_f')
# # d.add_edge('capital_gain_cat', 'capital_loss_cat')
# # d.add_edge('education_cat', 'income_f')
# # d.add_edge('education_cat', 'native_country_cat')
# # d.add_edge('race_cat', 'relationship_cat')
# # d.add_edge('race_cat', 'native_country_cat')
# # d.add_edge('occupation_cat', 'education_cat')
# # d.add_edge('workclass_cat', 'occupation_cat')
# # d.add_edge('marital_status_cat', 'income_f')
# # d.add_edge('relationship_cat', 'marital_status_cat')
# # d.add_edge('relationship_cat', 'age_cat')
# # d.add_edge('relationship_cat', 'capital_gain_cat')
# # d.add_edge('relationship_cat', 'capital_loss_cat')
# # """

# # # Applying the replacements
# # for old_value, new_value in replacement_dict.items():
# #     code_block = code_block.replace(old_value, new_value)

# # # Print the updated code block
# # print(code_block)