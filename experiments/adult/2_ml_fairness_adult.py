import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

## Set seeds
seed_bn = 123
seed_ml = 123

df_adult = pd.read_csv('data/data_adult/adult_final.csv')

df_adult.income_f.value_counts(normalize=True)
# 0    0.75919
# 1    0.24081

df_adult.sex.value_counts(normalize=True)
# Male      0.669205
# Female    0.330795

""" **********************************************
4. Parameter Learning
**********************************************"""
from FairnessAwareHC.models import BayesianNetwork
from FairnessAwareHC.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from FairnessAwareHC.factors.discrete import State
from FairnessAwareHC.sampling import BayesianModelSampling


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Sample 0: Initial graph
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dag0 = BayesianNetwork()
dag0.add_nodes_from(['sex', 'age_cat', 'capital_gain_cat', 'capital_loss_cat',
                    'hours_per_week_cat', 'education_cat', 'native_country_cat', 'race_cat',
                    'occupation_cat', 'workclass_cat', 'marital_status_cat',
                    'relationship_cat', 'income_f'])

## Add edges
dag0.add_edge('sex', 'relationship_cat')
dag0.add_edge('relationship_cat', 'marital_status_cat')
dag0.add_edge('relationship_cat', 'income_f')
dag0.add_edge('race_cat', 'relationship_cat')
dag0.add_edge('age_cat', 'marital_status_cat')
dag0.add_edge('education_cat', 'income_f')
dag0.add_edge('capital_gain_cat', 'income_f')
dag0.add_edge('capital_loss_cat', 'income_f')
dag0.add_edge('race_cat', 'native_country_cat')
dag0.add_edge('age_cat', 'hours_per_week_cat')
dag0.add_edge('education_cat', 'occupation_cat')
dag0.add_edge('occupation_cat', 'workclass_cat')


adult_bn_0 = BayesianNetwork(ebunch=dag0.edges())
adult_bn_0.fit(
    data=df_adult,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

## forward sample
inference = BayesianModelSampling(adult_bn_0)
adult_bn_1_sample_0 = inference.forward_sample(size=30000, include_latents=False, seed=seed_bn)

## Export the data
adult_bn_1_sample_0.to_csv('data_adult/5zz2_adult_initial_graph.csv', index=False)


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Sample 1: With fairness constraints and lambda = 0.5
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dag1 = BayesianNetwork()
dag1.add_nodes_from(['sex', 'age_cat', 'capital_gain_cat', 'capital_loss_cat',
                    'hours_per_week_cat', 'education_cat', 'native_country_cat', 'race_cat',
                    'occupation_cat', 'workclass_cat', 'marital_status_cat',
                    'relationship_cat', 'income_f'])

## Add edges
dag1.add_edge('sex', 'relationship_cat')
dag1.add_edge('sex', 'occupation_cat')
dag1.add_edge('sex', 'hours_per_week_cat')
dag1.add_edge('sex', 'race_cat')
dag1.add_edge('sex', 'workclass_cat')
dag1.add_edge('sex', 'age_cat')
dag1.add_edge('relationship_cat', 'marital_status_cat')
dag1.add_edge('relationship_cat', 'income_f')
dag1.add_edge('relationship_cat', 'age_cat')
dag1.add_edge('relationship_cat', 'capital_gain_cat')
dag1.add_edge('relationship_cat', 'capital_loss_cat')
dag1.add_edge('race_cat', 'relationship_cat')
dag1.add_edge('race_cat', 'native_country_cat')
dag1.add_edge('age_cat', 'marital_status_cat')
dag1.add_edge('age_cat', 'hours_per_week_cat')
dag1.add_edge('age_cat', 'education_cat')
dag1.add_edge('age_cat', 'workclass_cat')
dag1.add_edge('education_cat', 'native_country_cat')
dag1.add_edge('capital_gain_cat', 'capital_loss_cat')
dag1.add_edge('capital_loss_cat', 'income_f')
dag1.add_edge('hours_per_week_cat', 'income_f')
dag1.add_edge('occupation_cat', 'education_cat')
dag1.add_edge('workclass_cat', 'occupation_cat')


adult_bn_1 = BayesianNetwork(ebunch=dag1.edges())
adult_bn_1.fit(
    data=df_adult,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

## forward sample
inference = BayesianModelSampling(adult_bn_1)
adult_bn_1_sample_1 = inference.forward_sample(size=30000, include_latents=False, seed=seed_bn)

## Export the data
adult_bn_1_sample_1.to_csv('data_adult/5zz2_adult_with_fairness_lambda_0_5.csv', index=False)

##------------------------
## Generate data with larger volume and
## equal number of class 1 and class 0
##------------------------
inference = BayesianModelSampling(adult_bn_1)
adult_bn_1_sample_1_0 = inference.forward_sample(size=60000, include_latents=False, seed=seed_bn)

## Use reject sampling to sample equal number of class 1 and class 0
sample_1 = inference.rejection_sample(evidence=[State(var='income_f', state=1)], size=15000, include_latents=False, seed=seed_bn)
sample_0 = inference.rejection_sample(evidence=[State(var='income_f', state=0)], size=15000, include_latents=False, seed=seed_bn)
adult_bn_1_sample_1_1_oc = pd.concat([sample_1, sample_0])

## Check the class distribution
adult_bn_1_sample_1_1_oc.income_f.value_counts(normalize=True)
# 1    15000
# 0    15000

adult_bn_1_sample_1_1_oc.sex.value_counts(normalize=True)
# Male      0.7167
# Female    0.2833

## Select the equal number of female and male and the total number of samples is 30000
sample_1 = inference.rejection_sample(evidence=[State(var='sex', state='Female')], size=15000, include_latents=False, seed=seed_bn)
sample_0 = inference.rejection_sample(evidence=[State(var='sex', state='Male')], size=15000, include_latents=False, seed=seed_bn)
adult_bn_1_sample_1_1_sex = pd.concat([sample_1, sample_0])

## Check the class distribution
adult_bn_1_sample_1_1_sex.income_f.value_counts(normalize=True)
# 0    0.781333
# 1    .218667

adult_bn_1_sample_1_1_sex.sex.value_counts(normalize=True)
# Female    0.5
# Male      0.5

                                           
## Export the data
adult_bn_1_sample_1_0.to_csv('data_adult/5zz2_adult_double_size.csv', index=False)
adult_bn_1_sample_1_1_oc.to_csv('data_adult/5zz2_adult_equal_outcome.csv', index=False)
adult_bn_1_sample_1_1_sex.to_csv('data_adult/5zz2_adult_equal_gender.csv', index=False)

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Sample 2: With fairness constraints and lambda = 0.2
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dag1 = BayesianNetwork()
dag1.add_nodes_from(['sex', 'age_cat', 'capital_gain_cat', 'capital_loss_cat',
                    'hours_per_week_cat', 'education_cat', 'native_country_cat', 'race_cat',
                    'occupation_cat', 'workclass_cat', 'marital_status_cat',
                    'relationship_cat', 'income_f'])

## Add edges
dag1.add_edge('sex', 'relationship_cat')
dag1.add_edge('sex', 'occupation_cat')
dag1.add_edge('sex', 'hours_per_week_cat')
dag1.add_edge('sex', 'race_cat')
dag1.add_edge('sex', 'workclass_cat')
dag1.add_edge('sex', 'age_cat')
dag1.add_edge('relationship_cat', 'marital_status_cat')
dag1.add_edge('relationship_cat', 'income_f')
dag1.add_edge('relationship_cat', 'age_cat')
dag1.add_edge('relationship_cat', 'capital_gain_cat')
dag1.add_edge('relationship_cat', 'capital_loss_cat')
dag1.add_edge('race_cat', 'relationship_cat')
dag1.add_edge('race_cat', 'native_country_cat')
dag1.add_edge('age_cat', 'marital_status_cat')
dag1.add_edge('age_cat', 'hours_per_week_cat')
dag1.add_edge('age_cat', 'education_cat')
dag1.add_edge('age_cat', 'workclass_cat')
dag1.add_edge('education_cat', 'native_country_cat')
dag1.add_edge('capital_gain_cat', 'capital_loss_cat')
dag1.add_edge('capital_loss_cat', 'income_f')
dag1.add_edge('occupation_cat', 'income_f')
dag1.add_edge('occupation_cat', 'education_cat')
dag1.add_edge('workclass_cat', 'occupation_cat')


adult_bn_1 = BayesianNetwork(ebunch=dag1.edges())
adult_bn_1.fit(
    data=df_adult,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

## forward sample
inference = BayesianModelSampling(adult_bn_1)
adult_bn_1_sample_1 = inference.forward_sample(size=30000, include_latents=False, seed=seed_bn)

## Export the data
adult_bn_1_sample_1.to_csv('data_adult/5zz2_adult_with_fairness_lambda_0_2.csv', index=False)

##------------------------
## Generate data with larger volume and
## equal number of class 1 and class 0
##------------------------
inference = BayesianModelSampling(adult_bn_1)
adult_bn_1_sample_1_0 = inference.forward_sample(size=60000, include_latents=False, seed=seed_bn)

## Use reject sampling to sample equal number of class 1 and class 0
sample_1 = inference.rejection_sample(evidence=[State(var='income_f', state=1)], size=15000, include_latents=False, seed=seed_bn)
sample_0 = inference.rejection_sample(evidence=[State(var='income_f', state=0)], size=15000, include_latents=False, seed=seed_bn)
adult_bn_1_sample_1_1_oc = pd.concat([sample_1, sample_0])

## Check the class distribution
adult_bn_1_sample_1_1_oc.income_f.value_counts(normalize=True)
# 1    15000
# 0    15000

adult_bn_1_sample_1_1_oc.sex.value_counts(normalize=True)
# Male      0.7167
# Female    0.2833

## Select the equal number of female and male and the total number of samples is 30000
sample_1 = inference.rejection_sample(evidence=[State(var='sex', state='Female')], size=15000, include_latents=False, seed=seed_bn)
sample_0 = inference.rejection_sample(evidence=[State(var='sex', state='Male')], size=15000, include_latents=False, seed=seed_bn)
adult_bn_1_sample_1_1_sex = pd.concat([sample_1, sample_0])

## Check the class distribution
adult_bn_1_sample_1_1_sex.income_f.value_counts(normalize=True)
# 0    0.781333
# 1    .218667

adult_bn_1_sample_1_1_sex.sex.value_counts(normalize=True)
# Female    0.5
# Male      0.5

                                           
## Export the data
adult_bn_1_sample_1_0.to_csv('data_adult/5zz2_adult_double_size_lambda_0_2.csv', index=False)
adult_bn_1_sample_1_1_oc.to_csv('data_adult/5zz2_adult_equal_outcome_lambda_0_2.csv', index=False)
adult_bn_1_sample_1_1_sex.to_csv('data_adult/5zz2_adult_equal_gender_lambda_0_2.csv', index=False)



##-----------------------------------
## With fairness constraints - lambda = 1
##-----------------------------------
dag1 = BayesianNetwork()
dag1.add_nodes_from(['sex', 'age_cat', 'capital_gain_cat', 'capital_loss_cat',
                    'hours_per_week_cat', 'education_cat', 'native_country_cat', 'race_cat',
                    'occupation_cat', 'workclass_cat', 'marital_status_cat',
                    'relationship_cat', 'income_f'])

## Add edges
dag1.add_edge('sex', 'relationship_cat')
dag1.add_edge('sex', 'occupation_cat')
dag1.add_edge('sex', 'hours_per_week_cat')
dag1.add_edge('sex', 'race_cat')
dag1.add_edge('sex', 'workclass_cat')
dag1.add_edge('sex', 'age_cat')
dag1.add_edge('relationship_cat', 'marital_status_cat')
dag1.add_edge('relationship_cat', 'age_cat')
dag1.add_edge('relationship_cat', 'capital_gain_cat')
dag1.add_edge('relationship_cat', 'capital_loss_cat')
dag1.add_edge('race_cat', 'relationship_cat')
dag1.add_edge('race_cat', 'native_country_cat')
dag1.add_edge('age_cat', 'marital_status_cat')
dag1.add_edge('age_cat', 'hours_per_week_cat')
dag1.add_edge('age_cat', 'education_cat')
dag1.add_edge('age_cat', 'workclass_cat')
dag1.add_edge('education_cat', 'native_country_cat')
dag1.add_edge('capital_gain_cat', 'capital_loss_cat')
dag1.add_edge('capital_loss_cat', 'income_f')
dag1.add_edge('occupation_cat', 'income_f')
dag1.add_edge('occupation_cat', 'education_cat')
dag1.add_edge('workclass_cat', 'income_f')
dag1.add_edge('workclass_cat', 'occupation_cat')


adult_bn_1 = BayesianNetwork(ebunch=dag1.edges())
adult_bn_1.fit(
    data=df_adult,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

## forward sample
inference = BayesianModelSampling(adult_bn_1)
adult_bn_1_lambda_1 = inference.forward_sample(size=30000, include_latents=False, seed=seed_bn)

## Export the data
adult_bn_1_lambda_1.to_csv('data_adult/5zz2_adult_with_fairness_lambda_1.csv', index=False)

##------------------------
## Generate data with larger volume and
## equal number of class 1 and class 0
##------------------------
inference = BayesianModelSampling(adult_bn_1)
adult_bn_1_sample_1_0 = inference.forward_sample(size=60000, include_latents=False, seed=seed_bn)

## Use reject sampling to sample equal number of class 1 and class 0
sample_1 = inference.rejection_sample(evidence=[State(var='income_f', state=1)], size=15000, include_latents=False, seed=seed_bn)
sample_0 = inference.rejection_sample(evidence=[State(var='income_f', state=0)], size=15000, include_latents=False, seed=seed_bn)
adult_bn_1_sample_1_1_oc = pd.concat([sample_1, sample_0])

## Check the class distribution
adult_bn_1_sample_1_1_oc.income_f.value_counts(normalize=True)
# 1    0.5
# 0    0.5

adult_bn_1_sample_1_1_oc.sex.value_counts(normalize=True)
# Male      0.66
# Female    0.34

## Select the equal number of female and male and the total number of samples is 30000
sample_1 = inference.rejection_sample(evidence=[State(var='sex', state='Female')], size=15000, include_latents=False, seed=seed_bn)
sample_0 = inference.rejection_sample(evidence=[State(var='sex', state='Male')], size=15000, include_latents=False, seed=seed_bn)
adult_bn_1_sample_1_1_sex = pd.concat([sample_1, sample_0])

## Check the class distribution
adult_bn_1_sample_1_1_sex.income_f.value_counts(normalize=True)
# 0    0.746867
# 1    0.253133

adult_bn_1_sample_1_1_sex.sex.value_counts(normalize=True)
# Female    0.5
# Male      0.5

                                           
## Export the data
adult_bn_1_sample_1_0.to_csv('data_adult/5zz2_adult_double_size_lambda_1.csv', index=False)
adult_bn_1_sample_1_1_oc.to_csv('data_adult/5zz2_adult_equal_outcome_lambda_1.csv', index=False)
adult_bn_1_sample_1_1_sex.to_csv('data_adult/5zz2_adult_equal_gender_lambda_1.csv', index=False)


##-----------------------------------
## With fairness constraints - lambda = 5
##-----------------------------------
dag1 = BayesianNetwork()
dag1.add_nodes_from(['sex', 'age_cat', 'capital_gain_cat', 'capital_loss_cat',
                    'hours_per_week_cat', 'education_cat', 'native_country_cat', 'race_cat',
                    'occupation_cat', 'workclass_cat', 'marital_status_cat',
                    'relationship_cat', 'income_f'])

## Add edges
dag1.add_edge('sex', 'relationship_cat')
dag1.add_edge('sex', 'occupation_cat')
dag1.add_edge('sex', 'hours_per_week_cat')
dag1.add_edge('sex', 'workclass_cat')
dag1.add_edge('sex', 'age_cat')
dag1.add_edge('relationship_cat', 'marital_status_cat')
dag1.add_edge('relationship_cat', 'age_cat')
dag1.add_edge('relationship_cat', 'capital_gain_cat')
dag1.add_edge('relationship_cat', 'capital_loss_cat')
dag1.add_edge('race_cat', 'relationship_cat')
dag1.add_edge('race_cat', 'native_country_cat')
dag1.add_edge('race_cat', 'sex')
dag1.add_edge('age_cat', 'marital_status_cat')
dag1.add_edge('age_cat', 'hours_per_week_cat')
dag1.add_edge('age_cat', 'education_cat')
dag1.add_edge('age_cat', 'workclass_cat')
dag1.add_edge('education_cat', 'native_country_cat')
dag1.add_edge('capital_gain_cat', 'capital_loss_cat')
dag1.add_edge('capital_loss_cat', 'income_f')
dag1.add_edge('occupation_cat', 'income_f')
dag1.add_edge('occupation_cat', 'education_cat')
dag1.add_edge('workclass_cat', 'income_f')
dag1.add_edge('workclass_cat', 'occupation_cat')


adult_bn_1 = BayesianNetwork(ebunch=dag1.edges())
adult_bn_1.fit(
    data=df_adult,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

## forward sample
inference = BayesianModelSampling(adult_bn_1)
adult_bn_1_lambda_5 = inference.forward_sample(size=30000, include_latents=False, seed=seed_bn)

## Export the data
adult_bn_1_lambda_5.to_csv('data_adult/5zz2_adult_with_fairness_lambda_5.csv', index=False)

##-----------------------------------
## With fairness constraints - lambda = 10
##-----------------------------------
dag1 = BayesianNetwork()
dag1.add_nodes_from(['sex', 'age_cat', 'capital_gain_cat', 'capital_loss_cat',
                    'hours_per_week_cat', 'education_cat', 'native_country_cat', 'race_cat',
                    'occupation_cat', 'workclass_cat', 'marital_status_cat',
                    'relationship_cat', 'income_f'])

## Add edges
dag1.add_edge('sex', 'relationship_cat')
dag1.add_edge('sex', 'occupation_cat')
dag1.add_edge('sex', 'hours_per_week_cat')
dag1.add_edge('sex', 'workclass_cat')
dag1.add_edge('sex', 'age_cat')
dag1.add_edge('relationship_cat', 'marital_status_cat')
dag1.add_edge('relationship_cat', 'age_cat')
dag1.add_edge('relationship_cat', 'capital_gain_cat')
dag1.add_edge('relationship_cat', 'capital_loss_cat')
dag1.add_edge('race_cat', 'relationship_cat')
dag1.add_edge('race_cat', 'native_country_cat')
dag1.add_edge('race_cat', 'sex')
dag1.add_edge('age_cat', 'marital_status_cat')
dag1.add_edge('age_cat', 'hours_per_week_cat')
dag1.add_edge('age_cat', 'income_f')
dag1.add_edge('age_cat', 'education_cat')
dag1.add_edge('age_cat', 'workclass_cat')
dag1.add_edge('education_cat', 'native_country_cat')
dag1.add_edge('capital_gain_cat', 'capital_loss_cat')
dag1.add_edge('capital_loss_cat', 'income_f')
dag1.add_edge('native_country_cat', 'income_f')
dag1.add_edge('occupation_cat', 'income_f')
dag1.add_edge('occupation_cat', 'education_cat')
dag1.add_edge('workclass_cat', 'occupation_cat')


adult_bn_1 = BayesianNetwork(ebunch=dag1.edges())
adult_bn_1.fit(
    data=df_adult,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

## forward sample
inference = BayesianModelSampling(adult_bn_1)
adult_bn_1_lambda_10 = inference.forward_sample(size=30000, include_latents=False, seed=seed_bn)

## Export the data
adult_bn_1_lambda_10.to_csv('data_adult/5zz2_adult_with_fairness_lambda_10.csv', index=False)


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Sample 2: No fairness constraints
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dag2 = BayesianNetwork()
dag2.add_nodes_from(['sex', 'age_cat', 'capital_gain_cat', 'capital_loss_cat',
                    'hours_per_week_cat', 'education_cat', 'native_country_cat', 'race_cat',
                    'occupation_cat', 'workclass_cat', 'marital_status_cat',
                    'relationship_cat', 'income_f'])

## Add edges
dag2.add_edge('sex', 'relationship_cat')
dag2.add_edge('sex', 'occupation_cat')
dag2.add_edge('sex', 'hours_per_week_cat')
dag2.add_edge('sex', 'race_cat')
dag2.add_edge('sex', 'workclass_cat')
dag2.add_edge('sex', 'age_cat')
dag2.add_edge('age_cat', 'marital_status_cat')
dag2.add_edge('age_cat', 'hours_per_week_cat')
dag2.add_edge('age_cat', 'education_cat')
dag2.add_edge('age_cat', 'workclass_cat')
dag2.add_edge('capital_gain_cat', 'income_f')
dag2.add_edge('capital_gain_cat', 'capital_loss_cat')
dag2.add_edge('education_cat', 'income_f')
dag2.add_edge('education_cat', 'native_country_cat')
dag2.add_edge('race_cat', 'relationship_cat')
dag2.add_edge('race_cat', 'native_country_cat')
dag2.add_edge('occupation_cat', 'education_cat')
dag2.add_edge('workclass_cat', 'occupation_cat')
dag2.add_edge('marital_status_cat', 'income_f')
dag2.add_edge('relationship_cat', 'marital_status_cat')
dag2.add_edge('relationship_cat', 'age_cat')
dag2.add_edge('relationship_cat', 'capital_gain_cat')
dag2.add_edge('relationship_cat', 'capital_loss_cat')


adult_bn_2 = BayesianNetwork(ebunch=dag2.edges())
adult_bn_2.fit(
    data=df_adult,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

## forward sample
inference = BayesianModelSampling(adult_bn_2)
adult_bn_1_sample_2 = inference.forward_sample(size=30000, include_latents=False, seed=seed_bn)

## Export the data
adult_bn_1_sample_2.to_csv('data_adult/5zz2_adult_without_fairness.csv', index=False)



##---------------------------
## Pass this througn ML model
##---------------------------

## Import the custom built ml model functions and evaluation metrics
from run_ml_models import *

######################################
## Import data that has been preprocessed, all numerical features are converted to categorical
######################################
cols_ohe =  ['relationship_cat', 'sex', 'age_cat', 'hours_per_week_cat','education_cat', 'occupation_cat','workclass_cat', 'marital_status_cat'
             , 'capital_gain_cat', 'capital_loss_cat', 'race_cat', 'native_country_cat']
## race and native_country are not included in the model
target_name = 'income_f'

## 0.0 Import original adult data
df00_1, X00, y00 = preprocess_data(df_adult, cols_ohe, target_name)

## 99.1 Create a new dataset with sex column dropped
df01 = df_adult.drop(columns='sex')    
cols_ohe_1 =  ['relationship_cat', 'age_cat', 'hours_per_week_cat','education_cat', 'occupation_cat','workclass_cat', 'marital_status_cat'
             , 'capital_gain_cat', 'capital_loss_cat', 'race_cat', 'native_country_cat']
df01_1, X01, y01 = preprocess_data(df01, cols_ohe_1, target_name)

## 99.2 Create a new dataset with sex and marital_status_cat and  relationship_cat columns dropped
df02 = df_adult.drop(columns=['sex', 'marital_status_cat', 'relationship_cat']) 
cols_ohe_2 =  ['age_cat', 'hours_per_week_cat','education_cat', 'occupation_cat','workclass_cat', 'capital_gain_cat', 'capital_loss_cat', 'race_cat', 'native_country_cat']
df02_1, X02, y02 = preprocess_data(df02, cols_ohe_2, target_name)

## 1.0 Sampled from initial graph
df0 = pd.read_csv("data_adult/5zz2_adult_initial_graph.csv")
df0_1, X0, y0 = preprocess_data(df0, cols_ohe, target_name)

## 2. Sampled from graph without fairness constraints
df1 = pd.read_csv("data_adult/5zz2_adult_without_fairness.csv")
df1_1, X1, y1 = preprocess_data(df1, cols_ohe, target_name)

## 3. Sampled from graph with fairness constraints, with lambda = 0.5
df2 = pd.read_csv("data_adult/5zz2_adult_with_fairness_lambda_0_5.csv")
df2_1, X2, y2 = preprocess_data(df2, cols_ohe, target_name)

## 3.1 Sampled from graph with fairness constraints, with lambda = 1
df21 = pd.read_csv("data_adult/5zz2_adult_with_fairness_lambda_1.csv")
df21_1, X21, y21 = preprocess_data(df21, cols_ohe, target_name)

## 3.2 Sampled from graph with fairness constraints, with lambda = 5
df22 = pd.read_csv("data_adult/5zz2_adult_with_fairness_lambda_5.csv")
df22_1, X22, y22 = preprocess_data(df22, cols_ohe, target_name)

## 3.3 Sampled from graph with fairness constraints, with lambda = 10
df23 = pd.read_csv("data_adult/5zz2_adult_with_fairness_lambda_10.csv")
df23_1, X23, y23 = preprocess_data(df23, cols_ohe, target_name)

## 3.4 Sampled from graph with fairness constraints, with lambda = 0.2
df24 = pd.read_csv("data_adult/5zz2_adult_with_fairness_lambda_0_2.csv")
df24_1, X24, y24 = preprocess_data(df24, cols_ohe, target_name)

## 4. Sampeled from graph with fairness constraints and double the size of original dataset
df3 = pd.read_csv("data_adult/5zz2_adult_double_size.csv")
df3_1, X3, y3 = preprocess_data(df3, cols_ohe, target_name)

## 5. Sampeled from graph with fairness constraints and equal number of class 1 and class 0
df4 = pd.read_csv("data_adult/5zz2_adult_equal_outcome.csv")
df4_1, X4, y4 = preprocess_data(df4, cols_ohe, target_name)

## 6. Sampeled from graph with fairness constraints and equal number of male and female
df5 = pd.read_csv("data_adult/5zz2_adult_equal_gender.csv")
df5_1, X5, y5 = preprocess_data(df5, cols_ohe, target_name)


######################################
## Build models and evaluate fairness
######################################

##-----------------------------------
## Run~
##-----------------------------------
df_model_names = ['df00_1', 'df01_1', 'df02_1', 'df0_1', 'df1_1', 'df2_1', 'df21_1', 'df22_1', 'df23_1', 'df24_1', 'df3_1', 'df4_1', 'df5_1']

sample_names = ['original dataset'
                , 'original dataset with gender column removed'
                , 'original dataset with gender and proxies columns removed'
                , 'Initial DAG'
                , 'HC without fairness constraints'
                , 'HC with fairness constraints lambda = 0.5' 
                , 'HC with fairness constraints lambda = 1'
                , 'HC with fairness constraints lambda = 5'
                , 'HC with fairness constraints lambda = 10'
                , 'HC with fairness constraints lambda = 0.2'    
                , 'HC with fairness constraints and double the size of original dataset'
                , 'HC with fairness constraints and equal number of class 1 and class 0'
                , 'HC with fairness constraints and equal number of gender male and female'
                ]

X_train_names = ['X00',  'X01', 'X02', 'X0', 'X1', 'X2', 'X21', 'X22', 'X23', 'X24', 'X3', 'X4', 'X5']
y_train_names = ['y00',  'y01', 'y02', 'y0','y1', 'y2', 'y21', 'y22', 'y23', 'y24', 'y3', 'y4', 'y5']

## Define the datasets used for model testing. If testing data is the same as training data, then leave it as None.
X_test_names = ['X00',  'X01', 'X02', 'X0', 'X1', 'X2', 'X21', 'X22', 'X23', 'X24', 'X00', 'X00', 'X00']
y_test_names = ['y00',  'y01', 'y02', 'y0','y1', 'y2', 'y21', 'y22', 'y23', 'y24',  'y00', 'y00', 'y00']

sf_indata = ['Y', 'N', 'N', 'Y', 'Y' ,'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y']

m_results_f = pd.DataFrame()

for i, df_name in enumerate(df_model_names):

    df_model = globals()[df_name]  # Get the dataframe by name

    sample_name = sample_names[i]
    X_tr = globals()[X_train_names[i]]
    y_tr = globals()[y_train_names[i]]
    X_te = globals()[X_test_names[i]]
    y_te = globals()[y_test_names[i]]

    sf_indata_f = sf_indata[i]
    random_state=seed_ml

    print(f"Processing {sample_name}")

    sf_col_name = 'sex'
    benchmark_col = "('sex_Male',)"
    protected_col = "('sex_Female',)"

    # Call your custom function here and pass the dataframe as an argument
    result = run_samples(sample_name, df_model, target_name, X_tr, y_tr, X_te, y_te, benchmark_col, protected_col, sf_col_name, df_original=df00_1, random_state=random_state, sf_indata=sf_indata_f)

    m_results_f = pd.concat([m_results_f, result])

m_results_f.to_csv('data_adult/5zz2_adult_model_fairness_results_20240321.csv', index=False)

