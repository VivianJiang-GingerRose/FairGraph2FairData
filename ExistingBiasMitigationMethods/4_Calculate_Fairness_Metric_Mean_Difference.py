import pandas as pd
import numpy as np


def calc_mean_difference(df, protected, outcome, previlaged_group, unprevilaged_group):
    """
    This function calculates the mean difference between the protected and unprotected groups
    :param df: dataframe
    :param protected: protected attribute
    :param outcome: outcome variable
    :param previlaged_group: previlaged group
    :param unprevilaged_group: unprevilaged group
    :return: mean difference
    """
    protected_group = df[df[protected] == previlaged_group]
    unprotected_group = df[df[protected] == unprevilaged_group]

    outcome_1_protected = protected_group[protected_group[outcome]==1].shape[0]
    outcome_1_unprotected = unprotected_group[unprotected_group[outcome]==1].shape[0]

    md = abs(round(outcome_1_protected/protected_group.shape[0] - outcome_1_unprotected/unprotected_group.shape[0], 4))

    return md

##-------------------
# Law School Data
##-------------------
## Original Data
df_law_orig = pd.read_csv('data/data_law_school/law_school_clean_processed.csv')
## Remove NA rows
df_law_orig = df_law_orig.dropna()
md_law_orig = calc_mean_difference(df=df_law_orig, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data without fairness regularization
df_law_no_fairness = pd.read_csv('data/data_law_school/log_20240616/law_without_fairness.csv')
md_law_no_fairness = calc_mean_difference(df=df_law_no_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.5
df_law_fairness_2_0 = pd.read_csv('data/data_law_school/log_20240616/law_with_fairness_lambda_2.0.csv')
md_law_fairness_2_0 = calc_mean_difference(df=df_law_fairness_2_0, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.5 and double size
df_law_fairness_2_0_double_size = pd.read_csv('data/data_law_school/log_20240616/law_with_fairness_double_size_lambda_2.0.csv')
md_law_fairness_2_0_double_size = calc_mean_difference(df=df_law_fairness_2_0_double_size, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_law_fairness_2_0_equal_outcome = pd.read_csv('data/data_law_school/log_20240616/law_with_fairness_equal_outcome_lambda_2.0.csv')
md_law_fairness_2_0_equal_outcome = calc_mean_difference(df=df_law_fairness_2_0_equal_outcome, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_law_fairness_2_0_equal_gender = pd.read_csv('data/data_law_school/log_20240616/law_with_fairness_equal_protected_attr_lambda_2.0.csv')
md_law_fairness_2_0_equal_gender = calc_mean_difference(df=df_law_fairness_2_0_equal_gender, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')


##-------------------
# Dutch Census Data
##-------------------
## Original Data
df_dutch_orig = pd.read_csv('data/data_dutch_census/dutch.csv')
md_dutch_orig = calc_mean_difference(df=df_dutch_orig, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data without fairness regularization
df_dutch_no_fairness = pd.read_csv('data/data_dutch_census/log_20240610/dutch_without_fairness.csv')
md_dutch_no_fairness = calc_mean_difference(df=df_dutch_no_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness lambda = 1.1
df_dutch_fairness_0_1 = pd.read_csv('data/data_dutch_census/log_20240610/5zz3_dutch_with_fairness_lambda_0.1.csv')
md_dutch_fairness_0_1 = calc_mean_difference(df=df_dutch_fairness_0_1, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness lambda = 1.1 and double size
df_dutch_fairness_0_1_double_size = pd.read_csv('data/data_dutch_census/log_20240610/5zz3_dutch_with_fairness_double_size_lambda_0.1.csv')
md_dutch_fairness_0_1_double_size = calc_mean_difference(df=df_dutch_fairness_0_1_double_size, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_dutch_fairness_0_1_equal_outcome = pd.read_csv('data/data_dutch_census/log_20240610/5zz3_dutch_with_fairness_equal_outcome_lambda_0.1.csv')
md_dutch_fairness_0_1_equal_outcome = calc_mean_difference(df=df_dutch_fairness_0_1_equal_outcome, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_dutch_fairness_0_1_equal_gender = pd.read_csv('data/data_dutch_census/log_20240610/5zz3_dutch_with_fairness_equal_protected_attr_lambda_0.1.csv')
md_dutch_fairness_0_1_equal_gender = calc_mean_difference(df=df_dutch_fairness_0_1_equal_gender, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')


##-------------------
# Adult Census Data
##-------------------
## Original Data
df_adult_orig = pd.read_csv('data/data_adult/adult_final.csv')
md_adult_orig = calc_mean_difference(df=df_adult_orig, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data without fairness regularization
df_adult_no_fairness = pd.read_csv('data/data_adult/log_20240603/5zz3_adult_without_fairness_v2.csv')
md_adult_no_fairness = calc_mean_difference(df=df_adult_no_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5
df_adult_fairness_0_8 = pd.read_csv('data/data_adult/log_20240603/5zz3_adult_with_fairness_lambda_0.8.csv')
md_adult_fairness_0_8 = calc_mean_difference(df=df_adult_fairness_0_8, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and double size
df_adult_fairness_0_8_double_size = pd.read_csv('data/data_adult/log_20240603/5zz3_adult_with_fairness_double_size_lambda_0.8.csv')
md_adult_fairness_0_8_double_size = calc_mean_difference(df=df_adult_fairness_0_8_double_size, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_adult_fairness_0_8_equal_outcome = pd.read_csv('data/data_adult/log_20240603/5zz3_adult_with_fairness_equal_outcome_lambda_0.8.csv')
md_adult_fairness_0_8_equal_outcome = calc_mean_difference(df=df_adult_fairness_0_8_equal_outcome, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_adult_fairness_0_8_equal_gender = pd.read_csv('data/data_adult/log_20240603/5zz3_adult_with_fairness_equal_protected_attr_lambda_0.8.csv')
md_adult_fairness_0_8_equal_gender = calc_mean_difference(df=df_adult_fairness_0_8_equal_gender, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')


##-------------------
# Compas Data
##-------------------
## Original Data
df_compas_orig = pd.read_csv('data/data_compas/compas_final.csv')
## Revert the outcome label as label = 0 is the disirable outcome.
df_compas_orig['two_year_recid'] = np.where(df_compas_orig['two_year_recid'] == 0, 1, 0)
md_compas_orig = calc_mean_difference(df=df_compas_orig, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data without fairness regularization
df_compas_no_fairness = pd.read_csv('data/data_compas/log_20240610/compas_without_fairness.csv')
df_compas_no_fairness['two_year_recid'] = np.where(df_compas_no_fairness['two_year_recid'] == 0, 1, 0)
md_compas_no_fairness = calc_mean_difference(df=df_compas_no_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5
df_compas_fairness_50 = pd.read_csv('data/data_compas/log_20240610/compas_with_fairness_lambda_50.0.csv')
df_compas_fairness_50['two_year_recid'] = np.where(df_compas_fairness_50['two_year_recid'] == 0, 1, 0)
md_compas_fairness_50 = calc_mean_difference(df=df_compas_fairness_50, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and double size
df_compas_fairness_50_double_size = pd.read_csv('data/data_compas/log_20240610/compas_with_fairness_double_size_lambda_50.0.csv')
df_compas_fairness_50_double_size['two_year_recid'] = np.where(df_compas_fairness_50_double_size['two_year_recid'] == 0, 1, 0)
md_compas_fairness_50_double_size = calc_mean_difference(df=df_compas_fairness_50_double_size, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_compas_fairness_50_equal_outcome = pd.read_csv('data/data_compas/log_20240610/compas_with_fairness_equal_outcome_lambda_50.0.csv')
df_compas_fairness_50_equal_outcome['two_year_recid'] = np.where(df_compas_fairness_50_equal_outcome['two_year_recid'] == 0, 1, 0)
md_compas_fairness_50_equal_outcome = calc_mean_difference(df=df_compas_fairness_50_equal_outcome, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_compas_fairness_50_equal_race = pd.read_csv('data/data_compas/log_20240610/compas_with_fairness_equal_protected_attr_lambda_50.0.csv')
df_compas_fairness_50_equal_race['two_year_recid'] = np.where(df_compas_fairness_50_equal_race['two_year_recid'] == 0, 1, 0)
md_compas_fairness_50_equal_race = calc_mean_difference(df=df_compas_fairness_50_equal_race, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')


##-------------------
# Combine them all together
##-------------------
df_md_f = []

df_md_f.append({'dataset': 'dutch_census', 'data_type':'original data', 'mean_difference': md_dutch_orig})
df_md_f.append({'dataset': 'dutch_census', 'data_type':'generated data without fairness', 'mean_difference': md_dutch_no_fairness})
df_md_f.append({'dataset': 'dutch_census', 'data_type':'generated data with fairness lambda = 0.1', 'mean_difference': md_dutch_fairness_0_1})
df_md_f.append({'dataset': 'dutch_census', 'data_type':'generated data with fairness lambda = 0.1 and double size', 'mean_difference': md_dutch_fairness_0_1_double_size})
df_md_f.append({'dataset': 'dutch_census', 'data_type':'generated data with fairness lambda = 0.1 and euqale size outcome', 'mean_difference': md_dutch_fairness_0_1_equal_outcome})
df_md_f.append({'dataset': 'dutch_census', 'data_type':'generated data with fairness lambda = 0.1 and euqale size protected attribute', 'mean_difference': md_dutch_fairness_0_1_equal_gender})

df_md_f.append({'dataset': 'adult_census', 'data_type':'original data', 'mean_difference': md_adult_orig})
df_md_f.append({'dataset': 'adult_census', 'data_type':'generated data without fairness', 'mean_difference': md_adult_no_fairness})
df_md_f.append({'dataset': 'adult_census', 'data_type':'generated data with fairness lambda = 0.8', 'mean_difference': md_adult_fairness_0_8})
df_md_f.append({'dataset': 'adult_census', 'data_type':'generated data with fairness lambda = 0.8 and double size', 'mean_difference': md_adult_fairness_0_8_double_size})
df_md_f.append({'dataset': 'adult_census', 'data_type':'generated data with fairness lambda = 0.8 and euqale size outcome', 'mean_difference': md_adult_fairness_0_8_equal_outcome})
df_md_f.append({'dataset': 'adult_census', 'data_type':'generated data with fairness lambda = 0.8 and euqale size protected attribute', 'mean_difference': md_adult_fairness_0_8_equal_gender})

df_md_f.append({'dataset': 'compas', 'data_type':'original data', 'mean_difference': md_compas_orig})
df_md_f.append({'dataset': 'compas', 'data_type':'generated data without fairness', 'mean_difference': md_compas_no_fairness})
df_md_f.append({'dataset': 'compas', 'data_type':'generated data with fairness lambda = 50', 'mean_difference': md_compas_fairness_50})
df_md_f.append({'dataset': 'compas', 'data_type':'generated data with fairness lambda = 50 and double size', 'mean_difference': md_compas_fairness_50_double_size})
df_md_f.append({'dataset': 'compas', 'data_type':'generated data with fairness lambda = 50 and euqale size outcome', 'mean_difference': md_compas_fairness_50_equal_outcome})
df_md_f.append({'dataset': 'compas', 'data_type':'generated data with fairness lambda = 50 and euqale size protected attribute', 'mean_difference': md_compas_fairness_50_equal_race})

df_md_f.append({'dataset': 'law_school', 'data_type':'original data', 'mean_difference': md_law_orig})
df_md_f.append({'dataset': 'law_school', 'data_type':'generated data without fairness', 'mean_difference': md_law_no_fairness})
df_md_f.append({'dataset': 'law_school', 'data_type':'generated data with fairness lambda = 2.0', 'mean_difference': md_law_fairness_2_0})
df_md_f.append({'dataset': 'law_school', 'data_type':'generated data with fairness lambda = 2.0 and double size', 'mean_difference': md_law_fairness_2_0_double_size})
df_md_f.append({'dataset': 'law_school', 'data_type':'generated data with fairness lambda = 2.0 and euqale size outcome', 'mean_difference': md_law_fairness_2_0_equal_outcome})
df_md_f.append({'dataset': 'law_school', 'data_type':'generated data with fairness lambda = 2.0 and euqale size protected attribute', 'mean_difference': md_law_fairness_2_0})


df_md_f = pd.DataFrame(df_md_f)                
df_md_f.to_csv('results/mean_difference_20240621.csv', index=False)
                
                