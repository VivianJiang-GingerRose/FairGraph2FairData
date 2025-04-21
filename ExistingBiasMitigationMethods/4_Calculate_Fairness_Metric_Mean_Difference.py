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
df_law_orig = pd.read_csv('data_law_school/law_school_clean_for_vj_testing.csv')
## Remove NA rows
df_law_orig = df_law_orig.dropna()
md_law_orig = calc_mean_difference(df=df_law_orig, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

#******************************************************
# BIC DPD
## Generated Data without fairness regularization
df_law_bic_dpd_no_fairness = pd.read_csv('data_law_school/log_20240616/law_without_fairness.csv')
md_law_bic_dpd_no_fairness = calc_mean_difference(df=df_law_bic_dpd_no_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness
df_law_bic_dpd_fairness = pd.read_csv('data_law_school/log_20240616/law_with_fairness_lambda_2.0.csv')
md_law_bic_dpd_fairness = calc_mean_difference(df=df_law_bic_dpd_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness and double size
df_law_bic_dpd_fairness_double_size = pd.read_csv('data_law_school/log_20240616/law_with_fairness_double_size_lambda_2.0.csv')
md_law_bic_dpd_fairness_double_size = calc_mean_difference(df=df_law_bic_dpd_fairness_double_size, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness and euqale size outcome
df_law_bic_dpd_fairness_equal_outcome = pd.read_csv('data_law_school/log_20240616/law_with_fairness_equal_outcome_lambda_2.0.csv')
md_law_bic_dpd_fairness_equal_outcome = calc_mean_difference(df=df_law_bic_dpd_fairness_equal_outcome, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness and euqale size protected attribute
df_law_bic_dpd_fairness_equal_gender = pd.read_csv('data_law_school/log_20240616/law_with_fairness_equal_protected_attr_lambda_2.0.csv')
md_law_bic_dpd_fairness_equal_gender = calc_mean_difference(df=df_law_bic_dpd_fairness_equal_gender, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

#******************************************************
# BIC EO
df_law_bic_eo_no_fairness = pd.read_csv('data_law_school/log_bic_eo_20241002/law_without_fairness.csv')
md_law_bic_eo_no_fairness = calc_mean_difference(df=df_law_bic_eo_no_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6
df_law_bic_eo_fairness = pd.read_csv('data_law_school/log_bic_eo_20241002/law_with_fairness_lambda_0.6.csv')
md_law_bic_eo_fairness = calc_mean_difference(df=df_law_bic_eo_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and double size
df_law_bic_eo_fairness_double_size = pd.read_csv('data_law_school/log_bic_eo_20241002/law_with_fairness_double_size_lambda_0.6.csv')
md_law_bic_eo_fairness_double_size = calc_mean_difference(df=df_law_bic_eo_fairness_double_size, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and euqale size outcome
df_law_bic_eo_fairness_equal_outcome = pd.read_csv('data_law_school/log_bic_eo_20241002/law_with_fairness_equal_outcome_lambda_0.6.csv')
md_law_bic_eo_fairness_equal_outcome = calc_mean_difference(df=df_law_bic_eo_fairness_equal_outcome, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and euqale size protected attribute
df_law_bic_eo_fairness_equal_gender = pd.read_csv('data_law_school/log_bic_eo_20241002/law_with_fairness_equal_protected_attr_lambda_0.6.csv')
md_law_bic_eo_fairness_equal_gender = calc_mean_difference(df=df_law_bic_eo_fairness_equal_gender, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

#******************************************************
# AIC DPD
df_law_aic_dpd_no_fairness = pd.read_csv('data_law_school/log_aic_dpd_20240811/law_without_fairness.csv')
md_law_aic_dpd_no_fairness = calc_mean_difference(df=df_law_aic_dpd_no_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6
df_law_aic_dpd_fairness = pd.read_csv('data_law_school/log_aic_dpd_20240811/law_with_fairness_lambda_1.1.csv')
md_law_aic_dpd_fairness = calc_mean_difference(df=df_law_aic_dpd_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and double size
df_law_aic_dpd_fairness_double_size = pd.read_csv('data_law_school/log_aic_dpd_20240811/law_with_fairness_double_size_lambda_1.1.csv')
md_law_aic_dpd_fairness_double_size = calc_mean_difference(df=df_law_aic_dpd_fairness_double_size, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and euqale size outcome
df_law_aic_dpd_fairness_equal_outcome = pd.read_csv('data_law_school/log_aic_dpd_20240811/law_with_fairness_equal_outcome_lambda_1.1.csv')
md_law_aic_dpd_fairness_equal_outcome = calc_mean_difference(df=df_law_aic_dpd_fairness_equal_outcome, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and euqale size protected attribute
df_law_aic_dpd_fairness_equal_gender = pd.read_csv('data_law_school/log_aic_dpd_20240811/law_with_fairness_equal_protected_attr_lambda_1.1.csv')
md_law_aic_dpd_fairness_equal_gender = calc_mean_difference(df=df_law_aic_dpd_fairness_equal_gender, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

#******************************************************
# AIC EO
df_law_aic_eo_no_fairness = pd.read_csv('data_law_school/log_aic_eo_20241110/law_without_fairness.csv')
md_law_aic_eo_no_fairness = calc_mean_difference(df=df_law_aic_eo_no_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6
df_law_aic_eo_fairness = pd.read_csv('data_law_school/log_aic_eo_20241110/law_with_fairness_lambda_1.6.csv')
md_law_aic_eo_fairness = calc_mean_difference(df=df_law_aic_eo_fairness, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and double size
df_law_aic_eo_fairness_double_size = pd.read_csv('data_law_school/log_aic_eo_20241110/law_with_fairness_double_size_lambda_1.6.csv')
md_law_aic_eo_fairness_double_size = calc_mean_difference(df=df_law_aic_eo_fairness_double_size, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and euqale size outcome
df_law_aic_eo_fairness_equal_outcome = pd.read_csv('data_law_school/log_aic_eo_20241110/law_with_fairness_equal_outcome_lambda_1.6.csv')
md_law_aic_eo_fairness_equal_outcome = calc_mean_difference(df=df_law_aic_eo_fairness_equal_outcome, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

## Generated Data with fairness lambda = 0.6 and euqale size protected attribute
df_law_aic_eo_fairness_equal_gender = pd.read_csv('data_law_school/log_aic_eo_20241110/law_with_fairness_equal_protected_attr_lambda_1.6.csv')
md_law_aic_eo_fairness_equal_gender = calc_mean_difference(df=df_law_aic_eo_fairness_equal_gender, protected='race', outcome='admit', previlaged_group='White', unprevilaged_group='Non-White')

##-------------------
# Dutch Census Data
##-------------------
## Original Data
df_dutch_orig = pd.read_csv('data_dutch_census/dutch.csv')
md_dutch_orig = calc_mean_difference(df=df_dutch_orig, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

#******************************************************
# BIC DPD
## Generated Data without fairness regularization
df_dutch_bic_dpd_no_fairness = pd.read_csv('data_dutch_census/log_20240610/dutch_without_fairness.csv')
md_dutch_bic_dpd_no_fairness = calc_mean_difference(df=df_dutch_bic_dpd_no_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness
df_dutch_bic_dpd_fairness = pd.read_csv('data_dutch_census/log_20240610/5zz3_dutch_with_fairness_lambda_0.1.csv')
md_dutch_bic_dpd_fairness = calc_mean_difference(df=df_dutch_bic_dpd_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and double size
df_dutch_bic_dpd_fairness_double_size = pd.read_csv('data_dutch_census/log_20240610/5zz3_dutch_with_fairness_double_size_lambda_0.1.csv')
md_dutch_bic_dpd_fairness_double_size = calc_mean_difference(df=df_dutch_bic_dpd_fairness_double_size, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and euqale size outcome
df_dutch_bic_dpd_fairness_equal_outcome = pd.read_csv('data_dutch_census/log_20240610/5zz3_dutch_with_fairness_equal_outcome_lambda_0.1.csv')
md_dutch_bic_dpd_fairness_equal_outcome = calc_mean_difference(df=df_dutch_bic_dpd_fairness_equal_outcome, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and euqale size protected attribute
df_dutch_bic_dpd_fairness_equal_gender = pd.read_csv('data_dutch_census/log_20240610/5zz3_dutch_with_fairness_equal_protected_attr_lambda_0.1.csv')
md_dutch_bic_dpd_fairness_equal_gender = calc_mean_difference(df=df_dutch_bic_dpd_fairness_equal_gender, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

#******************************************************
# BIC EO
## Generated Data without fairness regularization
df_dutch_bic_eo_no_fairness = pd.read_csv('data_dutch_census/log_bic_eo_20241011/dutch_without_fairness.csv')
md_dutch_bic_eo_no_fairness = calc_mean_difference(df=df_dutch_bic_eo_no_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness
df_dutch_bic_eo_fairness = pd.read_csv('data_dutch_census/log_bic_eo_20241011/dutch_with_fairness_lambda_9.0.csv')
md_dutch_bic_eo_fairness = calc_mean_difference(df=df_dutch_bic_eo_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and double size
df_dutch_bic_eo_fairness_double_size = pd.read_csv('data_dutch_census/log_bic_eo_20241011/dutch_with_fairness_double_size_lambda_9.0.csv')
md_dutch_bic_eo_fairness_double_size = calc_mean_difference(df=df_dutch_bic_eo_fairness_double_size, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and euqale size outcome
df_dutch_bic_eo_fairness_equal_outcome = pd.read_csv('data_dutch_census/log_bic_eo_20241011/dutch_with_fairness_equal_outcome_lambda_9.0.csv')
md_dutch_bic_eo_fairness_equal_outcome = calc_mean_difference(df=df_dutch_bic_eo_fairness_equal_outcome, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and euqale size protected attribute
df_dutch_bic_eo_fairness_equal_gender = pd.read_csv('data_dutch_census/log_bic_eo_20241011/dutch_with_fairness_equal_protected_attr_lambda_9.0.csv')
md_dutch_bic_eo_fairness_equal_gender = calc_mean_difference(df=df_dutch_bic_eo_fairness_equal_gender, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

#******************************************************
# AIC DPD
## Generated Data without fairness regularization
df_dutch_aic_dpd_no_fairness = pd.read_csv('data_dutch_census/log_aic_20240819/dutch_without_fairness.csv')
md_dutch_aic_dpd_no_fairness = calc_mean_difference(df=df_dutch_aic_dpd_no_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness
df_dutch_aic_dpd_fairness = pd.read_csv('data_dutch_census/log_aic_20240819/dutch_with_fairness_lambda_0.5.csv')
md_dutch_aic_dpd_fairness = calc_mean_difference(df=df_dutch_aic_dpd_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and double size
df_dutch_aic_dpd_fairness_double_size = pd.read_csv('data_dutch_census/log_aic_20240819/dutch_with_fairness_double_size_lambda_0.5.csv')
md_dutch_aic_dpd_fairness_double_size = calc_mean_difference(df=df_dutch_aic_dpd_fairness_double_size, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and euqale size outcome
df_dutch_aic_dpd_fairness_equal_outcome = pd.read_csv('data_dutch_census/log_aic_20240819/dutch_with_fairness_equal_outcome_lambda_0.5.csv')
md_dutch_aic_dpd_fairness_equal_outcome = calc_mean_difference(df=df_dutch_aic_dpd_fairness_equal_outcome, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and euqale size protected attribute
df_dutch_aic_dpd_fairness_equal_gender = pd.read_csv('data_dutch_census/log_aic_20240819/dutch_with_fairness_equal_protected_attr_lambda_0.5.csv')
md_dutch_aic_dpd_fairness_equal_gender = calc_mean_difference(df=df_dutch_aic_dpd_fairness_equal_gender, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

#******************************************************
# AIC EO
## Generated Data without fairness regularization
df_dutch_aic_eo_no_fairness = pd.read_csv('data_dutch_census/log_aic_eo_20241020/dutch_without_fairness.csv')
md_dutch_aic_eo_no_fairness = calc_mean_difference(df=df_dutch_aic_eo_no_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness
df_dutch_aic_eo_fairness = pd.read_csv('data_dutch_census/log_aic_eo_20241020/dutch_with_fairness_lambda_0.5.csv')
md_dutch_aic_eo_fairness = calc_mean_difference(df=df_dutch_aic_eo_fairness, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and double size
df_dutch_aic_eo_fairness_double_size = pd.read_csv('data_dutch_census/log_aic_eo_20241020/dutch_with_fairness_double_size_lambda_0.5.csv')
md_dutch_aic_eo_fairness_double_size = calc_mean_difference(df=df_dutch_aic_eo_fairness_double_size, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and euqale size outcome
df_dutch_aic_eo_fairness_equal_outcome = pd.read_csv('data_dutch_census/log_aic_eo_20241020/dutch_with_fairness_equal_outcome_lambda_0.5.csv')
md_dutch_aic_eo_fairness_equal_outcome = calc_mean_difference(df=df_dutch_aic_eo_fairness_equal_outcome, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')

## Generated Data with fairness and euqale size protected attribute
df_dutch_aic_eo_fairness_equal_gender = pd.read_csv('data_dutch_census/log_aic_eo_20241020/dutch_with_fairness_equal_protected_attr_lambda_0.5.csv')
md_dutch_aic_eo_fairness_equal_gender = calc_mean_difference(df=df_dutch_aic_eo_fairness_equal_gender, protected='sex', outcome='occupation', previlaged_group='male', unprevilaged_group='female')


##-------------------
# Adult Census Data
##-------------------
## Original Data
df_adult_orig = pd.read_csv('data_adult/adult_final.csv')
md_adult_orig = calc_mean_difference(df=df_adult_orig, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

#******************************************************
# BIC DPD
## Generated Data without fairness regularization
df_adult_bic_dpd_no_fairness = pd.read_csv('data_adult/log_20240603/5zz3_adult_without_fairness_v2.csv')
md_adult_bic_dpd_no_fairness = calc_mean_difference(df=df_adult_bic_dpd_no_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness
df_adult_bic_dpd_fairness = pd.read_csv('data_adult/log_20240603/5zz3_adult_with_fairness_lambda_0.8.csv')
md_adult_bic_dpd_fairness = calc_mean_difference(df=df_adult_bic_dpd_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and double size
df_adult_bic_dpd_fairness_double_size = pd.read_csv('data_adult/log_20240603/5zz3_adult_with_fairness_double_size_lambda_0.8.csv')
md_adult_bic_dpd_fairness_double_size = calc_mean_difference(df=df_adult_bic_dpd_fairness_double_size, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_adult_bic_dpd_fairness_equal_outcome = pd.read_csv('data_adult/log_20240603/5zz3_adult_with_fairness_equal_outcome_lambda_0.8.csv')
md_adult_bic_dpd_fairness_equal_outcome = calc_mean_difference(df=df_adult_bic_dpd_fairness_equal_outcome, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_adult_bic_dpd_fairness_equal_gender = pd.read_csv('data_adult/log_20240603/5zz3_adult_with_fairness_equal_protected_attr_lambda_0.8.csv')
md_adult_bic_dpd_fairness_equal_gender = calc_mean_difference(df=df_adult_bic_dpd_fairness_equal_gender, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

#******************************************************
# BIC EO
## Generated Data without fairness regularization
df_adult_bic_eo_no_fairness = pd.read_csv('data_adult/log_bic_eo_20241027/adult_without_fairness.csv')
md_adult_bic_eo_no_fairness = calc_mean_difference(df=df_adult_bic_eo_no_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness
df_adult_bic_eo_fairness = pd.read_csv('data_adult/log_bic_eo_20241027/adult_with_fairness_lambda_30.0.csv')
md_adult_bic_eo_fairness = calc_mean_difference(df=df_adult_bic_eo_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and double size
df_adult_bic_eo_fairness_double_size = pd.read_csv('data_adult/log_bic_eo_20241027/adult_with_fairness_double_size_lambda_30.0.csv')
md_adult_bic_eo_fairness_double_size = calc_mean_difference(df=df_adult_bic_eo_fairness_double_size, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_adult_bic_eo_fairness_equal_outcome = pd.read_csv('data_adult/log_bic_eo_20241027/adult_with_fairness_equal_outcome_lambda_30.0.csv')
md_adult_bic_eo_fairness_equal_outcome = calc_mean_difference(df=df_adult_bic_eo_fairness_equal_outcome, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_adult_bic_eo_fairness_equal_gender = pd.read_csv('data_adult/log_bic_eo_20241027/adult_with_fairness_equal_protected_attr_lambda_30.0.csv')
md_adult_bic_eo_fairness_equal_gender = calc_mean_difference(df=df_adult_bic_eo_fairness_equal_gender, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

#******************************************************
# AIC DPD
## Generated Data without fairness regularization
df_adult_aic_dpd_no_fairness = pd.read_csv('data_adult/log_aic_20240901/adult_without_fairness.csv')
md_adult_aic_dpd_no_fairness = calc_mean_difference(df=df_adult_aic_dpd_no_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness
df_adult_aic_dpd_fairness = pd.read_csv('data_adult/log_aic_20240901/adult_with_fairness_lambda_2.0.csv')
md_adult_aic_dpd_fairness = calc_mean_difference(df=df_adult_aic_dpd_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness and double size
df_adult_aic_dpd_fairness_double_size = pd.read_csv('data_adult/log_aic_20240901/adult_with_fairness_double_size_lambda_2.0.csv')
md_adult_aic_dpd_fairness_double_size = calc_mean_difference(df=df_adult_aic_dpd_fairness_double_size, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness  and euqale size outcome
df_adult_aic_dpd_fairness_equal_outcome = pd.read_csv('data_adult/log_aic_20240901/adult_with_fairness_equal_outcome_lambda_2.0.csv')
md_adult_aic_dpd_fairness_equal_outcome = calc_mean_difference(df=df_adult_aic_dpd_fairness_equal_outcome, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness and euqale size protected attribute
df_adult_aic_dpd_fairness_equal_gender = pd.read_csv('data_adult/log_aic_20240901/adult_with_fairness_equal_protected_attr_lambda_2.0.csv')
md_adult_aic_dpd_fairness_equal_gender = calc_mean_difference(df=df_adult_aic_dpd_fairness_equal_gender, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

#******************************************************
# AIC EO
## Generated Data without fairness regularization
df_adult_aic_eo_no_fairness = pd.read_csv('data_adult/log_aic_eo_20241103/adult_without_fairness.csv')
md_adult_aic_eo_no_fairness = calc_mean_difference(df=df_adult_aic_eo_no_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness
df_adult_aic_eo_fairness = pd.read_csv('data_adult/log_aic_eo_20241103/adult_with_fairness_lambda_0.3.csv')
md_adult_aic_eo_fairness = calc_mean_difference(df=df_adult_aic_eo_fairness, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness and double size
df_adult_aic_eo_fairness_double_size = pd.read_csv('data_adult/log_aic_eo_20241103/adult_with_fairness_double_size_lambda_0.3.csv')
md_adult_aic_eo_fairness_double_size = calc_mean_difference(df=df_adult_aic_eo_fairness_double_size, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness  and euqale size outcome
df_adult_aic_eo_fairness_equal_outcome = pd.read_csv('data_adult/log_aic_eo_20241103/adult_with_fairness_equal_outcome_lambda_0.3.csv')
md_adult_aic_eo_fairness_equal_outcome = calc_mean_difference(df=df_adult_aic_eo_fairness_equal_outcome, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')

## Generated Data with fairness and euqale size protected attribute
df_adult_aic_eo_fairness_equal_gender = pd.read_csv('data_adult/log_aic_eo_20241103/adult_with_fairness_equal_protected_attr_lambda_0.3.csv')
md_adult_aic_eo_fairness_equal_gender = calc_mean_difference(df=df_adult_aic_eo_fairness_equal_gender, protected='sex', outcome='income_f', previlaged_group='Male', unprevilaged_group='Female')


##-------------------
# Compas Data
##-------------------
## Original Data
df_compas_orig = pd.read_csv('data_compas/compas_final.csv')
## Revert the outcome label as label = 0 is the disirable outcome.
df_compas_orig['two_year_recid'] = np.where(df_compas_orig['two_year_recid'] == 0, 1, 0)
md_compas_orig = calc_mean_difference(df=df_compas_orig, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

#******************************************************
# BIC DPD
## Generated Data without fairness regularization
df_compas_bic_dpd_no_fairness = pd.read_csv('data_compas/log_20240610/compas_without_fairness.csv')
df_compas_bic_dpd_no_fairness['two_year_recid'] = np.where(df_compas_bic_dpd_no_fairness['two_year_recid'] == 0, 1, 0)
md_compas_bic_dpd_no_fairness = calc_mean_difference(df=df_compas_bic_dpd_no_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5
df_compas_bic_dpd_fairness = pd.read_csv('data_compas/log_20240610/compas_with_fairness_lambda_0.5.csv')
df_compas_bic_dpd_fairness['two_year_recid'] = np.where(df_compas_bic_dpd_fairness['two_year_recid'] == 0, 1, 0)
md_compas_bic_dpd_fairness = calc_mean_difference(df=df_compas_bic_dpd_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and double size
df_compas_bic_dpd_fairness_double_size = pd.read_csv('data_compas/log_20240610/compas_with_fairness_double_size_lambda_0.5.csv')
df_compas_bic_dpd_fairness_double_size['two_year_recid'] = np.where(df_compas_bic_dpd_fairness_double_size['two_year_recid'] == 0, 1, 0)
md_compas_bic_dpd_fairness_double_size = calc_mean_difference(df=df_compas_bic_dpd_fairness_double_size, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_compas_bic_dpd_fairness_equal_outcome = pd.read_csv('data_compas/log_20240610/compas_with_fairness_equal_outcome_lambda_0.5.csv')
df_compas_bic_dpd_fairness_equal_outcome['two_year_recid'] = np.where(df_compas_bic_dpd_fairness_equal_outcome['two_year_recid'] == 0, 1, 0)
md_compas_bic_dpd_fairness_equal_outcome = calc_mean_difference(df=df_compas_bic_dpd_fairness_equal_outcome, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_compas_bic_dpd_fairness_equal_race = pd.read_csv('data_compas/log_20240610/compas_with_fairness_equal_protected_attr_lambda_0.5.csv')
df_compas_bic_dpd_fairness_equal_race['two_year_recid'] = np.where(df_compas_bic_dpd_fairness_equal_race['two_year_recid'] == 0, 1, 0)
md_compas_bic_dpd_fairness_equal_race = calc_mean_difference(df=df_compas_bic_dpd_fairness_equal_race, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

#******************************************************
# BIC EO
## Generated Data without fairness regularization
df_compas_bic_eo_no_fairness = pd.read_csv('data_compas/log_bic_eo_20241007/compas_without_fairness.csv')
df_compas_bic_eo_no_fairness['two_year_recid'] = np.where(df_compas_bic_eo_no_fairness['two_year_recid'] == 0, 1, 0)
md_compas_bic_eo_no_fairness = calc_mean_difference(df=df_compas_bic_eo_no_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5
df_compas_bic_eo_fairness = pd.read_csv('data_compas/log_bic_eo_20241007/compas_with_fairness_lambda_0.4.csv')
df_compas_bic_eo_fairness['two_year_recid'] = np.where(df_compas_bic_eo_fairness['two_year_recid'] == 0, 1, 0)
md_compas_bic_eo_fairness = calc_mean_difference(df=df_compas_bic_eo_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and double size
df_compas_bic_eo_fairness_double_size = pd.read_csv('data_compas/log_bic_eo_20241007/compas_with_fairness_double_size_lambda_0.4.csv')
df_compas_bic_eo_fairness_double_size['two_year_recid'] = np.where(df_compas_bic_eo_fairness_double_size['two_year_recid'] == 0, 1, 0)
md_compas_bic_eo_fairness_double_size = calc_mean_difference(df=df_compas_bic_eo_fairness_double_size, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_compas_bic_eo_fairness_equal_outcome = pd.read_csv('data_compas/log_bic_eo_20241007/compas_with_fairness_equal_outcome_lambda_0.4.csv')
df_compas_bic_eo_fairness_equal_outcome['two_year_recid'] = np.where(df_compas_bic_eo_fairness_equal_outcome['two_year_recid'] == 0, 1, 0)
md_compas_bic_eo_fairness_equal_outcome = calc_mean_difference(df=df_compas_bic_eo_fairness_equal_outcome, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_compas_bic_eo_fairness_equal_race = pd.read_csv('data_compas/log_bic_eo_20241007/compas_with_fairness_equal_protected_attr_lambda_0.4.csv')
df_compas_bic_eo_fairness_equal_race['two_year_recid'] = np.where(df_compas_bic_eo_fairness_equal_race['two_year_recid'] == 0, 1, 0)
md_compas_bic_eo_fairness_equal_race = calc_mean_difference(df=df_compas_bic_eo_fairness_equal_race, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

#******************************************************
# AIC DPD
## Generated Data without fairness regularization
df_compas_aic_dpd_no_fairness = pd.read_csv('data_compas/log_aic_20240815/compas_without_fairness.csv')
df_compas_aic_dpd_no_fairness['two_year_recid'] = np.where(df_compas_aic_dpd_no_fairness['two_year_recid'] == 0, 1, 0)
md_compas_aic_dpd_no_fairness = calc_mean_difference(df=df_compas_aic_dpd_no_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5
df_compas_aic_dpd_fairness = pd.read_csv('data_compas/log_aic_20240815/compas_with_fairness_lambda_0.2.csv')
df_compas_aic_dpd_fairness['two_year_recid'] = np.where(df_compas_aic_dpd_fairness['two_year_recid'] == 0, 1, 0)
md_compas_aic_dpd_fairness = calc_mean_difference(df=df_compas_aic_dpd_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and double size
df_compas_aic_dpd_fairness_double_size = pd.read_csv('data_compas/log_aic_20240815/compas_with_fairness_double_size_lambda_0.2.csv')
df_compas_aic_dpd_fairness_double_size['two_year_recid'] = np.where(df_compas_aic_dpd_fairness_double_size['two_year_recid'] == 0, 1, 0)
md_compas_aic_dpd_fairness_double_size = calc_mean_difference(df=df_compas_aic_dpd_fairness_double_size, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_compas_aic_dpd_fairness_equal_outcome = pd.read_csv('data_compas/log_aic_20240815/compas_with_fairness_equal_outcome_lambda_0.2.csv')
df_compas_aic_dpd_fairness_equal_outcome['two_year_recid'] = np.where(df_compas_aic_dpd_fairness_equal_outcome['two_year_recid'] == 0, 1, 0)
md_compas_aic_dpd_fairness_equal_outcome = calc_mean_difference(df=df_compas_aic_dpd_fairness_equal_outcome, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_compas_aic_dpd_fairness_equal_race = pd.read_csv('data_compas/log_aic_20240815/compas_with_fairness_equal_protected_attr_lambda_0.2.csv')
df_compas_aic_dpd_fairness_equal_race['two_year_recid'] = np.where(df_compas_aic_dpd_fairness_equal_race['two_year_recid'] == 0, 1, 0)
md_compas_aic_dpd_fairness_equal_race = calc_mean_difference(df=df_compas_aic_dpd_fairness_equal_race, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

#******************************************************
# AIC EO
## Generated Data without fairness regularization
df_compas_aic_eo_no_fairness = pd.read_csv('data_compas/log_aic_eo_20241009/compas_without_fairness.csv')
df_compas_aic_eo_no_fairness['two_year_recid'] = np.where(df_compas_aic_eo_no_fairness['two_year_recid'] == 0, 1, 0)
md_compas_aic_eo_no_fairness = calc_mean_difference(df=df_compas_aic_eo_no_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5
df_compas_aic_eo_fairness = pd.read_csv('data_compas/log_aic_eo_20241009/compas_with_fairness_lambda_1.6.csv')
df_compas_aic_eo_fairness['two_year_recid'] = np.where(df_compas_aic_eo_fairness['two_year_recid'] == 0, 1, 0)
md_compas_aic_eo_fairness = calc_mean_difference(df=df_compas_aic_eo_fairness, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and double size
df_compas_aic_eo_fairness_double_size = pd.read_csv('data_compas/log_aic_eo_20241009/compas_with_fairness_double_size_lambda_1.6.csv')
df_compas_aic_eo_fairness_double_size['two_year_recid'] = np.where(df_compas_aic_eo_fairness_double_size['two_year_recid'] == 0, 1, 0)
md_compas_aic_eo_fairness_double_size = calc_mean_difference(df=df_compas_aic_eo_fairness_double_size, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size outcome
df_compas_aic_eo_fairness_equal_outcome = pd.read_csv('data_compas/log_aic_eo_20241009/compas_with_fairness_equal_outcome_lambda_1.6.csv')
df_compas_aic_eo_fairness_equal_outcome['two_year_recid'] = np.where(df_compas_aic_eo_fairness_equal_outcome['two_year_recid'] == 0, 1, 0)
md_compas_aic_eo_fairness_equal_outcome = calc_mean_difference(df=df_compas_aic_eo_fairness_equal_outcome, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')

## Generated Data with fairness lambda = 0.5 and euqale size protected attribute
df_compas_aic_eo_fairness_equal_race = pd.read_csv('data_compas/log_aic_eo_20241009/compas_with_fairness_equal_protected_attr_lambda_1.6.csv')
df_compas_aic_eo_fairness_equal_race['two_year_recid'] = np.where(df_compas_aic_eo_fairness_equal_race['two_year_recid'] == 0, 1, 0)
md_compas_aic_eo_fairness_equal_race = calc_mean_difference(df=df_compas_aic_eo_fairness_equal_race, protected='race_cat', outcome='two_year_recid', previlaged_group='white', unprevilaged_group='non_white')


##-------------------
# Combine them all together
##-------------------
df_md_f = []

# Dutch Census Data
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-DPD', 'data_type':'original data', 'mean_difference': md_dutch_orig})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-DPD', 'data_type':'generated data without fairness', 'mean_difference': md_dutch_bic_dpd_no_fairness})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness', 'mean_difference': md_dutch_bic_dpd_fairness})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and double size', 'mean_difference': md_dutch_bic_dpd_fairness_double_size})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_dutch_bic_dpd_fairness_equal_outcome})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_dutch_bic_dpd_fairness_equal_gender})

df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-EO', 'data_type':'generated data without fairness', 'mean_difference': md_dutch_bic_eo_no_fairness})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-EO', 'data_type':'generated data with fairness', 'mean_difference': md_dutch_bic_eo_fairness})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and double size', 'mean_difference': md_dutch_bic_eo_fairness_double_size})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_dutch_bic_eo_fairness_equal_outcome})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_dutch_bic_eo_fairness_equal_gender})

df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-DPD', 'data_type':'generated data without fairness', 'mean_difference': md_dutch_aic_dpd_no_fairness})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness', 'mean_difference': md_dutch_aic_dpd_fairness})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and double size', 'mean_difference': md_dutch_aic_dpd_fairness_double_size})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_dutch_aic_dpd_fairness_equal_outcome})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_dutch_aic_dpd_fairness_equal_gender})

df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-EO', 'data_type':'generated data without fairness', 'mean_difference': md_dutch_aic_eo_no_fairness})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-EO', 'data_type':'generated data with fairness', 'mean_difference': md_dutch_aic_eo_fairness})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and double size', 'mean_difference': md_dutch_aic_eo_fairness_double_size})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_dutch_aic_eo_fairness_equal_outcome})
df_md_f.append({'dataset': 'dutch_census', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_dutch_aic_eo_fairness_equal_gender})

# Adult
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-DPD', 'data_type':'original data', 'mean_difference': md_adult_orig})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-DPD', 'data_type':'generated data without fairness', 'mean_difference': md_adult_bic_dpd_no_fairness})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness', 'mean_difference': md_adult_bic_dpd_fairness})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and double size', 'mean_difference': md_adult_bic_dpd_fairness_double_size})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_adult_bic_dpd_fairness_equal_outcome})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_adult_bic_dpd_fairness_equal_gender})

df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-EO', 'data_type':'generated data without fairness', 'mean_difference': md_adult_bic_eo_no_fairness})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-EO', 'data_type':'generated data with fairness', 'mean_difference': md_adult_bic_eo_fairness})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and double size', 'mean_difference': md_adult_bic_eo_fairness_double_size})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_adult_bic_eo_fairness_equal_outcome})
df_md_f.append({'dataset': 'adult_census', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_adult_bic_eo_fairness_equal_gender})

df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-DPD', 'data_type':'generated data without fairness', 'mean_difference': md_adult_aic_dpd_no_fairness})
df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness', 'mean_difference': md_adult_aic_dpd_fairness})
df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and double size', 'mean_difference': md_adult_aic_dpd_fairness_double_size})
df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_adult_aic_dpd_fairness_equal_outcome})
df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_adult_aic_dpd_fairness_equal_gender})

df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-EO', 'data_type':'generated data without fairness', 'mean_difference': md_adult_aic_eo_no_fairness})
df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-EO', 'data_type':'generated data with fairness', 'mean_difference': md_adult_aic_eo_fairness})
df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and double size', 'mean_difference': md_adult_aic_eo_fairness_double_size})
df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_adult_aic_eo_fairness_equal_outcome})
df_md_f.append({'dataset': 'adult_census', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_adult_aic_eo_fairness_equal_gender})

# Compas
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-DPD', 'data_type':'original data', 'mean_difference': md_compas_orig})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-DPD', 'data_type':'generated data without fairness', 'mean_difference': md_compas_bic_dpd_no_fairness})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness', 'mean_difference': md_compas_bic_dpd_fairness})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and double size', 'mean_difference': md_compas_bic_dpd_fairness_double_size})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_compas_bic_dpd_fairness_equal_outcome})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_compas_bic_dpd_fairness_equal_race})

df_md_f.append({'dataset': 'compas', 'parameters':'BIC-EO', 'data_type':'generated data without fairness', 'mean_difference': md_compas_bic_eo_no_fairness})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-EO', 'data_type':'generated data with fairness', 'mean_difference': md_compas_bic_eo_fairness})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and double size', 'mean_difference': md_compas_bic_eo_fairness_double_size})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_compas_bic_eo_fairness_equal_outcome})
df_md_f.append({'dataset': 'compas', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_compas_bic_eo_fairness_equal_race})

df_md_f.append({'dataset': 'compas', 'parameters':'AIC-DPD', 'data_type':'generated data without fairness', 'mean_difference': md_compas_aic_dpd_no_fairness})
df_md_f.append({'dataset': 'compas', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness', 'mean_difference': md_compas_aic_dpd_fairness})
df_md_f.append({'dataset': 'compas', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and double size', 'mean_difference': md_compas_aic_dpd_fairness_double_size})
df_md_f.append({'dataset': 'compas', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_compas_aic_dpd_fairness_equal_outcome})
df_md_f.append({'dataset': 'compas', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_compas_aic_dpd_fairness_equal_race})

df_md_f.append({'dataset': 'compas', 'parameters':'AIC-EO', 'data_type':'generated data without fairness', 'mean_difference': md_compas_aic_eo_no_fairness})
df_md_f.append({'dataset': 'compas', 'parameters':'AIC-EO', 'data_type':'generated data with fairness', 'mean_difference': md_compas_aic_eo_fairness})
df_md_f.append({'dataset': 'compas', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and double size', 'mean_difference': md_compas_aic_eo_fairness_double_size})
df_md_f.append({'dataset': 'compas', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_compas_aic_eo_fairness_equal_outcome})
df_md_f.append({'dataset': 'compas', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_compas_aic_eo_fairness_equal_race})
                
# Law School
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-DPD', 'data_type':'original data', 'mean_difference': md_law_orig})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-DPD', 'data_type':'generated data without fairness', 'mean_difference': md_law_bic_dpd_no_fairness})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness', 'mean_difference': md_law_bic_dpd_fairness})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and double size', 'mean_difference': md_law_bic_dpd_fairness_double_size})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_law_bic_dpd_fairness_equal_outcome})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-DPD', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_law_bic_dpd_fairness_equal_gender})

df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-EO', 'data_type':'generated data without fairness', 'mean_difference': md_law_bic_eo_no_fairness})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-EO', 'data_type':'generated data with fairness', 'mean_difference': md_law_bic_eo_fairness})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and double size', 'mean_difference': md_law_bic_eo_fairness_double_size})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_law_bic_eo_fairness_equal_outcome})
df_md_f.append({'dataset': 'law_school', 'parameters':'BIC-EO', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_law_bic_eo_fairness_equal_gender})

df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-DPD', 'data_type':'generated data without fairness', 'mean_difference': md_law_aic_dpd_no_fairness})
df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness', 'mean_difference': md_law_aic_dpd_fairness})
df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and double size', 'mean_difference': md_law_aic_dpd_fairness_double_size})
df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_law_aic_dpd_fairness_equal_outcome})
df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-DPD', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_law_aic_dpd_fairness_equal_gender})

df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-EO', 'data_type':'generated data without fairness', 'mean_difference': md_law_aic_eo_no_fairness})
df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-EO', 'data_type':'generated data with fairness', 'mean_difference': md_law_aic_eo_fairness})
df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and double size', 'mean_difference': md_law_aic_eo_fairness_double_size})
df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and euqale size outcome', 'mean_difference': md_law_aic_eo_fairness_equal_outcome})
df_md_f.append({'dataset': 'law_school', 'parameters':'AIC-EO', 'data_type':'generated data with fairness and euqale size protected attribute', 'mean_difference': md_law_aic_eo_fairness_equal_gender})

df_md_f = pd.DataFrame(df_md_f)                
df_md_f.to_csv('results_bias_mitigation/mean_difference_extended_paper_20250225.csv', index=False)
                
                