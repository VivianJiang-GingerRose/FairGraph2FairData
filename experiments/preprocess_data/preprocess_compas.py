import os
import sys
import copy
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

df_compas = pd.read_csv("data/data_compas/compas-scores-two-years_clean.csv")
print(df_compas.shape)
# (6172, 53)

cols_to_drop = ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob','age','c_case_number', 'c_offense_date', 'c_arrest_date', 'c_charge_desc',
                'r_case_number','r_offense_date','r_charge_desc','violent_recid','vr_case_number', 'vr_offense_date', 'vr_charge_desc',
                'type_of_assessment','score_text', 'screening_date','v_type_of_assessment','v_score_text','v_screening_date','in_custody', 'out_custody', 
                'c_jail_in', 'c_jail_out','r_jail_in', 'r_jail_out']

df_compas = df_compas.drop(columns=cols_to_drop)

## Rename decile_score.1 to decile_score_1
df_compas = df_compas.rename(columns={'decile_score.1': 'decile_score_1'})
df_compas = df_compas.rename(columns={'priors_count.1': 'priors_count_1'})

## Categorise numerical data
df_compas_1 = df_compas.copy()

df_compas_1['race_cat'] = np.where(df_compas_1['race'] == 'Caucasian', 'white', 'non_white')
df_compas_1['juv_fel_count_cat'] = np.where(df_compas_1['juv_fel_count'] > 0, 'juv_fel_count_1+', 'juv_fel_count_0')
df_compas_1['decile_score_cat'] = pd.cut(df_compas_1['decile_score'], bins=[1, 2, 3, 5, 7, 10000], labels=['decile_score_1', 'decile_score_2-3', 'decile_score_4-5', 'decile_score_6-7', 'decile_score_7+']).values.add_categories('missing')
df_compas_1['juv_misd_count_cat'] = pd.cut(df_compas_1['juv_misd_count'], bins=[-1, 0, 1, 20000], labels=['juv_misd_count_0', 'juv_misd_count_1','cjuv_misd_count_2+'])
df_compas_1['juv_other_count_cat'] = pd.cut(df_compas_1['juv_other_count'], bins=[-1, 0, 1, 20000], labels=['juv_other_count_0', 'juv_other_count_1','juv_other_count_2+'])
df_compas_1['priors_count_cat'] = pd.cut(df_compas_1['priors_count'], bins=[-1, 0, 1, 3, 6, 9, 15, 10000], labels=['priors_count_0','priors_count_1','priors_count_2-3','priors_count_4-6','priors_count_7-9','priors_count_10-15', 'priors_count_15+'])
df_compas_1['days_b_screening_arrest_cat'] = pd.cut(df_compas_1['days_b_screening_arrest'], bins=[-1000, -3, -2, -1, 0, 1000], labels=['days_b_screening_arrest_lt_-3', 'days_b_screening_arrest_-2', 'days_b_screening_arrest_-1', 'days_b_screening_arrest_0', 'days_b_screening_arrest_1+'])
df_compas_1['c_days_from_compas_cat'] = pd.cut(df_compas_1['c_days_from_compas'], bins=[-1, 0, 1, 7, 14, 21, 28, 1000], labels=['c_days_from_compas_0','c_days_from_compas_1','c_days_from_compas_2-7','c_days_from_compas_8-14','c_days_from_compas_15-21','c_days_from_compas_22-28', 'c_days_from_compas_29+']).values.add_categories('missing')
df_compas_1['r_days_from_arrest_cat'] = pd.cut(df_compas_1['r_days_from_arrest'], bins=[-1, 0, 1, 1000], labels=['r_days_from_arrest_0','r_days_from_arrest_1','r_days_from_arrest_1+']).values.add_categories('missing')
df_compas_1['start_cat'] = pd.cut(df_compas_1['start'], bins=[-1, 0, 1, 2, 5, 10, 20, 30, 50, 1000], labels=['start_0','start_1','start_2','start_3-5','start_6-10','start_11-20','start_21-30','start_31-50','start_51+'])
df_compas_1['end_cat'] = pd.cut(df_compas_1['end'], bins=[-1, 100, 300, 500, 1000, 10000000], labels=['end_1-100','end_100-300','end_301-500','end_501-1000','end_1000+'])

## Overwrite missing working class and occupation with "Unknown"
df_compas_1.loc[df_compas_1["r_charge_degree"].isna(), "r_charge_degree"] = "missing"
df_compas_1.loc[df_compas_1["r_days_from_arrest_cat"].isna(), "r_days_from_arrest_cat"] = "missing"
df_compas_1.loc[df_compas_1["decile_score_cat"].isna(), "decile_score_cat"] = "missing"
df_compas_1.loc[df_compas_1["vr_charge_degree"].isna(), "vr_charge_degree"] = "missing"
df_compas_1.loc[df_compas_1["c_days_from_compas_cat"].isna(), "c_days_from_compas_cat"] = "missing"

## Remove both brackets () in r_charge_degree
df_compas_1['r_charge_degree'] = df_compas_1['r_charge_degree'].str.replace(r'\(|\)', '', regex=True)
df_compas_1['vr_charge_degree'] = df_compas_1['vr_charge_degree'].str.replace(r'\(|\)', '', regex=True)


## keep c_charge_degree, r_charge_degree, is_violent_recid, vr_charge_degree, decile_score_1, v_decile_score

cols_to_drop = [ 'race', 'juv_fel_count', 'decile_score','juv_misd_count', 'juv_other_count', 'priors_count'
                ,'days_b_screening_arrest', 'c_days_from_compas', 'r_days_from_arrest',  'start', 'end']

df_compas_1 = df_compas_1.drop(columns=cols_to_drop)


## Drop some variables that are based on or correlated to the outcome variable
cols_to_drop = ['is_recid', 'is_violent_recid', 'decile_score_1', 'v_decile_score', 'decile_score_cat', 'priors_count_1', 'event', 'start_cat', 'end_cat'
                , 'r_charge_degree', 'vr_charge_degree', 'r_days_from_arrest_cat']

df_compas_1 = df_compas_1.drop(columns=cols_to_drop)

print(df_compas_1.shape)
# (6172, 11)

## Also export data for future usage
df_compas_1.to_csv('data/data_compas/compas_final.csv', index=False)