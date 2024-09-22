import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)

df_adult = pd.read_csv("data/data_adult/adult.csv")

## Overwrite missing working class and occupation with "Unknown"
df_adult.loc[df_adult["workclass"] == "?", "workclass"] = "Unknown"
df_adult.loc[df_adult["occupation"] == "?", "occupation"] = "Unknown"
df_adult.loc[df_adult["native.country"] == "?", "native.country"] = "Unknown"

## Categorise numerical data
df_adult_1 = df_adult.copy()
df_adult_1['age_cat'] = pd.cut(df_adult['age'], bins=[0, 20, 25, 30, 35, 40, 50, 60, 100], labels=['age_0-20', 'age_20-25', 'age_25-30', 'age_30-35', 'age_35-40', 'age_40-50', 'age_50-60', 'age_60+'])
df_adult_1['fnlwgt_cat'] = pd.cut(df_adult['fnlwgt'], bins=[0, 100000, 150000, 200000, 300000, 100000000], labels=['0-100k', '100k-150k', '150k-200k', '200k-300k', '300k+'])
df_adult_1['capital_gain_cat'] = pd.cut(df_adult_1['capital.gain'], bins=[0, 5000, 20000], labels=['capital_gain_0-5000', 'capital_gain_5000+'])
df_adult_1['capital_gain_cat'] = np.where(df_adult_1['capital_gain_cat'].isnull(), 'capital_gain_0', df_adult_1['capital_gain_cat'])
df_adult_1['capital_loss_cat'] = pd.cut(df_adult_1['capital.loss'], bins=[0, 40, 10000], labels=['capital_loss_0-40','capital_loss_40+'])
df_adult_1['capital_loss_cat'] = np.where(df_adult_1['capital_loss_cat'].isnull(), 'capital_loss_0', df_adult_1['capital_loss_cat'])
df_adult_1['hours_per_week_cat'] = pd.cut(df_adult_1['hours.per.week'], bins=[0, 20, 39, 60, 1000], labels=['hours_per_week_0-20', 'hours_per_week_21-39', 'hours_per_week_40-60', 'hours_per_week_60+'])


## Reduce the levels of education
df_adult_1['education_cat'] = np.where(df_adult_1['education'].isin(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th']), 'education_Some-HS', df_adult_1['education'])

## Reduce the levels of native country
df_adult_1['native_country_cat'] = np.where(~df_adult_1['native.country'].isin(['United-States']), 'native_country_Non-US', df_adult_1['native.country'])

## Reduce the levels of race
df_adult_1['race_cat'] = np.where(~df_adult_1['race'].isin(['White', 'Black']), 'Other-race', df_adult_1['race'])

## Reduce the levels of occupation
df_adult_1['occupation_cat'] = np.where(df_adult_1['occupation'].isin(['Prof-specialty', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Tech-support']), 'White-collar', df_adult_1['occupation'])
df_adult_1['occupation_cat'] = np.where(df_adult_1['occupation'].isin(['Craft-repair', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners','Farming-fishing', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']), 'Blue-collar', df_adult_1['occupation_cat'])
df_adult_1['occupation_cat'] = np.where(~df_adult_1['occupation_cat'].isin(['White-collar', 'Blue-collar']), 'Other-occupation', df_adult_1['occupation_cat'])

## Reduce the levels of workclass
df_adult_1['workclass_cat'] = np.where(df_adult_1['workclass'].isin(['Self-emp-not-inc', 'Self-emp-inc']), 'Self-emp', df_adult_1['workclass'])
df_adult_1['workclass_cat'] = np.where(df_adult_1['workclass'].isin(['Local-gov', 'State-gov', 'Federal-gov']), 'Gov', df_adult_1['workclass_cat'])
df_adult_1['workclass_cat'] = np.where(df_adult_1['workclass'].isin(['Without-pay', 'Never-worked', 'Unknown']), 'Other-workclass', df_adult_1['workclass_cat'])

## Reduce the levels of marital.status
df_adult_1['marital_status_cat'] = np.where(df_adult_1['marital.status'].isin(['Divorced', 'Separated', 'Widowed']), 'Divorced-Separated-Widowed', df_adult_1['marital.status'])
df_adult_1['marital_status_cat'] = np.where(df_adult_1['marital.status'].isin(['Married-spouse-absent', 'Married-AF-spouse']), 'Married-other', df_adult_1['marital_status_cat'] )

## Combine husband and wife into one category, as they are highly correlated to gender
df_adult_1['relationship_cat'] = df_adult_1['relationship'].apply(lambda x: 'Married' if x in ['Husband', 'Wife'] else x)

## Edit the target
df_adult_1['income_f'] = np.where(df_adult_1['income'] == "<=50K", 0, 1)

df_adult_2 = df_adult_1.drop(columns=['fnlwgt', 'education', 'education.num', 'workclass', 'occupation', 'native.country', 'marital.status', 'income', 'race', 'age',  'capital.gain', 'capital.loss', 'hours.per.week','fnlwgt_cat', 'relationship'])

## Also export data for future usage
df_adult_2.to_csv('data/data_adult/adult_final.csv', index=False)