import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


## Set seeds
seed_bn = 123
seed_ml = 123

df_twc = pd.read_csv('data/data_taiwan_credit/twc_raw.csv')

print(df_twc.shape)
# (30000, 25)

print(df_twc.dtypes)

df_twc.default.value_counts(normalize=True)
# 0    0.7788
# 1    0.2212


##------------------------------------
## Add an synthetic noise data - interest rate column. This interest rate is higher for female and lower for male.
##------------------------------------
Y, A = df_twc.loc[:, "default payment next month"], df_twc.loc[:, "SEX"]
X = df_twc.copy()
X.loc[:, "Interest"] = np.random.normal(loc=2 * Y, scale=A)

## Append back to df_twc
df_twc = pd.concat([df_twc, X[['Interest']]], axis=1)


##--------------------------------------------  
## Categorise numerical data
##--------------------------------------------  
df_twc_1 = df_twc.copy()

## Bin the synthetic interest rate variable
df_twc_1['interest_cat'] = pd.cut(df_twc_1['Interest'], bins=[-1000000, 0, 0.5, 1, 2, 3, 1000000], labels=['interest_lt0', 'interest_0-0.5', 'interest_0.5-1', 'interest_1-2', 'interest_2-3', 'interest_3'])


df_twc_1['age_cat'] = pd.cut(df_twc_1['AGE'], bins=[0, 20, 25, 30, 35, 40, 50, 60, 100], labels=['age_0-20', 'age_20-25', 'age_25-30', 'age_30-35', 'age_35-40', 'age_40-50', 'age_50-60', 'age_60+'])
df_twc_1['limit_bal_cat'] = pd.cut(df_twc_1['LIMIT_BAL'], bins=[0, 50000, 100000, 200000, 300000, 1000000], labels=['limit_bal_0-50k', 'limit_bal_50k-100k', 'limit_bal_100k-200k', 'limit_bal_200k-300k', 'limit_bal_300k+'])


## Bin the bill amount variables
df_twc_1['bill_amt1_cat'] = pd.cut(df_twc_1['BILL_AMT1'], bins=[-1000000, 0, 5000, 20000, 50000, 100000, 10000000], labels=['bill_amt1_lt0', 'bill_amt1_0-5k', 'bill_amt1_5k-20k', 'bill_amt1_20k-50k', 'bill_amt1_50k-100k', 'bill_amt1_100k+'])
df_twc_1['bill_amt2_cat'] = pd.cut(df_twc_1['BILL_AMT2'], bins=[-1000000, 0, 5000, 20000, 50000, 100000, 10000000], labels=['bill_amt2_lt0', 'bill_amt2_0-5k', 'bill_amt2_5k-20k', 'bill_amt2_20k-50k', 'bill_amt2_50k-100k', 'bill_amt2_100k+'])
df_twc_1['bill_amt3_cat'] = pd.cut(df_twc_1['BILL_AMT3'], bins=[-1000000, 0, 5000, 20000, 50000, 100000, 10000000], labels=['bill_amt3_lt0', 'bill_amt3_0-5k', 'bill_amt3_5k-20k', 'bill_amt3_20k-50k', 'bill_amt3_50k-100k', 'bill_amt3_100k+'])
df_twc_1['bill_amt4_cat'] = pd.cut(df_twc_1['BILL_AMT4'], bins=[-1000000, 0, 5000, 20000, 50000, 100000, 10000000], labels=['bill_amt4_lt0', 'bill_amt4_0-5k', 'bill_amt4_5k-20k', 'bill_amt4_20k-50k', 'bill_amt4_50k-100k', 'bill_amt4_100k+'])
df_twc_1['bill_amt5_cat'] = pd.cut(df_twc_1['BILL_AMT5'], bins=[-1000000, 0, 5000, 20000, 50000, 100000, 10000000], labels=['bill_amt5_lt0', 'bill_amt5_0-5k', 'bill_amt5_5k-20k', 'bill_amt5_20k-50k', 'bill_amt5_50k-100k', 'bill_amt5_100k+'])
df_twc_1['bill_amt6_cat'] = pd.cut(df_twc_1['BILL_AMT6'], bins=[-1000000, 0, 5000, 20000, 50000, 100000, 10000000], labels=['bill_amt6_lt0', 'bill_amt6_0-5k', 'bill_amt6_5k-20k', 'bill_amt6_20k-50k', 'bill_amt6_50k-100k', 'bill_amt6_100k+'])

## Bin th pay amount 
df_twc_1['pay_amt1_cat'] = pd.cut(df_twc_1['PAY_AMT1'], bins=[-1, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 10000000], labels=['pay_amt1_0-1k', 'pay_amt1_1-2k', 'pay_amt1_1-1.5k', 'pay_amt1_1.5k-3k', 'pay_amt1_3k-4k', 'pay_amt1_4k-5k', 'pay_amt1_5k-10k', 'pay_amt1_10k+'])
df_twc_1['pay_amt2_cat'] = pd.cut(df_twc_1['PAY_AMT2'], bins=[-1, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 10000000], labels=['pay_amt2_0-1k', 'pay_amt2_1-2k', 'pay_amt2_1-1.5k', 'pay_amt2_1.5k-3k', 'pay_amt2_3k-4k', 'pay_amt2_4k-5k', 'pay_amt2_5k-10k', 'pay_amt2_10k+'])
df_twc_1['pay_amt3_cat'] = pd.cut(df_twc_1['PAY_AMT3'], bins=[-1, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 10000000], labels=['pay_amt3_0-1k', 'pay_amt3_1-2k', 'pay_amt3_1-1.5k', 'pay_amt3_1.5k-3k', 'pay_amt3_3k-4k', 'pay_amt3_4k-5k', 'pay_amt3_5k-10k', 'pay_amt3_10k+'])
df_twc_1['pay_amt4_cat'] = pd.cut(df_twc_1['PAY_AMT4'], bins=[-1, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 10000000], labels=['pay_amt4_0-1k', 'pay_amt4_1-2k', 'pay_amt4_1-1.5k', 'pay_amt4_1.5k-3k', 'pay_amt4_3k-4k', 'pay_amt4_4k-5k', 'pay_amt4_5k-10k', 'pay_amt4_10k+'])
df_twc_1['pay_amt5_cat'] = pd.cut(df_twc_1['PAY_AMT5'], bins=[-1, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 10000000], labels=['pay_amt5_0-1k', 'pay_amt5_1-2k', 'pay_amt5_1-1.5k', 'pay_amt5_1.5k-3k', 'pay_amt5_3k-4k', 'pay_amt5_4k-5k', 'pay_amt5_5k-10k', 'pay_amt5_10k+'])
df_twc_1['pay_amt6_cat'] = pd.cut(df_twc_1['PAY_AMT6'], bins=[-1, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 10000000], labels=['pay_amt6_0-1k', 'pay_amt6_1-2k', 'pay_amt6_1-1.5k', 'pay_amt6_1.5k-3k', 'pay_amt6_3k-4k', 'pay_amt6_4k-5k', 'pay_amt6_5k-10k', 'pay_amt6_10k+'])

## Encode categorical data
dic_sex = {1:"male",2:"female"}
dic_education = {1:"graduate_school", 2:"university", 3:"high_school", 4:"education-others", 5:"education-others", 6:"education-others", 0:"education-others"}
dic_marriage = {1:"married", 2:"single", 3:"marriage-others", 0:"marriage-others"}
dic_pay = {-1:"pay_duly", 1:"payment_delay_1_mth", 2:"payment_delay_2_mths", 3:"payment_delay_3_mths", 4:"payment_delay_4_mths", 5:"payment_delay_5_mths", 6:"payment_delay_6_mths"
           , 7:"payment_delay_7_mths"  , 8:"payment_delay_8_mths", 9:"payment_delay_9+_mths", -2:"payment-other", 0:"payment-other"}

df_twc_1['gender_cat'] = df_twc['SEX'].map(dic_sex)
df_twc_1['education_cat'] = df_twc['EDUCATION'].map(dic_education)
df_twc_1['marriage_cat'] = df_twc['MARRIAGE'].map(dic_marriage)
df_twc_1['pay_1_cat'] = df_twc['PAY_1'].map(dic_pay)
df_twc_1['pay_2_cat'] = df_twc['PAY_2'].map(dic_pay)
df_twc_1['pay_3_cat'] = df_twc['PAY_3'].map(dic_pay)
df_twc_1['pay_4_cat'] = df_twc['PAY_4'].map(dic_pay)
df_twc_1['pay_5_cat'] = df_twc['PAY_5'].map(dic_pay)
df_twc_1['pay_6_cat'] = df_twc['PAY_6'].map(dic_pay)

df_twc_1['pay_1_cat'] = 'pay1_' + df_twc_1['pay_1_cat'].astype(str)
df_twc_1['pay_2_cat'] = 'pay2_' + df_twc_1['pay_2_cat'].astype(str)
df_twc_1['pay_3_cat'] = 'pay3_' + df_twc_1['pay_3_cat'].astype(str)
df_twc_1['pay_4_cat'] = 'pay4_' + df_twc_1['pay_4_cat'].astype(str)
df_twc_1['pay_5_cat'] = 'pay5_' + df_twc_1['pay_5_cat'].astype(str)
df_twc_1['pay_6_cat'] = 'pay6_' + df_twc_1['pay_6_cat'].astype(str)

## Check gender distribution
df_twc_1['gender_cat'].value_counts()
# female    18112
# male      11888


## Drop the columns not needed
df_twc_1 = df_twc_1.drop(columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                                  'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                                  'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'Interest'])

df_twc_1 = df_twc_1.drop(columns=['ID'])

## Check values
def print_value_counts(df):
    for column in df.columns:
        print(f"Value counts for column '{column}':")
        print(df[column].value_counts(dropna=False))
        print()
print_value_counts(df_twc_1)

## Rename the outcome variable
df_twc_1 = df_twc_1.rename(columns={'default payment next month': 'default_f'})


## Export data
df_twc_1.to_csv('data/data_taiwan_credit/twc_processed.csv', index=False)
