import pandas as pd


###--- Step 1: Load the data ---###
df_law = pd.read_csv('data/data_law_school/law_school_clean.csv')

df_law.shape
# (20798, 12)

## Construct the target variable using the method of paper 
## "Counterfactual Situation Testing: Uncovering Discrimination under Fairness given the Difference"
## Y is the weighted sum of the variables 'ugpa' (0.6) and 'lsat' (0.4). 
## If the weighted sum is greater than 14.8, the target variable is 1, otherwise 0.
## The 14.8 threshold is calculated by:
#  1. average LSAT score of top 20 US law schools is 172 on a 120-180 scale, and after converting to the 48-point scale, it is 42.6. (https://www.bestcolleges.com/research/average-lsat-score-accepted-law-students/#average-lsat-score)
#  2. average median UGPA score of top 20 US law schools is 3.86. (https://www.toplawschoolconsulting.com/blog/law-school-admissions-statistics/#:~:text=Across%20the%20country%2C%20the%20median,more%20about%202023%20acceptance%20rates.)
#  3. 0.6 * 3.86 + 0.4 * 42.6 = 19.5
#  The resulting overall admission rate is around 10%, which is consistant with the published statistics.
df_law['admit'] = 0
df_law.loc[(0.6 * df_law['ugpa'] + 0.4 * df_law['lsat']) > 19.5, 'admit'] = 1

## Overall addmission rate
df_law.admit.value_counts(dropna=False)
# 0    18768
# 1     2030

df_law.admit.value_counts(normalize=True, dropna=False)
# 0    0.902394
# 1    0.097606

## Check by gender
df_law.groupby('male').admit.value_counts(normalize=True)
# male  admit
# 0.0   0        0.914831
#       1        0.085169
# 1.0   0        0.892677
#       1        0.107323

## Check by race
df_law.groupby('race').admit.value_counts(normalize=True)
# race       admit
# Non-White  0        0.957061
#            1        0.042939
# White      0        0.892059
#            1        0.107941

## Check by family income
df_law.groupby('fam_inc').admit.value_counts(normalize=True)
# fam_inc  admit
# 1.0      0        0.952494
#          1        0.047506
# 2.0      0        0.929308
#          1        0.070692
# 3.0      0        0.920070
#          1        0.079930
# 4.0      0        0.888647
#          1        0.111353
# 5.0      0        0.854105
#          1        0.145895

##------------------
## Bin ugpa and lsat
##------------------
## 1. ugpa
df_law['ugpa'].describe()

## Construct a list from 1.5 to 4.0 with 0.25 increments
bins = [1.5 + 0.25*i for i in range(0, 12)]
## Construct a list labels to be a concatenation of 'ugpa_' and the lower and upper bounds of the bins
labels = ['ugpa_' + str(bins[i]) + '_' + str(bins[i+1]) for i in range(0, 11)]
df_law['ugpa_bin'] = pd.cut(df_law['ugpa'], bins=bins, labels=labels)

## 2. lsat
df_law['lsat'].describe()

## Construct a list from 11 to 48 with 4 units increments
bins = [11 + 4*i for i in range(0, 10)]
## Construct a list labels to be a concatenation of 'ugpa_' and the lower and upper bounds of the bins
labels = ['lsat_' + str(bins[i]) + '_' + str(bins[i+1]) for i in range(0, 9)]
df_law['lsat_bin'] = pd.cut(df_law['lsat'], bins=bins, labels=labels)


## Drop some columns
df_law.drop(columns=['decile1b', 'decile3', 'zfygpa', 'zgpa', 'fulltime', 'pass_bar', 'tier', 'ugpa', 'lsat'], inplace=True)

## Save the data
df_law.to_csv('data_law_school/law_school_clean_for_vj_testing.csv', index=False)


###--- Step 2: Make data binary ---###
df_law_new = df_law.copy()

# df_law_new.head()

## One-Hot Encoding all columns except 'admit'
df_law_new = pd.get_dummies(df_law_new, columns=['fam_inc', 'male', 'race', 'ugpa_bin', 'lsat_bin'], drop_first=True)

## Convert the true/false values to 1/0 for all columns except 'admit'
df_law_new = df_law_new.astype(int)

## Construct another variable for female, which is 1 if 'male_1.0' is 0, otherwise 0
df_law_new['female'] = 1 - df_law_new['male_1.0']

df_law_new['female'].value_counts()
# female
# 0    11675
# 1     9123

## Rename column male_1.0 to male
df_law_new = df_law_new.rename(columns={'male_1.0':'male'})

## Do the same thing for race
df_law_new['race_Non_White'] = 1 - df_law_new['race_White']
df_law_new['race_Non_White'].value_counts()

## Save the data
df_law_new.to_csv('data/data_law_school/law_school_clean_for_vj_testing_binary.csv', index=False)
