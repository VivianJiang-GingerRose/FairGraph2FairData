import numpy as np
import pandas as pd
import re
import glob

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# Set seeds
seed_bn = 123
seed_ml = 123


# *******************
# Test Compas data
# *******************
df_law = pd.read_csv(f'data_law_school/law_school_clean_preprocessed.csv')

print(df_law.shape)
# (20798, 6)

# Drop rows with missing values
df_law = df_law.dropna()
# Reset the index
df_law = df_law.reset_index(drop=True)

print(df_law.shape)
# (20527, 6)

# Check for missing values
df_law.isnull().sum()

df_law.dtypes

# Make sure all the columns are in the correct data type
df_law['fam_inc'] = df_law['fam_inc'].astype('object')
df_law['male'] = df_law['male'].astype('object')


df_law.race.value_counts()
# White        17237
# Non-White     3290

df_law.admit.value_counts()
# 0    18766
# 1     1761

node_list = df_law.columns.tolist()

""" **********************************************
4. Parameter Learning
**********************************************"""
from FairnessAwareHC.models import BayesianNetwork
from FairnessAwareHC.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from FairnessAwareHC.factors.discrete import State
from FairnessAwareHC.sampling import BayesianModelSampling


# Set parameters
df_bn = df_law.copy()

folder_name = 'experiments/law_school'
dt_name = "law"
outcome_name = 'admit'
outcome_name_val_1 = 0
outcome_name_val_2 = 1


protected_attr = "race"
protected_attr_val_1 = "White"
protected_attr_val_2 = "Non-White"
protected_attr_proxy = ['fam_inc']

cols_ohe =  ['fam_inc', 'male', 'race', 'ugpa_bin', 'lsat_bin']

original_data_size = 20527
large_data_size = original_data_size * 2
half_data_size = original_data_size // 2


# Specify the folder name where the logs are saved.
log_folder_name = 'outputs'

# Import all the logs in the folder, including the file name, where the last component indicates the lambda value.
# Get all the log files
log_files = glob.glob(f'{folder_name}/{log_folder_name}/*.txt')
# Remove the txt file from without fairness run from log_files
log_files = [file for file in log_files if 'withoutFairness' not in file]

# Get the lambda values
lambda_values = [re.search(r'lambda_fairness_(\d+\.\d+)', file).group(1) if re.search(r'lambda_fairness_(\d+\.\d+)', file) else None for file in log_files]
# Remove None from lambda_values
lambda_values = [l for l in lambda_values if l]


# Read the log files one by one, and only keep lines start with "d.add_edge(". Also keep the lambda value.
# Store the results in a dictionary.
log_data = {}
for i, file in enumerate(log_files):
    with open(file, 'r') as f:
        lines = f.readlines()
        # Remove leading and trailing whitespaces
        lines = [line.strip() for line in lines]
        # print(lines)
        log_data[lambda_values[i]] = [line for line in lines if line.startswith("d.add_edge")]




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sample 0: Initial graph
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dag0 = BayesianNetwork()
dag0.add_nodes_from(['fam_inc', 'male', 'race', 'admit', 'ugpa_bin', 'lsat_bin'])

# Add edges
dag0.add_edge('race', 'lsat_bin')
dag0.add_edge('race', 'fam_inc')
dag0.add_edge('lsat_bin', 'ugpa_bin')
dag0.add_edge('lsat_bin', 'admit')
dag0.add_edge('ugpa_bin', 'male')


law_bn_0 = BayesianNetwork(ebunch=dag0.edges())
law_bn_0.fit(
    data=df_law,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

# forward sample
inference = BayesianModelSampling(law_bn_0)
law_bn_1_sample_0 = inference.forward_sample(size=original_data_size, include_latents=False, seed=seed_bn)

# Export the data
law_bn_1_sample_0.to_csv(f'{folder_name}/{log_folder_name}/law_initial_graph.csv', index=False)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sample 1: No fairness constraints
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dag1 = BayesianNetwork()
dag1.add_nodes_from(['fam_inc', 'male', 'race', 'admit', 'ugpa_bin', 'lsat_bin'])

# Add edges
dag1.add_edge('race', 'lsat_bin')
dag1.add_edge('race', 'fam_inc')
dag1.add_edge('race', 'male')
dag1.add_edge('lsat_bin', 'ugpa_bin')
dag1.add_edge('lsat_bin', 'admit')
dag1.add_edge('ugpa_bin', 'male')


law_bn_1 = BayesianNetwork(ebunch=dag1.edges())
law_bn_1.fit(
    data=df_law,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=1000,
)

# forward sample
inference = BayesianModelSampling(law_bn_1)
law_bn_1_sample_1 = inference.forward_sample(size=original_data_size, include_latents=False, seed=seed_bn)

# Export the data
law_bn_1_sample_1.to_csv(f'{folder_name}/{log_folder_name}/law_without_fairness.csv', index=False)


""" **********************************************
1. Generate data based on the DAGs created
**********************************************"""
for lambda_val in lambda_values:

    # lambda_val = '0.1'

    # Create samples for each lambda value and dag
    d = BayesianNetwork()
    d.add_nodes_from(node_list)

    # Add edges using the log data
    for edge in log_data[lambda_val]:
        # Extract the edge
        edge = edge.split("(")[1].split(")")[0].split(", ")
        # Remove the ' from the edge
        edge = [re.sub(r"'", "", e) for e in edge]
        d.add_edge(edge[0], edge[1])

    # Fit the model
    model_bn = BayesianNetwork(ebunch=d.edges())
    model_bn.fit(
        data=df_bn,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=1000,
    )

    #------------------------
    # 1. forward sample and equal size to original data
    #------------------------
    inference = BayesianModelSampling(model_bn)
    model_bn_sample_1 = inference.forward_sample(size=original_data_size, include_latents=False, seed=seed_bn)

    # Export the data
    model_bn_sample_1.to_csv(f'{folder_name}/{log_folder_name}/{dt_name}_with_fairness_lambda_{lambda_val}.csv', index=False)
        

    #------------------------
    # 2. Generate data with larger volume
    #------------------------
    model_bn_sample_2 = inference.forward_sample(size=large_data_size, include_latents=False, seed=seed_bn)

    # Export the data
    model_bn_sample_2.to_csv(f'{folder_name}/{log_folder_name}/{dt_name}_with_fairness_double_size_lambda_{lambda_val}.csv', index=False)

    #------------------------
    # 3. Generate data with equal outcome class size
    #------------------------
    # Use reject sampling to sample equal number of class 1 and class 0
    sample_31 = inference.rejection_sample(evidence=[State(var=outcome_name, state=outcome_name_val_1)], size=half_data_size, include_latents=False, seed=seed_bn)
    sample_30 = inference.rejection_sample(evidence=[State(var=outcome_name, state=outcome_name_val_2)], size=half_data_size, include_latents=False, seed=seed_bn)
    model_bn_sample_3 = pd.concat([sample_31, sample_30])

    model_bn_sample_3.to_csv(f'{folder_name}/{log_folder_name}/{dt_name}_with_fairness_equal_outcome_lambda_{lambda_val}.csv', index=False)

    #------------------------
    # 4. Generate equal number of female and male and the total number of samples is 30000
    #------------------------
    sample_41 = inference.rejection_sample(evidence=[State(var=protected_attr, state=protected_attr_val_1)], size=half_data_size, include_latents=False, seed=seed_bn)
    sample_40 = inference.rejection_sample(evidence=[State(var=protected_attr, state=protected_attr_val_2)], size=half_data_size, include_latents=False, seed=seed_bn)
    model_bn_sample_4 = pd.concat([sample_41, sample_40])
                                    
    model_bn_sample_4.to_csv(f'{folder_name}/{log_folder_name}/{dt_name}_with_fairness_equal_protected_attr_lambda_{lambda_val}.csv', index=False)



#---------------------------
# Pass this througn ML model
#---------------------------

# Import the custom built ml model functions and evaluation metrics
from run_ml_models import *


# Make sure all columns except the outcome column are set as object type, not floats, even when it contains numerical values.
def reset_cols_to_object(df_in, oc=None):
    cols = df_in.columns

    # Remove outcome_name from cols
    if oc:
        cols = cols.drop(oc)
    # Set all cols to object type
    for col in cols:
        df_in[col] = df_in[col].astype('object')

    return df_in

#----------------------------
# Import data that has been preprocessed, all numerical features are converted to categorical
#----------------------------
# Create an empty dataframe to store all generated datasets
df_master = pd.DataFrame()
X_master = pd.DataFrame()
y_master = pd.DataFrame()

def add_df_to_master(df_list, model_name, X_train_name, y_train_name, lambda_val, sample_name):
    global df_master, X_master, y_master

    for df in df_list:
        df['lambda'] = lambda_val
        df['model_name'] = model_name
        df['sample_name'] = sample_name
        df['X_train_name'] = X_train_name
        df['y_train_name'] = y_train_name

    df_master = pd.concat([df_master, df_list[0]])
    X_master = pd.concat([X_master, df_list[1]])
    y_master = pd.concat([y_master, df_list[2]])
    return df_master, X_master, y_master



#---------------- 
# 0.0 Import original compas data
#---------------- 
# # First swap the label, as class 0 no recid is the disarable outcome
df_law_1 = df_law.copy()
df00_1, X00, y00 = preprocess_data(df_law_1, cols_ohe, outcome_name)

y00 = y00.to_frame()
# Add values to indicate which dataset it is to df00_1, X00, y00 all at once
df_list = [df00_1, X00, y00]
df_master, X_master, y_master = add_df_to_master(df_list, 'df00_1','X00','y00', np.NaN,'original dataset')

#---------------- 
# 99.1 Create a new dataset with race column dropped
#---------------- 
df01 = df_law_1.drop(columns='race')    
cols_ohe_1 =  ['fam_inc', 'male', 'admit', 'ugpa_bin', 'lsat_bin']
df01_1, X01, y01 = preprocess_data(df01, cols_ohe_1, outcome_name)

# Add values to indicate which dataset it is to df00_1, X00, y00 all at once
y01 = y01.to_frame()
df_list = [df01_1, X01, y01]
df_master, X_master, y_master = add_df_to_master(df_list, 'df01_1','X01','y01', np.NaN, 'original dataset with protected attribute removed (FTU)')

#---------------- 
# 99.2 Create a new dataset with race and priors_count_cat columns dropped
#---------------- 
df02 = df_law_1.drop(columns=['race', 'fam_inc']) 
cols_ohe_2 =  ['male', 'admit', 'ugpa_bin', 'lsat_bin']
df02_1, X02, y02 = preprocess_data(df02, cols_ohe_2, outcome_name)

# Add values to indicate which dataset it is to df00_1, X00, y00 all at once
y02 = y02.to_frame()
df_list = [df02_1, X02, y02]
df_master, X_master, y_master = add_df_to_master(df_list, 'df02_1','X02','y02',np.NaN, 'original dataset with protected attribute and proxies removed (FTU_2)')


#---------------- 
# 1.0 Sampled from initial DAG
#---------------- 
df0 = pd.read_csv(f'{folder_name}/{log_folder_name}/law_initial_graph.csv')
df0 = reset_cols_to_object(df0, outcome_name)
df0_1, X0, y0 = preprocess_data(df0, cols_ohe, outcome_name)

# Add values to indicate which dataset it is to df00_1, X00, y00 all at once
y0 = y0.to_frame()
df_list = [df0_1, X0, y0]
df_master, X_master, y_master = add_df_to_master(df_list, 'df0_1','X0','y0',np.NaN, 'Initial DAG')

#---------------- 
# 2. Sampled from HC without fairness constraints
#---------------- 
df1 = pd.read_csv(f'{folder_name}/{log_folder_name}/law_without_fairness.csv')
df1 = reset_cols_to_object(df1, outcome_name)
df1_1, X1, y1 = preprocess_data(df1, cols_ohe, outcome_name)

# Add values to indicate which dataset it is to df00_1, X00, y00 all at once
y1 = y1.to_frame()
df_list = [df1_1, X1, y1]
df_master, X_master, y_master = add_df_to_master(df_list, 'df1_1','X1','y1',np.NaN, 'HC without fairness constraints')

#---------------- 
# 3. Iterate through different lambda values and append the sampled datasets to the master datasets.
#----------------
def generate_missing_columns(df_original, df_new):
    # If column is missing in the generated data, then generate it at random to the proportion of the original dataset
    missing_cols = set(df_original.columns) - set(df_new.columns)
    for col in missing_cols:
        value_counts = df_original[col].value_counts(normalize=True)
        values = value_counts.index.values
        probabilities = value_counts.values
        df_new[col] = np.random.choice(values, size=len(df_new), p=probabilities)
    
    return df_new

for i, lambda_val in enumerate(lambda_values):

    # lambda_val = '1.5'
    # Find the position number of lambda_val in the list of lambda values
    lambda_pos = lambda_values.index(lambda_val)

    #****************
    # 3.1 Sampled from graph with fairness constraints
    #****************
    df31 = pd.read_csv(f'{folder_name}/{log_folder_name}/{dt_name}_with_fairness_lambda_{lambda_val}.csv')
    df31 = reset_cols_to_object(df31, outcome_name)

    df31 = generate_missing_columns(df_original=df_law, df_new=df31)
    df31_1, X31, y31 = preprocess_data(df31, cols_ohe, outcome_name)

    # Add values to indicate which dataset it is to df00_1, X00, y00 all at once
    y31 = y31.to_frame()
    df_list = [df31_1, X31, y31]
    model_name = 'df31_1'+'_'+str(i)
    X_train_name = 'X31'+'_'+str(i)
    y_train_name = 'y31'+'_'+str(i)
    df_master, X_master, y_master = add_df_to_master(df_list, model_name,X_train_name,y_train_name,lambda_val, f'HC with fairness constraints and same size as original dataset lambda = {lambda_val}')

    #****************
    # 3.2 Sampeled from graph with fairness constraints and double the size of original dataset
    #****************
    df32 = pd.read_csv(f'{folder_name}/{log_folder_name}/{dt_name}_with_fairness_double_size_lambda_{lambda_val}.csv')
    df32 = reset_cols_to_object(df32, outcome_name)

    df32 = generate_missing_columns(df_original=df_law, df_new=df32)
    df32_1, X32, y32 = preprocess_data(df32, cols_ohe, outcome_name)

    # Add values to indicate which dataset it is to df00_1, X00, y00 all at once
    y32 = y32.to_frame()
    df_list = [df32_1, X32, y32]
    model_name = 'df32_1'+'_'+str(i)
    X_train_name = 'X32'+'_'+str(i)
    y_train_name = 'y32'+'_'+str(i)
    df_master, X_master, y_master = add_df_to_master(df_list, model_name,X_train_name,y_train_name,lambda_val, f'HC with fairness constraints and double the size of original dataset lambda = {lambda_val}')

    #****************
    # 3.3 Sampeled from graph with fairness constraints and equal number of class 1 and class 0
    #****************
    df33 = pd.read_csv(f'{folder_name}/{log_folder_name}/{dt_name}_with_fairness_equal_outcome_lambda_{lambda_val}.csv')
    df33 = reset_cols_to_object(df33, outcome_name)
    df33 = generate_missing_columns(df_original=df_law, df_new=df33)
    df33_1, X33, y33 = preprocess_data(df33, cols_ohe, outcome_name)

    # Add values to indicate which dataset it is to df00_1, X00, y00 all at once
    y33 = y33.to_frame()
    df_list = [df33_1, X33, y33]
    model_name = 'df33_1'+'_'+str(i)
    X_train_name = 'X33'+'_'+str(i)
    y_train_name = 'y33'+'_'+str(i)
    df_master, X_master, y_master = add_df_to_master(df_list, model_name,X_train_name,y_train_name, lambda_val, f'HC with fairness constraints and equal number of class 1 and class 0 lambda = {lambda_val}')

    #****************
    # 3.4 Sampeled from graph with fairness constraints and equal number of male and female
    #****************
    df34 = pd.read_csv(f'{folder_name}/{log_folder_name}/{dt_name}_with_fairness_equal_protected_attr_lambda_{lambda_val}.csv')
    df34 = reset_cols_to_object(df33, outcome_name)
    df34_1, X34, y34 = preprocess_data(df34, cols_ohe, outcome_name)
    
    # Add values to indicate which dataset it is to df00_1, X00, y00 all at once
    y34 = y34.to_frame()
    df_list = [df34_1, X34, y34]
    model_name = 'df34_1'+'_'+str(i)
    X_train_name = 'X34'+'_'+str(i)
    y_train_name = 'y34'+'_'+str(i)
    df_master, X_master, y_master = add_df_to_master(df_list, model_name,X_train_name,y_train_name,lambda_val,  f'HC with fairness constraints and equal number of protected attribute lambda = {lambda_val}')

df_master.model_name.unique()
df_master.sample_name.unique()


#----------------------------
# Build models and evaluate fairness
#----------------------------

#-----------------------------------
# Run~
#-----------------------------------
df_model_names = df_master['model_name'].unique().tolist()
sample_names = df_master['sample_name'].unique().tolist()
  
X_train_names = df_master['X_train_name'].unique().tolist()
y_train_names = df_master['y_train_name'].unique().tolist()

# Define the datasets used for model testing. If testing data is the same as training data, then leave it as None.
X_test_names = df_master['X_train_name'].unique().tolist()
y_test_names = df_master['y_train_name'].unique().tolist()

# Test on the original data for original and FTU datasets.
# Test on the sythentic data for all generated datasets.
X_test_names = [x if x in ['X00','X01', 'X02', 'X0', 'X1'] else 'X00' for x in X_test_names]
y_test_names = [x if x in ['y00','y01','y02', 'y0', 'y1']  else 'y00' for x in y_test_names]


# sf_indata: first constract a list of 'Y's the same size as df_model_names, then the second and third elements are 'N's
# This is to indicate if the dataset has the sensitive attribute in the input data. If not, the code has to grab the sensitive attribute from the original dataset.
sf_indata = ['Y'] * len(df_model_names)
sf_indata[1] = 'N'
sf_indata[2] = 'N'


# Create an empty dataframe to store the results
m_results_f = pd.DataFrame()

for i, df_name in enumerate(df_model_names):

    df_model = df_master[df_master['model_name'] == df_name]
    X_tr = X_master[X_master['X_train_name'] == X_train_names[i]]
    y_tr = y_master[y_master['y_train_name'] == y_train_names[i]]
    X_te = X_master[X_master['X_train_name'] == X_test_names[i]]
    y_te = y_master[y_master['y_train_name'] == y_test_names[i]]

    # Drop the last five columns
    cols_to_drop = ['lambda', 'model_name', 'sample_name', 'X_train_name', 'y_train_name']
    df_model = df_model.drop(columns=cols_to_drop)
    X_tr = X_tr.drop(columns=cols_to_drop)
    X_te = X_te.drop(columns=cols_to_drop)

    # Convert the y_tr and y_te to series
    y_tr = y_tr.iloc[:,0]
    y_te = y_te.iloc[:,0]

    # Remove columns in the training and test data that only contain NaN values
    X_tr = X_tr.dropna(axis=1, how='all')
    X_te = X_te.dropna(axis=1, how='all')

    # Drop columns that have NaN values. This should only impact FTU datasets where the protected attribute has been removed.
    # This is to avoid error when building the model.
    df_model = df_model.dropna(axis=1, how='any')
    nan_rows = X_tr.isnull().any(axis=1)
    X_tr = X_tr[~nan_rows]
    y_tr = y_tr[~nan_rows]
    nan_rows = X_te.isnull().any(axis=1)
    X_te = X_te[~nan_rows]
    y_te = y_te[~nan_rows]

    # Drop the columns that had been one-hot-encoded if it is in the dataset
    df_model = df_model.drop(columns=[col for col in cols_ohe if col in df_model.columns])
    X_tr = X_tr.drop(columns=[col for col in cols_ohe if col in X_tr.columns])
    X_te = X_te.drop(columns=[col for col in cols_ohe if col in X_te.columns])

    # If the test data contains different columns than the training data, then drop the columns and the rows that are not in the test data.
    extra_columns = set(X_te.columns) - set(X_tr.columns)
    rows_to_remove = X_te[list(extra_columns)].any(axis=1)
    X_te = X_te.drop(columns=extra_columns)
    X_te = X_te[~rows_to_remove]
    y_te = y_te[~rows_to_remove]


    sample_name = sample_names[i]
    sf_indata_f = sf_indata[i]
    random_state=seed_ml

    print(f"Processing {sample_name}")

    sf_col_name = 'race'
    benchmark_col = "('race_White',)"
    protected_col = "('race_Non-White',)"


    # Call your custom function here and pass the dataframe as an argument
    result = run_samples(sample_name, df_model, outcome_name, X_tr, y_tr, X_te, y_te, benchmark_col, protected_col, sf_col_name, df_original=df00_1, random_state=random_state, sf_indata=sf_indata_f)

    m_results_f = pd.concat([m_results_f, result])

m_results_f.to_csv(f'{folder_name}/{log_folder_name}/{dt_name}_model_fairness_results.csv', index=False)

