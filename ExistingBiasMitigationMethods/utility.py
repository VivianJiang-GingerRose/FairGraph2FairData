import numpy as np
import pandas as pd


def convert_cat_to_num(dataset_orig):
    ## Convert all the categorical columns in dataset_orig into numerical columns
    dataset_orig_num = dataset_orig.copy()
    for col in dataset_orig_num.columns:
        if dataset_orig_num[col].dtype == 'object':
            dataset_orig_num[col] = pd.Categorical(dataset_orig_num[col]).codes

    return dataset_orig_num


def get_data(dataset_used, protected):
    if dataset_used == "adult":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            # dataset_orig = load_preproc_data_adult(['sex'])
            dataset_orig = pd.read_csv('data/data_adult/adult_final.csv')
            dataset_orig = convert_cat_to_num(dataset_orig)
        else:
            privileged_groups = [{'race_cat': 1}]
            unprivileged_groups = [{'race_cat': 0}]
            # dataset_orig = load_preproc_data_adult(['race'])
            dataset_orig = pd.read_csv('data/data_adult/adult_final.csv')
            dataset_orig = convert_cat_to_num(dataset_orig)
        optim_options = None
    elif dataset_used == "dutch":
        privileged_groups = [{'sex': 1}]  
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = pd.read_csv('data/data_dutch_census/dutch.csv')
        dataset_orig = convert_cat_to_num(dataset_orig)
        optim_options = None
    elif dataset_used == "twc":
        privileged_groups = [{'gender_cat': 1}]  
        unprivileged_groups = [{'gender_cat': 0}]
        dataset_orig = pd.read_csv('data/data_taiwan_credit/twc_processed.csv')
        dataset_orig = convert_cat_to_num(dataset_orig)
        optim_options = None     
    elif dataset_used == "compas":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = pd.read_csv('data/data_compas/compas_final.csv')
            ## Reverse the model label, as lable 0 is more favourable from a fairness perspective
            dataset_orig['two_year_recid'] = np.where(dataset_orig['two_year_recid'] == 1, 0, 1)
            dataset_orig = convert_cat_to_num(dataset_orig)
        else:
            privileged_groups = [{'race_cat': 1}]
            unprivileged_groups = [{'race_cat': 0}]
            dataset_orig = pd.read_csv('data/data_compas/compas_final.csv')
            dataset_orig['two_year_recid'] = np.where(dataset_orig['two_year_recid'] == 1, 0, 1)
            dataset_orig = convert_cat_to_num(dataset_orig)
        optim_options = None   
    elif dataset_used == "law":
        if protected == "race":
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = pd.read_csv('data/data_law_school/law_school_clean_processed.csv')
            ## Remove NA
            dataset_orig = dataset_orig.dropna()
            dataset_orig = convert_cat_to_num(dataset_orig)
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = pd.read_csv('data/data_law_school/law_school_clean_processed.csv')
            ## Remove NA
            dataset_orig = dataset_orig.dropna()
            dataset_orig = convert_cat_to_num(dataset_orig)
        optim_options = None  

    return dataset_orig, privileged_groups,unprivileged_groups,optim_options
