import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import PrecisionRecallDisplay, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt



def one_hot_encoding_dataframe_columns(df):
    """Function that one-hot encodes given columns"""
    ohc = OneHotEncoder()
    for col in df.columns:
        if (df[col].dtype == 'object') or (df[col].dtype.name == 'category'):
            df_ohc = pd.DataFrame(ohc.fit_transform(df[[col]]).toarray(), columns=ohc.categories_)

            df_ohc = df_ohc.add_prefix(col+'_')

            # # One-hot encode the column
            # df_ohc = pd.DataFrame(ohc.fit_transform(df[[col]]))

            # # Flatten multi-level column index
            # df_ohc.columns = [f"{col}_{category}" for category in ohc.categories_[0]]

            # Drop the original column and concatenate the one-hot encoded data
            df = pd.concat([df.drop([col], axis=1), df_ohc], axis=1)
    return df

   
def preprocess_data(df, cols_ohe, target_name):

    """Function that pre-processes the data, ready for modelling"""
    df1 = one_hot_encoding_dataframe_columns(df[cols_ohe])
    df2 = pd.concat([df1, df.drop(cols_ohe, axis=1)], axis=1)

    df2.columns = df2.columns.map(str)

    ## Also split data for modelling
    X = df2.drop([target_name], axis=1)
    y = df2[target_name]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    return df2, X, y

##-----------------------------------
## 1. Build logistic regression model
##-----------------------------------
def build_logistic_regression_model(X_train,  X_test, y_train, y_test):

    # Build logistic regression model
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    y_pred = logreg_model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # accuracy_score = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)


    return logreg_model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced


##-----------------------------------
## 2. Build Random Forest model
##-----------------------------------
def build_random_forest_model(X_train,  X_test, y_train, y_test):

    # Create a random forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    rf_model.fit(X_train, y_train)
    
    # Predict the target variable on the testing data
    y_pred = rf_model.predict(X_test)

    # Calculate the evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)
    
    return rf_model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced


##-----------------------------------
## 3. Build XGBoost model
##-----------------------------------
def build_xgboost_model(X_train,  X_test, y_train, y_test):
    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    xgb_model.fit(X_train, y_train)
    
    # Predict the target variable on the testing data
    y_pred = xgb_model.predict(X_test)
    
    # Calculate the evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)
    
    return xgb_model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced

##-----------------------------------
## 4. Build MLP model
##-----------------------------------
def build_mlp_model(X_train,  X_test, y_train, y_test):
    # Create an MLP classifier
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)

    # Fit the model to the training data
    mlp_model.fit(X_train, y_train)

    # Predict the target variable on the testing data
    y_pred = mlp_model.predict(X_test)

    # Calculate the evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)

    return mlp_model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced




##-----------------------------------
## Test fairness of model
##-----------------------------------

def calc_fairness_metrics(y, y_pred, sf_values):



    ##-------------
    ## 1. Demographic parity (DP), (TP + FP)
    ##-------------
    from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
    ## Check if y_pred only contain one class
    if len(np.unique(y_pred)) > 1:
        DPD = demographic_parity_difference(y, y_pred, sensitive_features=sf_values)
        DPR = demographic_parity_ratio(y, y_pred, sensitive_features=sf_values)
    else:
        DPD = 0
        DPR = 0

    ##-------------
    ## 2. Equalized odds (EO) FPR, TP / (TP + FN)
    ##-------------
    from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
    if len(np.unique(y_pred)) > 1:
        EO = equalized_odds_difference(y, y_pred,sensitive_features=sf_values)
    else:
        EO = 0

    return DPD, DPR, EO


def calc_ftu(model, X, y, benchmark_col, protected_col):

    ## Over-write all the protected column to benchmark population value
    X_benchmark = X.copy()
    X_benchmark[benchmark_col] = 1
    X_benchmark[protected_col] = 0

    ## Score the dataset
    benchmark_pred = model.predict(X_benchmark)
    benchmark_pred_label = (benchmark_pred >= 0.5).astype(int)

    ## Over-write all the protected column to protected population value
    X_protected = X.copy()
    X_protected[protected_col] = 1
    X_protected[benchmark_col] = 0

    ## Score the dataset
    protected_pred = model.predict(X_protected)
    protected_pred_label = (protected_pred >= 0.5).astype(int)

    ## Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm1=confusion_matrix(y, benchmark_pred_label)
    TN1, FP1, FN1, TP1 = cm1.ravel()
    ftu1 = (TP1+FP1)/(TP1+FP1+FN1+TN1)
    # print(ftu1)

    cm2=confusion_matrix(y, protected_pred_label)
    TN2, FP2, FN2, TP2 = cm2.ravel()
    ftu2 = (TP2+FP2)/(TP2+FP2+FN2+TN2)
    # print(ftu2)
    
    ftu_ratio =  ftu1/ftu2
    ftu_diff = abs(ftu1-ftu2)
 
    ## return the scored dataset
    return ftu_diff


## Function that takes in a list and convert them into mean ± std format
def mean_plus_minus_std(values):
    # Calculating the mean and standard deviation of the values in the list
    mean_value = np.mean(values)
    std_value = np.std(values)
    
    # Formatting the mean and standard deviation to 4 decimal places
    formatted_mean = f"{mean_value:.4f}"
    formatted_std = f"{std_value:.4f}"
    
    # Returning the combined formatted string
    return f"{formatted_mean} ± {formatted_std}"


def revert_one_hot_encoding(df, benchmark_col, protected_col):

    sf_cols = [benchmark_col, protected_col] 
    original_list = []
    
    for index, row in df.iterrows():
        for column in sf_cols:
            if column in df.columns and row[column] == 1:
                # Using regex to extract the gender value from the column name
                match = re.search(r"'(.*?)'", column)
                if match:
                    retrieved_value = match.group(1)  # Extracting the matched substring
                    original_list.append(retrieved_value)
    
    return original_list

## Cross validation
def perform_cv(X_tr, y_tr, X_te, y_te, model_type, benchmark_col, protected_col, sf_col_name, df_original, random_state=None, n_splits=5,  sf_indata='Y'):

    # Stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    
    auroc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_balanced_scores = []
    dpd_scores = []
    dpr_scores = []
    eo_scores = []
    ftu_scores = []
    
    for train_idx, test_idx in cv.split(X_tr, y_tr):

        ## If there is not a seperate testing data, we perform cross validation on the training data.
        if X_tr.equals(X_te):
            X_train, X_test = X_tr.iloc[train_idx], X_tr.iloc[test_idx]
            y_train, y_test = y_tr.iloc[train_idx], y_tr.iloc[test_idx]

        ## If there is a seperate testing data, we train the model with the cross validation training data 
        ## and test it on the cross validited seperate testing data.
        else:
            ## Create model training data based on the CV
            X_train, X_test_tmp = X_tr.iloc[train_idx], X_tr.iloc[test_idx]
            y_train, y_test_tmp = y_tr.iloc[train_idx], y_tr.iloc[test_idx]
            ## Seperately, split the training data
            for train_idx, test_idx in cv.split(X_te, y_te):
                X_train_tmp, X_test = X_te.iloc[train_idx], X_te.iloc[test_idx]
                y_train_tmp, y_test = y_te.iloc[train_idx], y_te.iloc[test_idx]


        ##-----------------------------------
        # 1. Calculate model performance metrics
        ##-----------------------------------
        if model_type == "logistic_regression":
            model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_logistic_regression_model(X_train,  X_test, y_train, y_test)
        elif model_type == "random_forest":
            model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_random_forest_model(X_train,  X_test, y_train, y_test)
        elif model_type == "xgboost":
            model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_xgboost_model(X_train,  X_test, y_train, y_test)
        elif model_type == "mlp":
            model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_mlp_model(X_train,  X_test, y_train, y_test)
        else:
            raise ValueError("Invalid model type. Choose 'logistic_regression', 'random_forest', 'xgboost' or 'mlp'.")
        
        auroc_scores.append(roc_auc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_balanced_scores.append(accuracy_balanced)

        ##-----------------------------------
        # 2. Calculate fairness metrics
        ##-----------------------------------
        if sf_indata == 'Y':
            ## Get sensitive feature values
            sf_data = revert_one_hot_encoding(X_test, benchmark_col, protected_col)

            ## Calculate fairness metrics
            DPD, DPR, EO = calc_fairness_metrics(y_test, y_pred, sf_data)
            ## Also calculate ftu
            FTU = calc_ftu(model, X_test, y_test, benchmark_col, protected_col)
        else:
            ## Append the protected attribute from teh original data
            X_test_original = df_original.iloc[test_idx]
            sf_data = revert_one_hot_encoding(X_test_original, benchmark_col, protected_col)
            ## Calculate fairness metrics
            DPD, DPR, EO = calc_fairness_metrics(y_test, y_pred, sf_data)
            FTU = 0
        
        dpd_scores.append(DPD)
        dpr_scores.append(DPR)
        eo_scores.append(EO)
        ftu_scores.append(FTU)


    # Combining the formatted values with "plus minus" symbol
    result_auroc = mean_plus_minus_std(auroc_scores)
    result_precision = mean_plus_minus_std(precision_scores)
    result_recall = mean_plus_minus_std(recall_scores)
    result_f1 = mean_plus_minus_std(f1_scores)
    result_accuracy_balanced = mean_plus_minus_std(accuracy_balanced_scores)

    result_dpd = mean_plus_minus_std(dpd_scores)
    result_dpr = mean_plus_minus_std(dpr_scores)
    result_eo = mean_plus_minus_std(eo_scores)

    return {
        "Precision": result_precision,
        "Recall": result_recall,
        "F1": result_f1,
        "AUROC": result_auroc,
        "Balanced Accuracy": result_accuracy_balanced,
        "DPD": result_dpd,
        "DPR": result_dpr,
        "EO": result_eo
    }

##-----------------------------------
## Combine them all together
##-----------------------------------
def run_samples(sample_name, df_model, target_name, X_tr, y_tr, X_te, y_te, benchmark_col, protected_col, sf_col_name, df_original, random_state=None, sf_indata='Y'):

    m_results = pd.DataFrame(columns=['sample_name','model_type', 'precision', 'recall', 'f1', 'AUROC', 'accuracy','DPD', 'DPR','EO'])

    ## Run cross validation
    results_logreg = perform_cv(X_tr, y_tr, X_te, y_te, "logistic_regression", benchmark_col, protected_col, sf_col_name, df_original, random_state, sf_indata=sf_indata)
    results_rf = perform_cv(X_tr, y_tr, X_te, y_te, "random_forest", benchmark_col, protected_col, sf_col_name, df_original, random_state, sf_indata=sf_indata)
    results_xgb = perform_cv(X_tr, y_tr, X_te, y_te, "xgboost", benchmark_col, protected_col, sf_col_name, df_original, random_state, sf_indata=sf_indata)
    results_mlp = perform_cv(X_tr, y_tr, X_te, y_te, "mlp", benchmark_col, protected_col, sf_col_name, df_original, random_state, sf_indata=sf_indata)

    ## Add logistic regression results to dataframe
    m_results_logreg = pd.DataFrame([{'sample_name':sample_name, 
                                       'model_type': 'logistic_regression', 
                                       'precision': results_logreg.get('Precision'), 
                                       'recall': results_logreg.get('Recall'), 
                                       'f1': results_logreg.get('F1'), 
                                       'AUROC': results_logreg.get('AUROC'), 
                                       'accuracy': results_logreg.get('Balanced Accuracy'), 
                                       'DPD': results_logreg.get('DPD'), 
                                       'DPR': results_logreg.get('DPR'), 
                                       'EO': results_logreg.get('EO')}])
    m_results = pd.concat([m_results, m_results_logreg], ignore_index=True)
   
    ## Add random forest results to dataframe 
    m_results_rf = pd.DataFrame([{'sample_name':sample_name, 
                                  'model_type': 'random_forest', 
                                  'precision': results_rf.get('Precision'), 
                                  'recall': results_rf.get('Recall'), 
                                  'f1': results_rf.get('F1'), 
                                  'AUROC': results_rf.get('AUROC'), 
                                  'accuracy': results_rf.get('Balanced Accuracy'), 
                                  'DPD': results_rf.get('DPD'), 
                                  'DPR': results_rf.get('DPR'), 
                                  'EO': results_rf.get('EO')}])
    m_results = pd.concat([m_results, m_results_rf], ignore_index=True)

    ## Add xgboost results to dataframe
    m_results_xgb = pd.DataFrame([{'sample_name':sample_name, 
                                   'model_type': 'xgboost', 
                                   'precision': results_xgb.get('Precision'), 
                                   'recall': results_xgb.get('Recall'), 
                                   'f1': results_xgb.get('F1'), 
                                   'AUROC': results_xgb.get('AUROC'), 
                                   'accuracy': results_xgb.get('Balanced Accuracy'), 
                                   'DPD': results_xgb.get('DPD'), 
                                   'DPR': results_xgb.get('DPR'), 
                                   'EO': results_xgb.get('EO')}])
    m_results = pd.concat([m_results, m_results_xgb], ignore_index=True)

    ## Add mlp results to dataframe
    m_results_mlp = pd.DataFrame([{'sample_name':sample_name, 
                                   'model_type': 'mlp', 
                                   'precision': results_mlp.get('Precision'), 
                                   'recall': results_mlp.get('Recall'), 
                                   'f1': results_mlp.get('F1'), 
                                   'AUROC': results_mlp.get('AUROC'), 
                                   'accuracy': results_mlp.get('Balanced Accuracy'), 
                                   'DPD': results_mlp.get('DPD'), 
                                   'DPR': results_mlp.get('DPR'), 
                                   'EO': results_mlp.get('EO')}])
    m_results = pd.concat([m_results, m_results_mlp], ignore_index=True)

    return m_results

