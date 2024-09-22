import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import PrecisionRecallDisplay, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover


##-----------------------------------
## 1. Build logistic regression model
##-----------------------------------
def build_logistic_regression_model(X_train,  X_test, y_train, y_test, bias_mit_method=None, sample_weight=None):

    # Build logistic regression model
    logreg_model = LogisticRegression()
    if bias_mit_method == 'DisparateImpactRemover':
        ## Apply reweighing
        logreg_model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = logreg_model.predict(X_test)
    else:
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
def build_random_forest_model(X_train,  X_test, y_train, y_test, bias_mit_method=None, sample_weight=None):

    # Create a random forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    if bias_mit_method == 'DisparateImpactRemover':
        rf_model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = rf_model.predict(X_test)
    else:
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
def build_xgboost_model(X_train,  X_test, y_train, y_test, bias_mit_method=None, sample_weight=None):
    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    
    # Fit the model to the training dat
    if bias_mit_method == 'DisparateImpactRemover':
        xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = xgb_model.predict(X_test)
    else:
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
def build_mlp_model(X_train,  X_test, y_train, y_test, bias_mit_method=None, sample_weight=None):
    # Create an MLP classifier
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)

    # Fit the model to the training data
    if bias_mit_method == 'DisparateImpactRemover':
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
    DPD = demographic_parity_difference(y, y_pred, sensitive_features=sf_values)
    DPR = demographic_parity_ratio(y, y_pred, sensitive_features=sf_values)

    ##-------------
    ## 2. Equalized odds (EO) FPR, TP / (TP + FN)
    ##-------------
    from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
    EO = equalized_odds_difference(y, y_pred,sensitive_features=sf_values)

    return DPD, DPR, EO



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


## Cross validation
def perform_cv(df_in, model_type, label_name, protected_col,  random_state=None, n_splits=5
               , bias_mit_method=None, privileged_group=None, unprivileged_group=None):

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

    X = df_in.drop([label_name], axis=1)
    y = df_in[label_name]

    for train_idx, test_idx in cv.split(X, y):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
        
        df_train_1 = BinaryLabelDataset(df=df_train, label_names=[label_name], protected_attribute_names=[protected_col])
        df_test_1 = BinaryLabelDataset(df=df_test, label_names=[label_name], protected_attribute_names=[protected_col])

        if bias_mit_method == 'DisparateImpactRemover':
            ## Apply DIR
            di = DisparateImpactRemover(sensitive_attribute=protected_col)
            train_repd = di.fit_transform(df_train_1)
            test_repd = di.fit_transform(df_test_1)
            index = df_train_1.feature_names.index(protected_col)
            X_train = np.delete(train_repd.features, index, axis=1)
            X_test = np.delete(test_repd.features, index, axis=1)
            y_train = train_repd.labels.ravel()
            y_test = test_repd.labels.ravel()

            ##-----------------------------------
            # 1. Calculate model performance metrics
            ##-----------------------------------
            if model_type == "logistic_regression":
                model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_logistic_regression_model(X_train, X_test, y_train, y_test, bias_mit_method=bias_mit_method, sample_weight=None)
            elif model_type == "random_forest":
                model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_random_forest_model(X_train, X_test, y_train, y_test, bias_mit_method=bias_mit_method, sample_weight=None)
            elif model_type == "xgboost":
                model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_xgboost_model(X_train, X_test, y_train, y_test, bias_mit_method=bias_mit_method, sample_weight=None)
            elif model_type == "mlp":
                model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_mlp_model(X_train, X_test, y_train, y_test, bias_mit_method=bias_mit_method, sample_weight=None)
            else:
                raise ValueError("Invalid model type. Choose 'logistic_regression', 'mlp', 'random_forest', or 'xgboost'.")
        
            auroc_scores.append(roc_auc)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            accuracy_balanced_scores.append(accuracy_balanced)

            ##-----------------------------------
            # 2. Calculate fairness metrics
            ##-----------------------------------
            ## Calculate fairness metrics
            DPD, DPR, EO = calc_fairness_metrics(y_test, y_pred, df_test[[protected_col]])

            dpd_scores.append(DPD)
            dpr_scores.append(DPR)
            eo_scores.append(EO)

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
        "EO": result_eo,
    }



##-----------------------------------
## Combine them all together
##-----------------------------------
def run_samples(dataset_name, bias_mitigation_method, df_in, target_name, protected_col, random_state=None
                , privileged_group=None, unprivileged_group=None):

    m_results = pd.DataFrame(columns=['dataset_name','bias_mitigation_method','model_type', 'precision', 'recall','f1', 'AUROC', 'balanced accuracy','DPD','DPR','EO'])

    ## Run cross validation
    results_logreg = perform_cv(df_in, "logistic_regression", target_name, protected_col, random_state, bias_mit_method=bias_mitigation_method, privileged_group=privileged_group, unprivileged_group=unprivileged_group)
    results_rf = perform_cv(df_in, "random_forest",target_name, protected_col, random_state,  bias_mit_method=bias_mitigation_method, privileged_group=privileged_group, unprivileged_group=unprivileged_group)
    results_xgb = perform_cv(df_in, "xgboost",target_name, protected_col, random_state,  bias_mit_method=bias_mitigation_method, privileged_group=privileged_group, unprivileged_group=unprivileged_group)
    results_mlp = perform_cv(df_in, "mlp",target_name, protected_col, random_state,  bias_mit_method=bias_mitigation_method, privileged_group=privileged_group, unprivileged_group=unprivileged_group)

    ## Add logistic regression results to dataframe
    m_results_logreg = pd.DataFrame([{'dataset_name':dataset_name, 
                                      'bias_mitigation_method': "Pre-processing: Disparte Impact Remover",
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
    m_results_rf = pd.DataFrame([{'dataset_name':dataset_name, 
                                  'bias_mitigation_method': "Pre-processing: Disparte Impact Remover",
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
    m_results_xgb = pd.DataFrame([{'dataset_name':dataset_name, 
                                   'bias_mitigation_method': "Pre-processing: Disparte Impact Remover",
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
    m_results_mlp = pd.DataFrame([{'dataset_name':dataset_name, 
                                   'bias_mitigation_method': "Pre-processing: Disparte Impact Remover",
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

