import pandas as pd
import sys
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.path.abspath('.'))
from utility import get_data
from ml_model_dir import run_samples

seed_ml = 123

bias_mitigation_method = 'DisparateImpactRemover'
dataset_names = ['adult', 'dutch', 'twc', 'compas', 'law']
protected_cols = ['sex', 'sex', 'gender_cat', 'race_cat', 'race']
target_names = ['income_f', 'occupation', 'default_f', 'two_year_recid', 'admit']

m_results_f = pd.DataFrame()

for i, df_name in enumerate(dataset_names):

    protected_col = protected_cols[i]
    target_name = target_names[i]
    random_state=seed_ml

    dataset_orig, privileged_group, unprivileged_group, optim_options = get_data(df_name, protected_col)

    print(f"Processing {df_name}")

    # Call your custom function here and pass the dataframe as an argument
    result = run_samples(dataset_name=df_name, bias_mitigation_method=bias_mitigation_method, df_in=dataset_orig
                         , target_name=target_name,  protected_col=protected_col, random_state=random_state
                         , privileged_group=privileged_group, unprivileged_group=unprivileged_group)

    m_results_f = pd.concat([m_results_f, result])

m_results_f.to_csv('results/preprocessing_disparateimpactremover_202402306.csv', index=False)
