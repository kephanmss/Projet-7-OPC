import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

import warnings
warnings.filterwarnings('ignore')

def increment_run_counter(file_path='run_counter.txt'):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('0')
    with open(file_path, 'r') as f:
        run_counter = int(f.read()) + 1
    with open(file_path, 'w') as f:
        f.write(str(run_counter))
    return run_counter

def clean_infinity(df, lower_bound=-1e10, upper_bound=1e10):
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].clip(lower_bound, upper_bound)
    return df

def one_hot_encoder(df, nan_as_category=True):
    cat_cols = df.select_dtypes('object').columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, dummy_na=nan_as_category)
    new_cols = df_encoded.columns.difference(df.columns).tolist()
    return df_encoded, new_cols

def application_train_test(num_rows=None):
    train = pd.read_csv('./Data/application_train.csv', nrows=num_rows)
    test = pd.read_csv('./Data/application_test.csv', nrows=num_rows)
    df = pd.concat([train, test]).reset_index(drop=True)
    df = df[df['CODE_GENDER'] != 'XNA']
    for col in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[col], _ = pd.factorize(df[col])
    df, _ = one_hot_encoder(df)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df

if __name__ == '__main__':
    run_counter = increment_run_counter()
    debug = True

    num_rows = 50000 if debug else None
    df = application_train_test(num_rows)

    X = df.loc[~df['TARGET'].isna()].drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = df.loc[df['TARGET'].notna(), 'TARGET']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include='object').columns

    # Numerical preprocessing
    X_num = clean_infinity(X[num_cols])
    num_imputer = SimpleImputer(strategy='mean')
    X_num_imputed = pd.DataFrame(num_imputer.fit_transform(X_num), columns=num_cols)

    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num_imputed), columns=num_cols)

    # Categorical preprocessing
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X[cat_cols]), columns=cat_cols)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_encoded = pd.DataFrame(encoder.fit_transform(X_cat_imputed), columns=encoder.get_feature_names_out(cat_cols))

    # Concatenate
    X_preprocessed = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

    # Data drift report
    X_old, X_new = train_test_split(X_preprocessed, test_size=0.5, random_state=42)

    drift_report = Report([DataDriftPreset(drift_type='data_drift', warnings=False)])
    drift_report.calculate(X_ancien, X_nouveau)
    drift_report.save(f'data_drift_report_{run_counter}.html')

    print(f"Run {run_counter} completed. Report saved.")