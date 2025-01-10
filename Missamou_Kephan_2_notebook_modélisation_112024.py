#%% # Pour le calcul
import numpy as np
import pandas as pd

# Pour plotter
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn pour le Machine Learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, fbeta_score, roc_auc_score

# Librairie imbalanced-learn pour le rééquilibrage des classes par undersampling
# Pas d'undersampling par sélection aléatoire bête et méchante
from imblearn.ensemble import BalancedRandomForestClassifier

import os, sys

# Pour ignorer les warnings
import warnings
warnings.filterwarnings('ignore')

# Pour MLflow
import mlflow
import mlflow.sklearn

# Pour XGBoost
from xgboost import XGBClassifier

# Pour les valeurs shapely
import shap
#%%

def main():
    debug = False

    def clean_infinity(df, lower_bound=-1e10, upper_bound=1e10):
        """
        Remplace les valeurs infinies par NaN, plafonne les valeurs extrêmes 
        et gère les colonnes non numériques.

        Args:
            df (pd.DataFrame): DataFrame à nettoyer.
            lower_bound (float): Valeur de plafonnement inférieure.
            upper_bound (float): Valeur de plafonnement supérieure.

        Returns:
            pd.DataFrame: DataFrame nettoyé.
        """
        df = df.copy()

        # Remplacement des infinis
        df = df.replace([np.inf, -np.inf], np.nan)

        # Plafonnement explicite pour chaque colonne numérique
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def reduce_memory_usage(df):
        """
        Réduit l'utilisation de la mémoire en optimisant les types de données
        """
        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)

        return df

    # One-hot encoding for categorical columns with get_dummies
    def one_hot_encoder(df, nan_as_category = True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns

    # Preprocess application_train.csv and application_test.csv
    def application_train_test(num_rows = None, nan_as_category = False):
        # Read data and merge
        df = pd.read_csv('./Data/application_train.csv', nrows= num_rows)
        test_df = pd.read_csv('./Data/application_test.csv', nrows= num_rows)
        print(f"Train samples: {len(df)}, test samples: {len(test_df)}")
        df = pd.concat([df, test_df]).reset_index()
        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        df = df[df['CODE_GENDER'] != 'XNA']

        # Categorical features with Binary encode (0 or 1; two categories)
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])
        # Categorical features with One-Hot encode
        df, cat_cols = one_hot_encoder(df, nan_as_category)

        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        # Some simple new features (percentages)
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df.loc[~df['TARGET'].isna(),:]
        del test_df
        return df

    # Preprocess bureau.csv and bureau_balance.csv
    def bureau_and_balance(num_rows = None, nan_as_category = True):
        bureau = pd.read_csv('./Data/bureau.csv', nrows = num_rows)
        bb = pd.read_csv('./Data/bureau_balance.csv', nrows = num_rows)
        bb, bb_cat = one_hot_encoder(bb, nan_as_category)
        bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
        del bb, bb_agg

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }
        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat: cat_aggregations[cat] = ['mean']
        for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
        del active, active_agg
        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
        del closed, closed_agg, bureau
        return bureau_agg

    # Preprocess previous_applications.csv
    def previous_applications(num_rows = None, nan_as_category = True):
        prev = pd.read_csv('./Data/previous_application.csv', nrows = num_rows)
        prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['min', 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'max', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']

        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        # Previous Applications: Approved Applications - only numerical features
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        del refused, refused_agg, approved, approved_agg, prev
        return prev_agg

    # Preprocess POS_CASH_balance.csv
    def pos_cash(num_rows = None, nan_as_category = True):
        pos = pd.read_csv('./Data/POS_CASH_balance.csv', nrows = num_rows)
        pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
        # Features
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']

        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        # Count pos cash accounts
        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
        del pos
        return pos_agg

    def installments_payments(num_rows = None, nan_as_category = True):
        ins = pd.read_csv('./Data/installments_payments.csv', nrows = num_rows)
        ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
        # Percentage and difference paid in each installment (amount paid and installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
            'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        del ins
        return ins_agg

    # Preprocess credit_card_balance.csv
    def credit_card_balance(num_rows = None, nan_as_category = True):
        cc = pd.read_csv('./Data/credit_card_balance.csv', nrows = num_rows)
        cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
        # General aggregations
        cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
        cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        del cc
        return cc_agg

    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    bureau = bureau_and_balance(num_rows)
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    prev = previous_applications(num_rows)
    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    pos = pos_cash(num_rows)
    print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    ins = installments_payments(num_rows)
    print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    cc = credit_card_balance(num_rows)
    print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc

    X = df.loc[df['TARGET'].notna(),[col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]]
    y = df.loc[df['TARGET'].notna(),'TARGET']

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    print(f'Len numerical_cols: {len(numerical_cols)}')
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f'Len categorical_cols: {len(categorical_cols)}')
    print(f'X[numerical_cols] shape: {X[numerical_cols].shape}')
    # 1. Pour les variables numériques
    # Imputation des valeurs manquantes
    num_imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
    X_num_imputed = num_imputer.fit_transform(clean_infinity(X[numerical_cols]))
    X_num_imputed = pd.DataFrame(X_num_imputed, columns=numerical_cols)	

    # Standardisation
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num_imputed)
    X_num_scaled_df = pd.DataFrame(X_num_scaled, columns=numerical_cols)

    # 2. Pour les variables catégorielles
    # Imputation des valeurs manquantes
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat_imputed = cat_imputer.fit_transform(X[categorical_cols])

    # Encodage one-hot
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_encoded = onehot.fit_transform(X_cat_imputed)
    X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=onehot.get_feature_names_out(categorical_cols))

    feature_names = list(numerical_cols) + list(onehot.get_feature_names_out(categorical_cols))

    # Concaténation horizontale des features
    print(f'X_num_scaled_df shape: {X_num_scaled_df.shape}')
    print(f'X_cat_encoded_df shape: {X_cat_encoded_df.shape}')

    X_preprocessed = pd.concat([X_num_scaled_df, X_cat_encoded_df], axis=1)

    # 4. Séparation des données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Feature names du train set

    feature_names = X_train.columns

    # fbêta score de sklearn.metrics
    fbeta_scoring = make_scorer(
        fbeta_score,
        beta=0.7
        )

    # stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_forest = RandomForestClassifier(random_state=42)
    regression_logistique = LogisticRegression(random_state=42)
    sgd = SGDClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42)

    param_grid = {
        'n_estimators': [n for n in range(10, 201, 10)], # Up to 200, inclusive
        'max_depth': [d for d in range(2, 21, 2)], # Up to 20 inclusive
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': ['balanced_subsample', 'balanced'],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None] # Removed 'auto' as it is deprecated
        }

    grid_search = RandomizedSearchCV(
        estimator=random_forest,
        param_distributions=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        n_iter=5,
        scoring=fbeta_scoring,
        return_train_score=True
        )

    mlflow.set_experiment("Test projet 7")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(run_name="RandomForestClassifier Prod 1"):
        grid_search.fit(X_train, y_train)

        mlflow.set_tags({
                "author": "Missamou Kephan",
                "model": "RandomForestClassifier",
                "model_type": "classification",
                "stratification": "False",
                "class_rebalancing": "True",
                'rebalancing_strategy': "class_weights : balanced_subsample ou balanced",
                "bêta": "0.7",
                "numerical_imputer": "SimpleImputer (mean)",
                "categorical_imputer": "SimpleImputer (most_frequent)",
                "scaler": "StandardScaler",
                "feature_engineering": "True",
                "feature_selection": "False",
                "hyperparameter_optimization": "True",
                "optimization_strategy": "RandomizedSearchCV",
                "cross_validation": "5 folds",
                "n_iter": "50",
                "scoring": "fbeta_score"
                })

        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        y_pred = grid_search.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report['weighted avg']['fbeta-score'] = fbeta_score(y_test, y_pred, beta=0.7)
        mlflow.log_metrics({
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score'],
            'support': report['weighted avg']['support'],
            'fbeta-score': report['weighted avg']['fbeta-score'],
            'roc_auc': roc_auc_score(y_test, y_pred)
            })
        # Log the confusion matrix
        confusion = confusion_matrix(y_test, y_pred)
        # Save confusion matrix as an image file
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Log confusion matrix image as an artifact
        mlflow.log_artifact('confusion_matrix.png')

        # Feature Importances
        best_model = grid_search.best_estimator_

        importances = best_model.feature_importances_

        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Trier les importances des caractéristiques par ordre décroissant
        feature_importances = feature_importances.sort_values('importance', ascending=False)

        # Conserver les features qui expliquent 80% de l'importance totale
        cumulative_importances = np.cumsum(feature_importances['importance'])
        keep_features = feature_importances[cumulative_importances <= 0.8]

        # Tracer les importances des caractéristiques
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances.loc[feature_importances['feature'].isin(keep_features['feature']),:])
        plt.title('Feature Importances')
        plt.savefig('feature_importances.png')
        plt.close()

        # Enregistrer l'image des importances des caractéristiques en tant qu'artefact
        mlflow.log_artifact('feature_importances.png')

        # Utiliser un échantillon du jeu de données pour le calcul des valeurs SHAP
        X_shap_sample = X_train.sample(1000, random_state=42)

        # Créer un explainer SHAP
        explainer = shap.TreeExplainer(best_model)

        # Calculer les valeurs SHAP pour l'échantillon
        shap_values = explainer.shap_values(X_shap_sample)
        final_shap_values = np.abs(shap_values).mean(axis=(0, 2))
 
        # Récupérer les noms des features pour notre échantillon
        feature_names = X_shap_sample.columns

        # Créer un DataFrame pour les importances de caractéristiques SHAP
        # Use the correct feature names after one-hot encoding
        shap_importances = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': final_shap_values
        })

        # Trier les importances de caractéristiques SHAP par ordre décroissant
        shap_importances = shap_importances.sort_values('shap_importance', ascending=False)
        # Conserver les 20 features shap les plus importantes
        keep_features = shap_importances.head(20)

        # Barplot des importances de caractéristiques SHAP
        plt.figure(figsize=(12, 8))
        sns.barplot(x='shap_importance', y='feature', data=shap_importances.loc[shap_importances['feature'].isin(keep_features['feature']),:])
        plt.title('SHAP Feature Importances')
        plt.savefig('shap_feature_importances.png')
        plt.close()

        # Enregistrer l'image des importances de caractéristiques SHAP en tant qu'artefact
        mlflow.log_artifact('shap_feature_importances.png')

        # Signature, input example et output example
        signature = {
            'input': 'pandas.DataFrame',
            'output': 'numpy.ndarray'
        }

        mlflow.sklearn.log_model(best_model, 'model')

if __name__ == "__main__":
    mlflow.end_run()
    main()