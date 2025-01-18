def test_imports():
    # Pour le calcul
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

    assert True