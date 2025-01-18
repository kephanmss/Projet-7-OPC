import pandas as pd
import numpy as np

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
