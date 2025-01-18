import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils import clean_infinity, reduce_memory_usage, one_hot_encoder # Remplace your_script

def test_clean_infinity():
    df = pd.DataFrame({'col1': [1, np.inf, -np.inf, 2]})
    cleaned_df = clean_infinity(df)
    assert not np.isinf(cleaned_df).any().any()
    assert cleaned_df.isna().any().any()

def test_reduce_memory_usage():
    df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [1.0, 2.0, 3.0, 4.0]})
    original_memory = df.memory_usage().sum()
    reduced_df = reduce_memory_usage(df)
    reduced_memory = reduced_df.memory_usage().sum()
    assert reduced_memory < original_memory

def test_one_hot_encoder():
    df = pd.DataFrame({'col1': ['a', 'b', 'a']})
    encoded_df, new_cols = one_hot_encoder(df)
    assert 'col1_a' in encoded_df.columns
    assert 'col1_b' in encoded_df.columns