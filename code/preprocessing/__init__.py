""" Calls module-script depending on key. """

# ## Imports
from . import preprocessing_data
import pandas as pd

def preprocess_data(df: pd.DataFrame, key: str) -> list or dict:
    if key == "doc2vec":
        data = preprocessing_data.preprocess_labeled(df)
        return data
    elif key == "tfidf" or key == 'preprocessing':
        data = preprocessing_data.preprocess_strings(df)
        return data