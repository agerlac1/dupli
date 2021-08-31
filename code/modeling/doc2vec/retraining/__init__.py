# *** Doc2Vec Retraining Step ***
""" Script manages the Training of a Doc2Vec model. 
    1. Check and Load Testdata
        * Checks if needed testdata is already passed or needs to be reloaded. 
    2. Preprocess Data
        * Pass data to Preprocessing and get a Dictionary with TaggedDocument Objects in return.
    3. Pass preprocessed Data to Retraining
        * Pass preprocessed data to Retraining."""

# ## Imports
from modeling import doc2vec
from . import doc2vec_retrain
from typing import Union
import logging
import modeling

# ## Define and Set Variables
dict_testdata = dict()
dict_testdata_prepro = dict()

# ## Functions

def step_retrain_doc2vec(dict_testdata: dict, step_key: str) -> Union[dict, dict]:
    """ Manages Loading and Preprocessing of data. Passes preprocessed data to Retraining.

    Parameters
    ----------
    dict_testdata: dict
        Dictionary with testdata -> keys: table_names, values: job-ads 
    step_key: str
        String with the step_key to define which parts of the program need to be used 
    
    Returns
    -------
    dict_testdata: dict
        Dictionary with testdata -> keys: table_names, values: job-ads 
    dict_testdata_prepro: dict
        Dictionary with testdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) """

    # Load Testdata (if not already loaded)
    dict_testdata = modeling.load_testdata(dict_testdata)
    # Preprocess Testdata (return TaggedDocument object saved in dict)
    dict_testdata_prepro = modeling.preprocess_data(dict_testdata, step_key)
    # Retrain the data
    doc2vec_retrain.retrain_data(dict_testdata_prepro)
    
    return dict_testdata, dict_testdata_prepro