# *** Doc2Vec Sanity_Check***
""" Script checks the sanity of a Doc2Vec model."""

# ## Imports
import modeling
from . import sanity_check
import logging

# ## Set Variables
dict_traindata_prepro = dict()
dict_testdata_prepro = dict()

# ## Functions
def step_sanity_check(dict_testdata: dict, dict_traindata: dict, dict_testdata_prepro: dict, dict_traindata_prepro: dict, step_key: str) -> None:
    """ Script manages the Loading and Preprocessing of needed Test- and Traindata.
    Afterwards the data is passed to the evaluation steps:
        1. sanity_check (checks if data finds itself as most similar)
        2. evaluate_sample (gives random sample for evaluation)

    Parameters
    ----------
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: job-ads 
    dict_traindata_prepro: dict
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) 
    dict_testdata: dict
        Dictionary with testdata -> keys: table_names, values: job-ads 
    dict_testdata_prepro: dict
        Dictionary with testdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) 
    step_key: str
        String with the step_key to define which parts of the program need to be used """

    # Manage Loading and Preprocessing of Testdata
    if not dict_testdata_prepro:
        dict_testdata_prepro = __get_testdata(dict_testdata, dict_testdata_prepro, step_key)
    else:
        pass
    # Manage Loading and Preprocessing of Traindata
    if not dict_traindata_prepro:
        dict_traindata_prepro = __get_traindata(dict_traindata, dict_traindata_prepro, step_key)
    else:
        pass

    # Sanity_Check
    logging.info('\n\n***Sanity_Check***\n')
    sanity_check.start_sanity_check(dict_traindata_prepro)
    sanity_check.evaluate_sample(dict_testdata_prepro, dict_traindata_prepro)
    
# Manage Loading and Preprocessing of Testdata
def __get_testdata(dict_testdata, dict_testdata_prepro, step_key):
    dict_testdata = modeling.load_testdata(dict_testdata)
    dict_testdata_prepro = modeling.preprocess_data(dict_testdata, step_key)
    return dict_testdata_prepro

# Manage Loading and Preprocessing of Testdata
def __get_traindata(dict_traindata, dict_traindata_prepro, step_key):
    dict_traindata = modeling.load_traindata(dict_traindata)
    dict_traindata_prepro = modeling.preprocess_data(dict_traindata, step_key)
    return dict_traindata_prepro