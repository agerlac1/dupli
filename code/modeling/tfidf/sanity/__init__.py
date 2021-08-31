# *** TF-IDF Sanity_Check***
""" Script checks the sanity of a TF-IDF model."""

# ## Imports
from . import sanity_check
import modeling
import logging

# ## Set Variables
dict_traindata_prepro = dict()
dict_testdata_prepro = dict()

# ## Functions
def step_sanity_check(dict_testdata: dict, dict_traindata: dict, dict_traindata_prepro: dict, step_key: str) -> None:
    """ Script manages the Loading and Preprocessing of needed Test- and Traindata.
    Afterwards the data is passed to the evaluation step:
        * sanity_check (vocabsize and gives random samples for evaluation)

    Parameters
    ----------
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: job-ads 
    dict_traindata_prepro: dict
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed OneStrings) 
    dict_testdata: dict
        Dictionary with testdata -> keys: table_names, values: job-ads 
    step_key: str
        String with the step_key to define which parts of the program need to be used """

    # Set Variable in Scope
    dict_testdata_prepro = dict()

    # Manage Loading and Preprocessing of Traindata
    if not dict_traindata_prepro:
        dict_traindata_prepro = __get_traindata(dict_traindata, dict_traindata_prepro, step_key)
    else:
        pass
    # Manage Loading and Preprocessing of Testdata
    if not dict_testdata_prepro:
        dict_testdata_prepro = __get_testdata(dict_testdata, dict_testdata_prepro, step_key)
    else:
        pass
    
    # Sanity_Check
    logging.info('\n\n***Sanity_Check***\n')
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