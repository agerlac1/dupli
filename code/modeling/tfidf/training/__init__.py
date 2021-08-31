# *** TF-IDF Training Step ***
""" Script manages the Training of a TF-IDF model. 
    1. Check and Load Data
        * Checks if needed traindata is already passed or needs to be reloaded. 
    2. Preprocess Data
        * Pass data to Preprocessing and get a Dictionary with OneStrings in return.
    3. Pass preprocessed Data to Training
        * Pass preprocessed data to Training."""

# ## Imports
from . import tfidf_training
import modeling
from typing import Union

# ## Define and Set Variables
dict_traindata = dict()
dict_traindata_prepro = dict()

# ## Functions
def step_train_tfidf(dict_traindata: dict, step_key: str) -> Union[dict, dict]:
    """ Manages Loading and Preprocessing of data. Passes preprocessed data to Training.

    Parameters
    ----------
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: job-ads 
    step_key: str
        String with the step_key to define which parts of the program need to be used 
    
    Returns
    -------
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: job-ads 
    dict_traindata_prepro: dict
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed OneStrings) """
    
    # Load Traindata (if not already loaded)
    dict_traindata = modeling.load_traindata(dict_traindata)
    # Preprocess Traindata (return TaggedDocument object saved in dict)
    dict_traindata_prepro = modeling.preprocess_data(dict_traindata, step_key)
    # Train the data
    tfidf_training.train_data(dict_traindata_prepro)

    return dict_traindata, dict_traindata_prepro