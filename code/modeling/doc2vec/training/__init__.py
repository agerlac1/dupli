# *** Doc2Vec Training Step ***
""" Script manages the Training of a Doc2Vec model. 
    1. Check and Load Data
        * Checks if needed traindata is already passed or needs to be reloaded. 
    2. Preprocess Data
        * Pass data to Preprocessing and get a Dictionary with TaggedDocument Objects in return.
    3. Pass preprocessed Data to Training
        * Pass preprocessed data to Training."""

# ## Imports
from . import doc2vec_train
from typing import Union
import modeling

# ## Define and Set Variables
dict_traindata = dict()
dict_traindata_prepro = dict()

# ## Functions
def step_train_doc2vec(dict_traindata: dict, step_key: str) -> Union[dict, dict]:
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
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) """
    
    # Load Traindata (if not already loaded)
    dict_traindata = modeling.load_traindata(dict_traindata)
    # Preprocess Traindata (return TaggedDocument object saved in dict)
    dict_traindata_prepro = modeling.preprocess_data(dict_traindata, step_key)
    # Train the data
    doc2vec_train.train_data(dict_traindata_prepro)

    return dict_traindata, dict_traindata_prepro