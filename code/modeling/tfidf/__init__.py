# *** TF-IDF Modeling ***
""" Script manages the two main steps in TF-IDF modeling:
    	* Training
        * Sanity_check
    Both steps can be called at once or repeated separately. 
    
    In Addition the modeling-support functions are noted at the end of the script. 
    The support-functions combine loading and saving of the models for all parts of the program. """

# ## Imports
from . import training
from . import sanity
import sklearn
from pathlib import Path
import yaml
import pickle
import sys
import logging

# ## Define Variables
dict_testdata = dict()
dict_traindata = dict()

# ## Open Configuration-file and set paths to model
with open(Path("config.yaml"), "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    model_paths = cfg['model_paths']
    model_path = model_paths['tfidf_model_path']

# ## Functions
def training_module(args: dict, dict_testdata: dict, dict_traindata: dict, step_key: str) -> None:
    """ training_module (manages all tfidf-modeling related tasks)
            * Training
            * Sanity_check

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: job-ads
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: job-ads 
    step_key: str
        String with the step_key to define which parts of the program need to be used """

    # ## Set Variables
    dict_traindata_prepro = dict()

    # Get values from ArgumentParser
    method_args = vars(args)

    # Depending on Arguments, call parts of the tfidf-modeling
    # Training
    if method_args['training'] == True:
        logging.info('Training of the model for TF-IDF started.')
        dict_traindata, dict_traindata_prepro = training.step_train_tfidf(dict_traindata, method_args['modeling_type'])
        logging.info('Training of the model for TF-IDF finished.')
    # Sanity_Check
    if method_args['sanity_check'] == True:
        logging.info('Sanity_Check of TF-IDF model started.')
        sanity.step_sanity_check(dict_testdata, dict_traindata, dict_traindata_prepro, step_key)
        logging.info('Sanity_Check of TF-IDF model finished.')
    # Training, Sanity_Check
    if method_args['training'] == False and method_args['sanity_check'] == False:
        logging.info('Training of the model for TF-IDF started.')
        dict_traindata, dict_traindata_prepro = training.step_train_tfidf(dict_traindata, method_args['modeling_type'])
        logging.info('Training of the model for TF-IDF finished.')
        logging.info('Sanity_Check of TF-IDF model started.')
        sanity.step_sanity_check(dict_testdata, dict_traindata, dict_traindata_prepro, step_key)
        logging.info('Sanity_Check of TF-IDF model finished.')

# ## Support Functions
""" Methods to load and save models from all parts of the program."""

def save_model(tfidf_model: sklearn.feature_extraction.text.TfidfVectorizer) -> None:
    """ Method saves a passed model in path (set in config.yaml).
        
    Parameters
    ----------
    model: sklearn.feature_extraction.text.TfidfVectorizer
        The model to be saved. Type: TfidfVectorizer """
    
    def __dumper(tfidf_model: sklearn.feature_extraction.text.TfidfVectorizer, model_path: Path):
        with open(Path(model_path), 'wb') as fw:
            pickle.dump(tfidf_model, fw)

    if Path(model_path).exists():
        logging.warning(f'Model {model_path} does already exist, will be overwritten.')
        __dumper(tfidf_model, model_path)
    else:
        __dumper(tfidf_model, model_path)

# Methods to load the tfidf-models (are used by sanity, analysis inside and outside)
def load_model(name: str) -> sklearn.feature_extraction.text.TfidfVectorizer:
    """ Method loads a model depending on chosen name. Path for model is set in config.yaml.
        1. __loader: loads the model and excepts Exceptions
        2. __check_model: checks the received model 

    Parameters
    ----------
    name : str
        Name of the model (tfidf_model)
    
    Raises
    ------
    FileNotFoundError
        Raise Exception if model could not be loaded
    
    Returns
    -------
    model: sklearn.feature_extraction.text.TfidfVectorizer
        The saved model. Type: TfidfVectorizer """

    tfidf_model = None

    def __loader(tfidf_model: None, name: str):
        try:
            if name == 'tfidf_model':
                tfidf_model = pickle.load(open(Path(model_path), 'rb'))
        except FileNotFoundError:
            tfidf_model = None
        return tfidf_model
    tfidf_model = __loader(tfidf_model, name)

    def __check_model(tfidf_model: sklearn.feature_extraction.text.TfidfVectorizer):
        if tfidf_model is not None:
            logging.info(f'Model {name} is loaded and returned to processing step.')
            return tfidf_model
        else:
            logging.error(f'Model {name} failed to be loaded. Check Settings in config.yaml and paths {model_path}.')
            print(f'Model {name} failed to be loaded. Check Settings in config.yaml and paths {model_path}.')
            sys.exit(1)
    tfidf_model = __check_model(tfidf_model)
    return tfidf_model