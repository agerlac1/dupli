# *** Doc2Vec Modeling ***
""" Script manages the three main steps in Doc2Vec modeling:
    	* Training
        * Retraining
        * Sanity_check
    All steps can be called at once or repeated separately. 
    
    In Addition the modeling-support functions are noted at the end of the script. 
    The support-functions combine loading and saving of the models for all parts of the program. """

# ## Imports
from . import training
from . import retraining
from . import sanity
from gensim.models import doc2vec
from pathlib import Path
import yaml
import logging
import sys

# ## Define Variables
dict_testdata = dict()
dict_traindata = dict()

# ## Open Configuration-file and set paths to models (trained and retrained)
with open(Path('config.yaml'), 'r') as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    model_paths = cfg['model_paths']
    model_path = model_paths['model_path']
    retrained_model_path = model_paths['retrained_model_path']

# ## Functions
def training_module(args: dict, dict_testdata: dict, dict_traindata: dict, step_key: str) -> None:
    """ training_module (manages all doc2vec-modeling related tasks)
            * Training
            * Retraining
            * Sanity_check (adjust in config.yaml manually, which kind of model is evaluated (d2v_model_type))

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

    # Set variables for function
    dict_testdata_prepro = dict()
    dict_traindata_prepro = dict()

    # Get values from ArgumentParser
    method_args = vars(args)
    
    # Depending on Arguments, call parts of the doc2vec-modeling
    # Training
    if method_args['training'] == True:
        logging.info('Training of the model for Doc2Vec started.')
        dict_traindata, dict_traindata_prepro = training.step_train_doc2vec(dict_traindata, step_key)
        logging.info('Training of the model for Doc2Vec finished.')
    # Retraining
    if method_args['retraining'] == True:
        logging.info('Retraining of the model for Doc2Vec started.')
        dict_testdata, dict_testdata_prepro = retraining.step_retrain_doc2vec(dict_testdata, step_key)
        logging.info('Retraining of the model for Doc2Vec finished.')
    # Sanity_Check
    if method_args['sanity_check'] == True:
        logging.info('Sanity_check of a Doc2Vec-model started.')
        sanity.step_sanity_check(dict_testdata, dict_traindata, dict_testdata_prepro, dict_traindata_prepro, step_key)
        logging.info('Sanity_check of a Doc2Vec-model finished.')
    # Training, Retraining, Sanity_Check
    if method_args['training'] == False and method_args['retraining'] == False and method_args['sanity_check'] == False:
        logging.info('Training of the model for Doc2Vec started.')
        dict_traindata, dict_traindata_prepro = training.step_train_doc2vec(dict_traindata, step_key)
        logging.info('Training of the model for Doc2Vec finished.')
        logging.info('Retraining of the model for Doc2Vec started.')
        dict_testdata, dict_testdata_prepro = retraining.step_retrain_doc2vec(dict_testdata, step_key)
        logging.info('Retraining of the model for Doc2Vec finished.')
        logging.info('Sanity_check of a Doc2Vec-model started.')
        sanity.step_sanity_check(dict_testdata, dict_traindata, dict_testdata_prepro, dict_traindata_prepro, step_key)
        logging.info('Sanity_check of a Doc2Vec-model finished.')

# ## Support Functions
""" Methods to load and save doc2vec-models from all parts of the program."""

def save_model(model: doc2vec.Doc2Vec, name: str) -> None:
    """ Method saves a model depending on chosen name.
    Doc2Vec has name options d2v_model or d2v_remodel. Can be modified in config.yaml.
    Paths are stored in config.yaml too.
        
    Parameters
    ----------
    name : str
        Name of the model (or model-type). Value ajdustment in config.yaml: d2v_model or d2v_remodel 
    model: doc2vec.Doc2Vec
        The model to be saved. Type: gensim.models.doc2vec.Doc2Vec """
    
    def __saver(path: Path, model: doc2vec.Doc2Vec):
        if Path(path).exists():
            logging.warning(f'Model {name} does already exist, will be overwritten.')
            model.save(path)
        else:
            model.save(path)

    # Training model
    if name == 'd2v_model':
        path = model_path
        __saver(path, model)
    # Retraining model
    elif name == 'd2v_remodel':
        path = retrained_model_path
        __saver(path, model)

# Load the model
def load_model(name: str) -> doc2vec.Doc2Vec:
    """ Method loads a model depending on chosen name.
    Doc2Vec has name options d2v_model or d2v_remodel. Can be modified in config.yaml.
    Paths are stored in config.yaml too.
        1. __loader: loads the model and excepts Exceptions
        2. __check_model: checks the received model 

    Parameters
    ----------
    name : str
        Name of the model (or model-type). Value ajdustment in config.yaml: d2v_model or d2v_remodel 
    
    Raises
    ------
    FileNotFoundError
        Raise Exception if model could not be loaded
    
    Returns
    -------
    model: doc2vec.Doc2Vec
        The saved model. Type: gensim.models.doc2vec.Doc2Vec """

    model = None
    def __loader(model: None, name: str):
        try:
            if name == 'd2v_model':
                model = doc2vec.Doc2Vec.load(model_path)
            elif name == 'd2v_remodel':
                model = doc2vec.Doc2Vec.load(retrained_model_path)
        except FileNotFoundError:
            model = None
        return model
    model = __loader(model, name)

    def __check_model(model: doc2vec.Doc2Vec):
        if model is not None:
            logging.info(f'Model {name} is loaded and returned to processing step.')
            return model
        else:
            logging.error(f'Model {name} failed to be loaded. Check Settings in config.yaml and paths {model_path}, {retrained_model_path}.')
            print(f'Model {name} failed to be loaded. Check Settings in config.yaml and paths {model_path}, {retrained_model_path}.')
            sys.exit(1)
    model = __check_model(model)
    return model