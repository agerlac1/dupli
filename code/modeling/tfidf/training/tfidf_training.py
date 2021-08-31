# *** TF-IDF Training***
""" Script trains a TF-IDF model."""

# ## Imports
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
from pathlib import Path
from modeling import tfidf
import logging
import itertools

# ## Open Configuration-file and set parameter for model to be trained
with open(Path('config.yaml'), 'r') as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    params = cfg['tfidf_model']

# ## Functions
def train_data(dict_traindata_prepro: dict):
    """ Method trains a TF-IDF model with following steps: 
        - Set Parameter for model: 'sublinear_tf' == True or False (modification in config.yaml)
        - Initiate TfidfVectorizer()-object
        - Fit Vocab with traindata (preprocessed)
        - Save the model 
    
    Parameters
    ----------
    dict_traindata_prepro: dict
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed OneStrings) """

    # SET PARAMETER AND INITIATE TFIDFVECTORIZER
    tfidftransformer = TfidfVectorizer(sublinear_tf = params['sublinear_tf'])
    logging.info(f'Model Settings: {tfidftransformer}')

    # MANAGE DATA AND FIT THE VOCAB
    train_data = list(itertools.chain(*[sum([list(v.values())], []) for v in dict_traindata_prepro.values()]))
    tfidf_model = tfidftransformer.fit(train_data)
    logging.info('Vocab was fitted.')
    logging.info(f'For the model {tfidf_model} the vocab size is: {len(tfidf_model.get_feature_names())}.')
 
    # SAVE MODEL
    tfidf.save_model(tfidf_model)