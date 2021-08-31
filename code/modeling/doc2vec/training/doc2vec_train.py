# *** Doc2Vec Training***
""" Script trains a Doc2Vec model."""

# ## Imports
import gensim
from pathlib import Path
import yaml
from modeling import doc2vec
import logging

# ## Open Configuration-file and set parameter for model to be trained
with open(Path('config.yaml'), 'r') as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    params = cfg['doc2vec_model']

# ## Functions
def train_data(dict_traindata_prepro: dict):
    """ Method trains a Doc2Vec model with following steps: 
        - Set Parameter for model: (vector_size=100, min_count=1, epochs=25, alpha=0.025) (modification in config.yaml)
        - Build Vocab with traindata (preprocessed)
        - Train the model
        - Save the model 
    
    Parameters
    ----------
    dict_traindata_prepro: dict
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) """

    # SET PARAMETER FOR MODEL
    model = gensim.models.doc2vec.Doc2Vec(vector_size=params['vector_size'], min_count=params['min_count'], epochs=params['epochs'], alpha=params['alpha'])
    logging.info(f'Model Settings: {model}')

    # BUILD VOCAB WITH TRAINDATA
    data = sum(dict_traindata_prepro.values(), [])
    model.build_vocab(data)
    logging.info('Vocab was builded.')

    # TRAIN THE MODEL
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    logging.info(f'For the model {len(model.docvecs)} DocVecs were computed.')
    
    # SAVE MODEL
    doc2vec.save_model(model, 'd2v_model')