# *** Modeling ***
""" Script manages the two different modeling steps:
        1. doc2vec (calls module doc2vec in modeling)
        2. tfidf (calls module tfidf in modeling) """

# ## Imports
from . import doc2vec
from . import tfidf
import logging
from services import connection_preparation
from services import manage_dfs
import preprocessing

# ## Functions
def modeling_dist(args:dict, dict_testdata: dict, dict_traindata: dict) -> None:
    """ modeling_dist (distributes next program steps)
            * doc2vec
            * tfidf

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: job-ads
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: job-ads """

    # Get values from ArgumentParser
    method_args = vars(args)
    logging.info('Modeling started.')
    # Depending on Arguments, call parts of the modeling
    # Doc2Vec
    if method_args['modeling_type'] == 'doc2vec':
        logging.info('Modeling with Doc2Vec started.')
        doc2vec.training_module(args, dict_testdata, dict_traindata, method_args['modeling_type'])
        logging.info('Modeling with Doc2Vec finished.')
    # TF-IDF
    elif method_args['modeling_type'] == 'tfidf':
        logging.info('Modeling with TF-IDF started.')
        tfidf.training_module(args, dict_testdata, dict_traindata, method_args['modeling_type'])
        logging.info('Modeling with TF-IDF finished.')
    

# Support functions to manage loading of data and preprocessing in all steps (doc2vec: training, retraining, sanity_check and tfidf: training, sanity_check)
# ------

def load_testdata(dict_testdata:dict) -> dict:
    if not dict_testdata:
        # get connection to data to be retrained (aka testdata)
        conn1 = connection_preparation.conn_testing()
        # Read dataframe as if it would be traindata (only need id and fulltext because of that use get_df_train)
        dict_testdata = manage_dfs.get_df_train(conn1)
    else:
        pass
    return dict_testdata

def load_traindata(dict_traindata:dict) -> dict:
    # Load the data
    if not dict_traindata:
        # get connection to train data
        conn1 = connection_preparation.conn_training()
        # Read dataframe
        dict_traindata = manage_dfs.get_df_train(conn1)
    else:
        pass
    return dict_traindata

def preprocess_data(dict_data: dict, step_key: str) -> dict:
    dict_data_prepro = dict()
    # Iterate over each table in data
    for name, df in dict_data.items():
        # Preprocess each dataframe row by row
        logging.info(f'Preprocessing for table {name} started.')
        test_corpus = preprocessing.preprocess_data(df, step_key)
        dict_data_prepro[name] = test_corpus
    return dict_data_prepro