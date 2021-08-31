# *** Analysis_outside***
""" Script manages both options for the analysis_outside.

    First option: Doc2Vec
        * preprocessing (preprocesses testdata)
        * mostsim (finds the most similar ids in traindata)
        * pairing (filters found potential duplicates)
            __write_final() (writes found duplicates in output-file)

    !!! IMPORTANT !!! for Doc2Vec: Traindata in analysis_outside and Traindata for model NEED to be the same (unique_ids are stored in model)!
    --> because: Script extracts the most_similar unique_ids from traindata and then loads those job-ads from id_train_data.db.
    --> If the unique_ids from the model don't match with the ones in the input-data, no duplicates can be written in output-file.
    --> So TRAIN a doc2vec-model with traindata and then use this traindata to analyze in analysis_outside.

    Second option: TF-IDF
        * preprocessing (preprocesses testdata)
        * mostsim (finds the most similar ids in traindata)
        * pairing (filters found potential duplicates)
            __write_final() (writes found duplicates in output-file) """

# ## Imports
import pandas as pd
from services import connection_preparation
from services import manage_dfs
import logging
import pickle
import sys

from . import doc2vec_finder
from . import tfidf_finder

# ## Variables
df_pairs = pd.DataFrame()

# ## Functions

def doc2vec_out(args: dict, dict_testdata: dict, dict_traindata: dict) -> None:
    """ analysis_outside (searches for duplicates for the testdata in the traindata).
        Option:
            * doc2vec
        Steps:
            * preprocessing (preprocesses testdata)
            * mostsim (finds the most similar ids in traindata)
            * pairing (filters found potential duplicates)
                __write_final() (writes found duplicates in output-file)

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads """
    
    # Set Variables
    # Dict to store preprocessed testdata
    dict_testdata_prepro = dict()

    # Dict and Lists to store similarity ids
    sims_dict = dict()
    mysimslist_test = list()
    mysimslist_train = list()

    # Compute ArgumentParser values for following steps
    method_args = vars(args)
    # Set step_key needed for preprocessing
    step_key = method_args['method_out']

    # Preprocessing
    if method_args['preprocessing']:
        logging.info('Step Preprocessing in analysis_outside started.')
        dict_testdata_prepro, dict_testdata = doc2vec_finder.step_preprocessing(dict_testdata, step_key)
        logging.info('Step Preprocessing in analysis_outside finished.')
    # Find_most_similar
    if method_args['mostsim']:
        logging.info('Step find_most_similar in analysis_outside started.')
        sims_dict, mysimslist_test, mysimslist_train = doc2vec_finder.step_most_similar(dict_testdata_prepro)
        logging.info('Step find_most_similar in analysis_outside finished.')
    # Pairing
    if method_args['pairing']:
        logging.info('Step pairing in analysis_outside started.')
        df_pairs = doc2vec_finder.step_pairing(sims_dict, mysimslist_test, mysimslist_train, dict_testdata, dict_traindata)
        logging.info('Step pairing in analysis_outside finished.')
        __write_final(df_pairs)
        logging.info('Found duplicates from analysis_outside were stored in output-file.')
    # Preprocessing, Find_most_similar, Pairing
    if method_args['preprocessing'] == False and method_args['mostsim'] == False and method_args['pairing'] == False:
        logging.info('Step Preprocessing in analysis_outside started.')
        dict_testdata_prepro, dict_testdata = doc2vec_finder.step_preprocessing(dict_testdata, step_key)
        logging.info('Step Preprocessing in analysis_outside finished.')
        logging.info('Step find_most_similar in analysis_outside started.')
        sims_dict, mysimslist_test, mysimslist_train = doc2vec_finder.step_most_similar(dict_testdata_prepro)
        logging.info('Step find_most_similar in analysis_outside finished.')
        logging.info('Step pairing in analysis_outside started.')
        df_pairs = doc2vec_finder.step_pairing(sims_dict, mysimslist_test, mysimslist_train, dict_testdata, dict_traindata)
        logging.info('Step pairing in analysis_outside finished.')
        __write_final(df_pairs)
        logging.info('Found duplicates from analysis_outside were stored in output-file.')

def tfidf_out(args: dict, dict_testdata: dict, dict_traindata: dict) -> None:
    """ analysis_outside (searches for duplicates for the testdata in the traindata).
        Option:
            * tfidf
        Steps:
            * preprocessing (preprocesses testdata)
            * mostsim (finds the most similar ids in traindata)
            * pairing (filters found potential duplicates)
                __write_final() (writes found duplicates in output-file)

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads """

    # Set Variables
    # Dicts with preprocessed test- and traindata
    dict_testdata_prepro = dict()
    dict_traindata_prepro = dict()
    # Dicts with the loaded data (not changed) to extract the final duplicates informations to store in output.
    dict_testdata_backup = dict()
    dict_traindata_backup = dict()

    # Dict and Lists to store similarity ids
    sims_dict = dict()
    mysimslist_test = list()
    mysimslist_train = list()

    # Compute ArgumentParser values for following steps
    method_args = vars(args)
    # Set step_key needed for preprocessing
    step_key = method_args['method_out']

    # Preprocessing
    if method_args['preprocessing']:
        logging.info('Step Preprocessing in analysis_outside started.')
        dict_testdata_backup, dict_traindata_backup, dict_testdata_prepro, dict_traindata_prepro = tfidf_finder.step_preprocessing(dict_testdata, step_key)
        logging.info('Step Preprocessing in analysis_outside finished.')
    # Find_most_similar
    if method_args['mostsim']:
        logging.info('Step Find_most_similar in analysis_outside started.')
        sims_dict, mysimslist_test, mysimslist_train = tfidf_finder.step_most_similar(dict_testdata_prepro, dict_traindata_prepro)
        logging.info('Step Find_most_similar in analysis_outside finished.')
    # Pairing
    if method_args['pairing']:
        logging.info('Step Pairing in analysis_outside started.')
        df_pairs = tfidf_finder.step_pairing(sims_dict, mysimslist_test, mysimslist_train, dict_testdata_backup, dict_traindata_backup)
        logging.info('Step Pairing in analysis_outside finished.')
        __write_final(df_pairs)
        logging.info('Found duplicates from analysis_outside were stored in output-file.')
    # Preprocessing, Find_most_similar, Pairing
    if method_args['preprocessing'] == False and method_args['mostsim'] == False and method_args['pairing'] == False:
        logging.info('Step Preprocessing in analysis_outside started.')
        dict_testdata_backup, dict_traindata_backup, dict_testdata_prepro, dict_traindata_prepro = tfidf_finder.step_preprocessing(dict_testdata, step_key)
        logging.info('Step Preprocessing in analysis_outside finished.')
        logging.info('Step Find_most_similar in analysis_outside started.')
        sims_dict, mysimslist_test, mysimslist_train = tfidf_finder.step_most_similar(dict_testdata_prepro, dict_traindata_prepro)
        logging.info('Step Find_most_similar in analysis_outside finished.')
        logging.info('Step Pairing in analysis_outside started.')
        df_pairs = tfidf_finder.step_pairing(sims_dict, mysimslist_test, mysimslist_train, dict_testdata_backup, dict_traindata_backup)
        logging.info('Step Pairing in analysis_outside finished.')
        __write_final(df_pairs)
        logging.info('Found duplicates from analysis_outside were stored in output-file.')

def __write_final(df_pairs: pd.DataFrame):
    # create connection to output for final duplicates list (duplicates from train_data)
    conn = connection_preparation.conn_final_output()
    # write the df in output
    table_name = 'traindata_output'
    if len(df_pairs) == 0:
        logging.warning('No similarities were found. If method Doc2Vec was used, check if unique_ids in model and train_data match.')
    else:
        manage_dfs.write_df_output(conn, table_name, df_pairs)

# Support Functions 

def dumper(path, data_to_dump):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data_to_dump, f)
        logging.info(f'File {path} was saved.')
    except FileExistsError:
        logging.warning(f'File {path} does already exists. Will be overwritten.')

def loader(path):
    data_to_load = None
    try:
        with open(path, 'rb') as f:
            data_to_load = pickle.loads(f.read())
        logging.info(f'File {path} was loaded.')
    except FileNotFoundError:
        logging.info(f'Program stopped. Analysis_outside needs to be restarted with all steps. Missing file {path} does not exist and variables cannot be filled.')
        print(f'Program stopped. Analysis_outside needs to be restarted with all steps. Missing file {path} does not exist and variables cannot be filled.')
        sys.exit(1)
    return data_to_load