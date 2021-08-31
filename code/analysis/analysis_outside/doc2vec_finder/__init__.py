# *** Doc2Vec_Finder ***
""" Script executes and organizes the three steps Preprocessing, Find_most_similar and Pairing. Checks if the needed data is given or loads it.
    Gets Testdata and returns found Duplicates from Traindata. """

# ## Imports
from services import connection_preparation
from services import manage_dfs
from analysis.analysis_outside.doc2vec_finder import find_most_similar
from analysis import pairing
import os
from pathlib import Path
import yaml
import preprocessing
from typing import Union
import logging
import pandas as pd
import analysis.analysis_outside as analysis_outside

# ## Open Configuration-file and set paths to storing_files (store similarities and token)
with open(Path("config.yaml"), "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    outside_paths = cfg['temp_paths']
    path_to_tokenslist = Path(outside_paths['path_to_tokenslist'])
    path_to_simsdict = Path(outside_paths['path_to_simsdict'])
    path_to_simslisttest = Path(outside_paths['path_to_simslisttest'])
    path_to_simslisttrain = Path(outside_paths['path_to_simslisttrain'])

# ## Functions

# ----- Preprocessing -----
def step_preprocessing(dict_testdata: dict, step_key: str) -> Union[dict, dict]:
    """* preprocessing
            1. Load data if not already loaded 
            2. Pass data to preprocessing and manage returned data
            3. Store preprocessed data in backup file and return it for next step

    Parameters
    ----------
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
    step_key: str
        String with the step_key to define which parts of the program need to be used
    
    Raises
    ------
    KeyError
        Checks if the column in the dataframe does already exist

    Returns
    -------
    dict_testdata_prepro: dict
        Dictionary with preprocessed and tagged testdata -> keys: table_names, values: list(TaggedDocument-objects(tokens_list, unique_id)) 
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads"""

    # Define variables
    dict_testdata_prepro = dict()

    # Check if passed data is the right one
    try:
        check_df = [df['unique_id'] for df in dict_testdata.values()]
    except KeyError:
        check_df = False
    
    # Check if passed data is empty or if column unique_id is missing (means wrong data was delivered)
    if not dict_testdata or check_df == False:
        # GET missing data
        # create connection to testdata and traindata inputf
        conn_test = connection_preparation.conn_testing()
        # load data for testdata
        dict_testdata = manage_dfs.get_df(conn_test)
    else:
        pass

    # Preprocessing
    for name, df in dict_testdata.items():
        test_corpus = preprocessing.preprocess_data(df, step_key)
        dict_testdata_prepro[name] = test_corpus

    # dump preprocessed dict in a backupfile
    analysis_outside.dumper(path_to_tokenslist, dict_testdata_prepro)

    return dict_testdata_prepro, dict_testdata


# ----- find_most_similar -----
def step_most_similar(dict_testdata_prepro: dict) -> Union[dict, list, list]:
    """ Uses the preprocessed testdata (as taggedobjects) to infer them in vectors and find the most_similar job-ads saved in the model.
    
    Parameters
    ----------
    dict_testdata_prepro: dict
        Dictionary with preprocessed and tagged testdata -> keys: table_names, values: list(TaggedDocument-objects(tokens_list, unique_id))

    Returns
    -------
    sims_dict: dict
        Dict contains each testdata unique_id and the depending traindata_unique_id -> keys: testdata unique_id, values: most similar unique_ids from traindata
    testdata_sims_ids: list
        List contains all testdata unique_ids from sims_dict keys
    traindata_sims_ids: list 
        List contains all traindata unique_ids from sims_dict values """

    # check if the dict is filled with token. Else load them from backup_file
    if not dict_testdata_prepro:
        # load preprocessed testdata token
        dict_testdata_prepro = analysis_outside.loader(path_to_tokenslist)
    else:
        pass
    
    logging.info(f'All preparations before the doc2vec_finder are done. Continue with most_similar in doc2vec.')

    # pass data to next module to find duplicates in traindata
    sims_dict, testdata_sim_ids, traindata_sim_ids = find_most_similar.finder(dict_testdata_prepro)

    # dump output files
    analysis_outside.dumper(path_to_simsdict, sims_dict)
    analysis_outside.dumper(path_to_simslisttest, testdata_sim_ids)
    analysis_outside.dumper(path_to_simslisttrain, traindata_sim_ids)
    
    return sims_dict, testdata_sim_ids, traindata_sim_ids

    
# ----- Pairing -----
def step_pairing(sims_dict: dict, testdata_sim_ids: list, traindata_sim_ids: list, dict_testdata: dict, dict_traindata: dict) -> pd.DataFrame:
    """ Uses the stored data in dict_testdata and dict_traindata as lookup-files to extract the matched most similar job-ads (saved in sims_dict).
    
    Parameters
    ----------
    sims_dict: dict
        Dict contains each testdata unique_id and the depending traindata_unique_id -> keys: testdata unique_id, values: most similar unique_ids from traindata
    testdata_sims_ids: list
        List contains all testdata unique_ids from sims_dict keys
    traindata_sims_ids: list 
        List contains all traindata unique_ids from sims_dict values
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
    dict_traindata : dict
        Dictionary with traindata -> keys: table_names, values: DataFrames with job-ads

    Returns
    -------
    df_paris: pd.Dataframe
        Dataframe with the pairwise chosen duplicates. """

    # If one of the passed items is empty, load them again
    if not sims_dict:
        sims_dict = analysis_outside.loader(path_to_simsdict)
    else:
        pass
    if not testdata_sim_ids:
        testdata_sim_ids = analysis_outside.loader(path_to_simslisttest)
    else:
        pass
    if not traindata_sim_ids:
        traindata_sim_ids = analysis_outside.loader(path_to_simslisttrain)
    else:
        pass
    if not dict_testdata:
        conn_test = connection_preparation.conn_testing()
        # load data for testdata
        dict_testdata = manage_dfs.get_df(conn_test)
    else:
        pass
    if not dict_traindata:
        conn_train = connection_preparation.conn_training()
        # load data from trainingdata
        dict_traindata = manage_dfs.get_df(conn_train)
    else:
        pass
    
    logging.info(f'All preparations before the doc2vec_pairing are done. Continue with pairing in doc2vec.')
    # Pass data to Pairing-Module and receive a dataframe with matched job-ads as duplicates
    df_pairs = pairing.pair_outside(sims_dict, testdata_sim_ids, traindata_sim_ids, dict_testdata, dict_traindata)
    return df_pairs