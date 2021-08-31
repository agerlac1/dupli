# ## ID_HANDLING
"""
Script gives unique_ids to all used input data (Trainingdata and Testdata).
Output: id_train_data.db or id_test_data.db files stored in /temp. 

Purpose: Give unique_ids to reidentify each job-ad. 

* Last used unique_id is saved in file 'last_unique_id.txt'.
* If you want to redo a step of handling, you need to reset the last_handle manually.
"""

# ## Imports
from services import connection_preparation
from services import manage_dfs
from . import give_unique_ids

import logging
from pathlib import Path
import yaml
from typing import Union
import sqlite3

# ## Define Variables
conn_train = None
conn_test = None
conn_output_test = None
conn_output_train = None

# ## Open Configuration-file and set path
with open(Path("config.yaml"), "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    path_last_id = Path(cfg['id_support'])

# ## Functions

def id_manager(args: dict, dict_testdata: dict, dict_traindata: dict) -> Union[dict, dict]:
    ''' Function manages the arguments passed by the ArgumentParser to decide, which data gets unique_ids.
        
        * Build connection to input and output databases
        * Depending on the given parameter (can be "test" or "train"), different input and output files will be chosen
        * --test: gets as input the raw-test-data and stores the final data in the output file temp/id_test_data.db
        * --train: gets as input the raw-train-data and stores the final data in the output file temp/id_train_data.db

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads
    
    Returns
    -------
    dict_testdata : dict
        Dictionary with testdata AND UNIQUE_IDS -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata AND UNIQUE_IDS -> keys: table_names, values: DataFrames with job-ads
    '''
    
    method_args = vars(args)
    if method_args['train'] == True:
        logging.info('Give unique_ids for Trainingdata.')
        # Process Trainingdata
        dict_traindata = __process_traindata()
    if method_args['test'] == True:
        logging.info('Give unique_ids for Testdata.')
        # Process Testdata
        dict_testdata = __process_testdata()
    if method_args['train'] == False and method_args['test'] == False :
        logging.info('Give unique_ids for Trainingdata.')
        # Process Trainingdata
        dict_traindata = __process_traindata()
        logging.info('Give unique_ids for Testdata.')
        # Process Testdata
        dict_testdata = __process_testdata()
    return dict_testdata, dict_traindata

def __process_traindata() -> dict:
    # Create Connection to Trainingdata (input)
    conn_train = connection_preparation.connect_raw_train_data()
    logging.info(f'Connection to Trainingdatabase {conn_train} was established.')
    # Create Conncetion to Output Database to store Trainingdata with unique_ids
    conn_output_train = connection_preparation.conn_training()
    logging.info(f'Connection to Output File /temp/id_train_data.db {conn_output_train} was established.')
    dict_traindata = __get_structure(conn_train, conn_output_train)
    return dict_traindata

def __process_testdata() -> dict:
    # Create Connection to Testdata (input)
    conn_test = connection_preparation.connect_raw_test_data()
    logging.info(f'Connection to Testdatabase {conn_test} was established.')
    # Create Conncetion to Output Database to store Testdata with unique_ids
    conn_output_test = connection_preparation.conn_testing()
    logging.info(f'Connection to Output File /temp/id_test_data.db {conn_output_test} was established.')
    dict_testdata = __get_structure(conn_test, conn_output_test)
    return dict_testdata
    
def __get_structure(conn: sqlite3.Connection, conn_output: sqlite3.Connection) -> dict:
    dict_data = dict()
    dict_id_data = dict()
    # Read dataframe from input conn
    dict_data = manage_dfs.get_df(conn)
    logging.info('Read Dataframe from Trainingdata (raw) input.')
    logging.info('Generated unique_ids for each row (aka dataset) in Database and add unique_ids in column "unique_id".')
    # Iterate over each Dataframe in Dictionary
    for key, df_data in dict_data.items():
        # Generate ids for each table (tablename = key and df_ids = one table df)
        df_ids = give_unique_ids.main(path_last_id, df_data)
        manage_dfs.write_df_output(conn_output, key, df_ids)
        dict_id_data[key] = df_ids
    logging.info('Unique_ids were generated and Dataframe was saved in new output file for ids.')
    # close connections
    conn.close()
    conn_output.close()
    return dict_id_data