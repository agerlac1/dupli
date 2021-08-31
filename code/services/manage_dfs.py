# ## Management of the SQL-Databases with pd.Datadict_data
""" Script contains functions to handle the data. 
    * SQL-Databases get processed via connections and are stored in Datadict_data. 
        Per table a DataFrame is stored in a Dictionary as value and with tablename as key. 
        IMPORTANT!
            chunksize: needs to be adjusted manually in config
            countermax_per_table: needs to be adjusted manually in config
            filter_tablename: needs to be adjusted manually in config or commented out
    * Furthermore the results from analysis can be mapped with the data from the Datadict_data to store the results. 
    * Output-Datadict_data are written in new created or old SQL-Databases via conncetion-objects """

# ## Imports
import pandas as pd
import sqlite3
import logging
import yaml
from pathlib import Path
import sys

# ## Open Configuration-file and set variables
with open(Path("config.yaml"), "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    chunk_size = (cfg['chunk_size'])
    countermax_per_table = (str(cfg['countermax_per_table']))
    try:
        filter_tablename = (str(cfg['filter_tablename']))
    except KeyError:
        filter_tablename = None

# ## Functions

# Read SQL-Databases and store the tables in DataFrames
def get_df(conn: sqlite3.Connection) -> dict:
    """
    Query all rows in the tasks tables and store them in a Dictionary as values (type: DataFrame)

    Parameters
    ----------
    conn: sqlite3.Connection
        the Connection object

    Returns
    -------
    dict_data: dict
        Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads
    """
    ### try dataframe
    # Tablenames will be extracted
    logging.info(f'Connection was delivered and data will be extracted.')
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    counter = 0
    dict_data = dict()
    for name in res:
        if not(filter_tablename) or filter_tablename not in name[0]:
            df_temp = pd.DataFrame()
            logging.info(f'The table {name[0]} was found and chunksize is set to {chunk_size}.')
            # Select all from data
            df_store = pd.read_sql(("SELECT * FROM {}".format(name[0])), conn, chunksize=chunk_size)
            for chunk in df_store: 
                counter+=chunk_size
                if (int(countermax_per_table) != 0 and counter == int(countermax_per_table)) or len(chunk) < chunk_size:
                    df_temp = df_temp.append(chunk)
                    df_temp = df_temp.reset_index(drop=True)
                    df_temp['index'] = df_temp.index
                    dict_data[name[0]] = df_temp
                    logging.info(f'In table {name[0]}, {counter} datarows were extracted or end of table is reached. Continue with next table.')
                    counter=0
                    break
                else: 
                    df_temp = df_temp.append(chunk)
                    pass
        else: 
            continue
    __check_if_filled(dict_data)
    logging.info('Data is processed and stored in Dictionary and is now returned to next step.')
    return dict_data
    

# function combines the result dictionary with specific values and the depending pariing label with the old df, this will be written in output file in the next step
def add_converted_data_to_frame(df: pd.DataFrame, result_dict: dict, step_key: str) -> pd.DataFrame:
    """ 
    Gets Dataframe with input data and the result_dictionary from analysis_step. Then the results are mapped via label to the data and the edited 
    dataframe is returend. 

    For all steps except preprocessing the pairing_label is used to map, because in the analysis calculation the pairing_dataset is used.
    Therefore some datarows are duplicated, because they have several potential duplicate-partners.
    Means, the unique_ids are not unique inside the analysis calculation, but the pairing_labels are.
    
    Parameters
    ----------
    df: pd.Dataframe
        the input dataframe based on one table from dict_data
    result_dict: dict
        Dictionary with results from analysis step e.g. preprocessed texts or Cosine-Scores. Keys are labels to identify the datarow in dataframe. 
    step_key: str
        Key to decide which step should be executed.

    Returns
    -------
    df_edited: pd.DataFrame
        Dataframe with the edited data. Contains one more column with the mapped informations.
    """
    # Mapping steps
    logging.info(f'Dictionary with with results and data were delivered. Depending on the step {step_key} the data and results will be mapped via ids.')
    df_edited = df.copy()
    if step_key == 'preprocessing':
        df_edited['OneString']= df_edited['unique_id'].map(result_dict)

    elif step_key == 'levenshtein':
        df_edited['levenshtein']= df_edited['pairing_label'].map(result_dict)
       
    elif step_key == 'countvec':
        df_edited['countvec']= df_edited['pairing_label'].map(result_dict)
        
    elif step_key == 'tfidf':
        df_edited['tfidf']= df_edited['pairing_label'].map(result_dict)
        
    elif step_key == 'doc2vec':
        df_edited['doc2vec']= df_edited['pairing_label'].map(result_dict)

    elif step_key == 'shingling':
        df_edited['shingling']= df_edited['pairing_label'].map(result_dict)
    logging.info(f'Result data is done. Dataframe now contains new column with results from step {step_key}')
    return df_edited

# creates table for output (depending on given connection and name of table) and writes the delivered df inside
def write_df_output(conn: sqlite3.Connection, key: str, df: pd.DataFrame) -> None:
    """
    Uses the final dataframe and the given SQL-connection to create or overwrite the output-file (depending on delivered connection 
    object). Key is the tablename for the new table. 

    Parameters
    ----------
    conn: sqlite3.Connection
        Connection to the output_file.
    df: pd.Dataframe
        the final dataframe to be stored in output file.
    key: str
        Key to name the output table.
    """
    logging.info('The Output-File will be created/filled or overwritten/filled.')
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS {tab} (Filler)'.format(tab=key))
    except sqlite3.DatabaseError:
        pass
        
    # Read all from new table (should have only the pseudo columns as generated above)
    data = pd.read_sql('SELECT * FROM {}'.format(key), conn)
    # Replace df with our new adjusted df with ids
    data = df
    # Add the new data to table in database
    data.to_sql(key, conn, if_exists = 'replace', chunksize = 1000, index = False)
       

# ONLY used for training (selects only unique_id and full_text to be faster)
def get_df_train(conn: sqlite3.Connection) -> dict:
    """
    Query all rows in the tasks tables and store them in a Dictionary as values (type: DataFrame)
    Difference to function get_df(conn): only used for training processes, because only unique_id and full_texts are selected. (faster)

    Parameters
    ----------
    conn: sqlite3.Connection
        the Connection object

    Returns
    -------
    dict_data: dict
        Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads (only unique_ids and full_texts)
    """
    ### try dataframe
    # Tablenames will be extracted
    logging.info(f'Connection was delivered and data will be extracted.')
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    counter = 0
    dict_data = dict()
    for name in res:
        if not(filter_tablename) or filter_tablename not in name[0]:
            df_temp = pd.DataFrame()
            logging.info(f'The table {name[0]} was found and chunksize is set to {chunk_size}.')
            # Select only unique_id and full_text from data
            df_store = pd.read_sql(("SELECT unique_id, full_text FROM {}".format(name[0])), conn, chunksize=chunk_size)
            for chunk in df_store: 
                counter+=chunk_size
                if (int(countermax_per_table) != 0 and counter == int(countermax_per_table)) or len(chunk) < chunk_size:
                    df_temp = df_temp.append(chunk)
                    df_temp = df_temp.reset_index(drop=True)
                    df_temp['index'] = df_temp.index
                    dict_data[name[0]] = df_temp
                    logging.info(f'In table {name[0]} {counter} datarows were extracted or end of table is reached. Continue with next table.')
                    counter=0
                    break
                else: 
                    df_temp = df_temp.append(chunk)
                    pass
        else:
            continue
    __check_if_filled(dict_data)
    return dict_data

def __check_if_filled(data):
    if not data or data == '':
        logging.error('Data could not be loaded. Check paths or databases.')
        print('Data Loading failed. Check paths and databases.')
        sys.exit(1)
    else:
        return data