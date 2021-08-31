# *** Analysis_inside ***
""" Script manages the analysis_inside and organizes the data handling in each step.
    The Data-Management deals with Loading of Input, Passing it to the needed analysis step and Storing/Returning the Output.
    
        * preprocessing
            1. Load data if not already loaded 
            2. Pass data to preprocessing and manage returned data
            3. Store preprocessed data in /temp and return data to next step.
        * pairing
            1. Load data if not already loaded 
            2. Pass data to pairing and manage returned data
            3. Store paired data in /temp and return data to next step.
        * calculation
            1. Load data if not already loaded 
            2. Pass data to calculation and manage returned data
            3. Store calculated data in /temp and return data to next step.
        * evaluation
            1. Load data if not already loaded 
            2. Pass data to evaluation and manage returned data
            3. Store evaluated data in /output. """

# ## Imports
from services import connection_preparation
from services import manage_dfs
from . import calculate_ratios
from . import evaluate_ratios
from analysis import pairing
import preprocessing
import logging
import sys


# ## Functions

# ----- Preprocessing -----
def step_preprocessing(dict_testdata: dict, step_key: str) -> dict:
    """* preprocessing
            1. Load data if not already loaded 
            2. Pass data to preprocessing and manage returned data
            3. Store preprocessed data in /temp and return data to next step.

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
        Dictionary with preprocessed testdata -> keys: table_names, values: DataFrames with job-ads (with new column OneString) """

    # Set Output Dictionary
    dict_testdata_prepro = dict()
    
    # Check if the passed data has unique_ids
    try:
        check_df = [df['unique_id'] for df in dict_testdata.values()]
    except KeyError:
        check_df = False
        
    if not dict_testdata or check_df == False:
        logging.info('Input-Data needs to be loaded first.')
        # Build connection Input
        conn = connection_preparation.conn_testing()
        # Read dataframe
        dict_testdata = manage_dfs.get_df(conn)
    else:
        pass
    # Call the support function to actually do the preprocessing and return the preprocessed data
    dict_testdata_prepro = __prepro_support(dict_testdata, step_key, dict_testdata_prepro)
    return dict_testdata_prepro

def __prepro_support(dict_testdata: dict, step_key: str, dict_testdata_prepro: dict) -> dict:
    # create temp file and make connection
    conn_temp = connection_preparation.create_prepro_connection()
    for name, df_table in dict_testdata.items():
        # preprocess data
        onestring_dict = preprocessing.preprocess_data(df_table, step_key)
        # add results in dataframe
        df_preprocessed = manage_dfs.add_converted_data_to_frame(df_table, onestring_dict, step_key)
        # store df_edited in temp for evaluation
        manage_dfs.write_df_output(conn_temp, name, df_preprocessed)
        dict_testdata_prepro[name] = df_preprocessed
    return dict_testdata_prepro

# ----- Pairing -----
def step_pairing(dict_testdata_prepro: dict) -> dict:
    """* pairing
            1. Load data if not already loaded 
            2. Pass data to pairing and manage returned data
            3. Store paired data in /temp and return data to next step.

    Parameters
    ----------
    dict_testdata_prepro : dict
        Dictionary with preprocessed testdata -> keys: table_names, values: DataFrames with job-ads (with column OneString)
    
    Raises
    ------
    KeyError
        Checks if the column in the dataframe does already exist

    Returns
    -------
    dict_testdata_pairs: dict
        Dictionary with paired testdata -> keys: table_names, values: DataFrames with job-ads pairwise with pairing_labels """

    # Set Output Dictionary
    dict_testdata_pairs = dict()

    # Check if the data has the column OneString
    try:
        check_df = [df['OneString'] for df in dict_testdata_prepro.values()]
    except KeyError:
        check_df = False

    # Check if dict_testdata_prepro is empty
    if not dict_testdata_prepro or check_df == False:
        logging.info('Input-Data needs to be loaded first.')
        # Build connection Input
        conn = connection_preparation.create_prepro_connection()
        # Get the dictionary with table_names and dataframes
        dict_testdata_prepro = manage_dfs.get_df(conn)
    else:
        pass
    # Call the support function to actually do the pairing and return the paired data
    dict_testdata_pairs = __pairing_support(dict_testdata_prepro, dict_testdata_pairs)
    return dict_testdata_pairs

def __pairing_support(dict_testdata_prepro: dict, dict_testdata_pairs: dict) -> dict:
    # Check if passed data has column OneString
    try:
        [df['OneString'] for df in dict_testdata_prepro.values()]
    except KeyError:
        logging.error(f'Preprocessed dataset was not found in data. Repeat Preprocessing step.')
        print(f'Preprocessed dataset was not found, repeat Preprocessing step.')
        sys.exit(1)
    # create temp file and make connection Output
    conn_temp = connection_preparation.create_pair_connection()
    for key, df_table in dict_testdata_prepro.items():
        df_table = pairing.pair_inside(df_table)
        # store df_edited in temp for evaluation
        manage_dfs.write_df_output(conn_temp, key, df_table)
        dict_testdata_pairs[key] = df_table
    conn_temp.close()
    return dict_testdata_pairs

# ----- Calculation -----
def step_analysis_calculate(dict_testdata_pairs: dict, step_key: str, jaccard: bool) -> dict:
    """* calculation
            1. Load data if not already loaded 
            2. Pass data to calculation and manage returned data
            3. Store calculated data in /temp and return data to next step.

    Parameters
    ----------
    dict_testdata_pairs : dict
        Dictionary with paired testdata -> keys: table_names, values: DataFrames with job-ads pairwise with pairing_labels.
    step_key: str
        String with the step_key to define which parts of the program need to be used.
    jaccard: bool
        Boolean from ArgumentParser to decide if inside method Shingling_Similarity the jaccard- or cosine - similarity needs to be calculated.
    
    Raises
    ------
    KeyError
        Checks if the column in the dataframe does already exist

    Returns
    -------
    dict_testdata_calc: dict
        Dictionary with calculated testdata -> keys: table_names, values: DataFrames with job-ads pairwise calculated scores """

    # Set Output Dictionary
    dict_testdata_calc = dict()

    # Check if the data has the column OneString
    try:
        check_df = [df['pairing_label'] for df in dict_testdata_pairs.values()]
    except KeyError:
        check_df = False

    # Check if dict_testdata_pairs is empty
    if not dict_testdata_pairs or check_df == False:
        logging.info('Input-Data needs to be loaded first.')
        # Build conncetion input
        conn = connection_preparation.create_pair_connection()
        # Get the dictionary with table_names and dataframes
        dict_testdata_pairs = manage_dfs.get_df(conn)
    else:
        pass
    # Call the support function to actually do the pairing and return the paired data
    dict_testdata_calc = __calc_support(dict_testdata_pairs, step_key, jaccard, dict_testdata_calc)
    return dict_testdata_calc

def __calc_support(dict_testdata_pairs: dict, step_key: str, jaccard: bool, dict_testdata_calc: dict) -> dict:
    logging.info(f'Caluclation with Method {step_key} was chosen.')
    # Check if passed data has column Pairing
    try:
        [df['pairing_label'] for df in dict_testdata_pairs.values()]
    except KeyError:
        logging.error(f'Paired dataset was not found in data. Repeat pairing step.')
        print(f'Paired dataset was not found, repeat pairing step.')
        sys.exit(1)

    conn_temp = connection_preparation.create_temp_connection()
    for key, df_preprocessed in dict_testdata_pairs.items():
        # calculate distances depending on chosen method from step_key (levenshtein, countvec_sim, tfidf, doc2vec or shingling)
        result_dict = calculate_ratios.calc(df_preprocessed, step_key, jaccard)
        # add results in dataframe
        df_result = manage_dfs.add_converted_data_to_frame(df_preprocessed, result_dict, step_key)
        # store df_edited in temp for evaluation
        manage_dfs.write_df_output(conn_temp, key, df_result)
        dict_testdata_calc[key] = df_result
    return dict_testdata_calc

def step_analysis_evaluate(dict_testdata_calc: dict, step_key: str):
    """* evaluation
            1. Load data if not already loaded 
            2. Pass data to evaluation and manage returned data
            3. Store evaluated data in /output.

    Parameters
    ----------
    dict_testdata_calc: dict
        Dictionary with calculated testdata -> keys: table_names, values: DataFrames with job-ads pairwise calculated scores 
    step_key: str
        String with the step_key to define which parts of the program need to be used
    
    Raises
    ------
    KeyError
        Checks if the column in the dataframe does already exist """

    # Check if column for chosen method already exists in Input data. Because it needs to be evaluated now.
    logging.info(f'Evaluation with method {step_key} was chosen.')
    try:
        check_df = [df[step_key] for df in dict_testdata_calc.values()]
    except KeyError:
        check_df = False

    # Check if Dictionary is empty
    if not dict_testdata_calc or check_df == False:
        # Build connection Input
        conn_temp = connection_preparation.create_temp_connection()
        # Get the dictionary with table_names and dataframes
        dict_testdata_calc = manage_dfs.get_df(conn_temp)
    else:
        pass
    # Evaluate the prior calculated data and store results in output
    __eval_support(dict_testdata_calc, step_key)

def __eval_support(dict_testdata_calc: dict, step_key: str):
    counter = 0
    # Check if passed data has column with the caluclation results to be evaluated
    try:
        [df[step_key] for df in dict_testdata_calc.values()]
    except KeyError:
        logging.error(f'calculation for {step_key} was not found in data. Repeat calculation step.')
        print(f'Calculation for {step_key} is not given, repeat calculation step.')
        sys.exit(1)
    for key, df_preprocessed in dict_testdata_calc.items():
        dict_testdata_eval = evaluate_ratios.starter(df_preprocessed, step_key)
        # Name the Output-table
        if counter == 0:
            key = 'testdata_output'
            counter = 1
        else:
            key = key + '_testdata_output'
        # Create connection to output for final duplicates list (duplicates from train_data)
        conn = connection_preparation.conn_final_output()
        # Write the df in output
        if len(dict_testdata_eval) == 0:
            print("no similarties were found")
        else:
            manage_dfs.write_df_output(conn, key, dict_testdata_eval)