""" Manages Pairing for analysis_inside and analysis_outside. 
    Distributes tasks for both strategies:
    * Filter and pair the potential duplicates
    * Give pairing_labels to the pairs from filtering
    * annotate_results (only for inside and masterthesis relevant) """

# ## Imports
from . import pairing
import pandas as pd
import logging
import sys

# ## Define Variables
message = None

# ## Functions

# Pairing for analysis_inside
def pair_inside(df: pd.DataFrame) -> pd.DataFrame:
    """ Coordinates the three steps for pairing in analysis_inside:
        1. Filtering with metadata
        2. Label the potential pairs from step 1
        3. Annotate results (ONLY in masterthesis needed else comment function out.
    
    Parameters
    ----------
    df: pd.DataFrame
        One table from dictionary with the whole dataset. Contains metadata and full_texts.

    Returns
    -------
    df_pairs: pd.DataFrame
        Paired dataset stored in a DataFrame."""

    # Step 1: 
    # Give DataFrame to metadata-filter and get the DataFrame with potential pairs back. 
    df_pairs = pairing.filter_inside(df)

    # Step 2: 
    # Label the new dataframe with new pairinglabels (pattern: 1_a, 1_b; 2_a, 2_b; etc...)
    df_pairs = pairing.pair_labeling(df_pairs)
    
    # Step 3:
    # Annotate the known real duplicates in the labeled Dataframe for evaluation. (ONLY needed in masterthesis and with known duplicates)
    # Won't be called if annotations are missing or can be commented out.
    df_pairs = pairing.annotate_results(df_pairs)
    return df_pairs


# Pairing for analysis_outside
def pair_outside(sims_dict: dict, mysimslist_test: list, mysimslist_train: list, dict_testdata: dict, dict_traindata: dict) -> pd.DataFrame:
    """ Coordinates the two steps for pairing in analysis_inside and the preparations beforehand:
        0. Step: Preparations
        1. Filtering with metadata
        2. Label the potential pairs from step 1
        
    Parameters
    ----------
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads
    sims_dict: dict
        Dictionary with most similar job-ads -> keys: dict_testdata unique_ids, values: most_similar unique_ids from dict_traindata depending on testdata
    mysimslist_test: list
        List with only the unique_ids for testdata from sims_dict (aka the keys)
    mysimslsit_train: list
        List with only the unique_ids for traindata from sims_dict (aka the values)
    
    Raises
    ------
    AttributeError
        If wrong traindata is used with different unique_ids compared to unique_ids saved in model (doc2vec), an empty df is used -> Problematic!

    Returns
    -------
    df_pairs: pd.DataFrame
        Paired dataset stored in a DataFrame."""
    
    # Define Variables
    global message
    df_testdata = pd.DataFrame()
    df_traindata = pd.DataFrame()

    df_pairs = pd.DataFrame()
    memory_list = list()

    # Step Preparations:
    # Load the testdata needed for comparison in a DataFrame to save MemoryStorage
    for key, df in dict_testdata.items(): 
        for ind in df.index:
            if df.unique_id[ind] in mysimslist_test:
                df_testdata = df_testdata.append(df.iloc[ind], ignore_index=True)
            else:
                continue
    # Load the traindata needed for comparison in a DataFrame to save MemoryStorage
    for key, df in dict_traindata.items():
        for ind in df.index:
            if df.unique_id[ind] in mysimslist_train:
                df_traindata = df_traindata.append(df.iloc[ind], ignore_index=True)
            else:
                continue

    # Step 1: 
    # Give DataFrame to metadata-filter and get the DataFrame with potential pairs back. 
    try:
        for test_id, train_ids in sims_dict.items():
            # First Datapiece to compare is always from Testdata
            datapiece_1 = df_testdata[df_testdata.unique_id == test_id]
            for train_id in train_ids:
                # The second Datapiece is one of the related Traindatapieces.
                datapiece_2 = df_traindata[df_traindata.unique_id == train_id]
                if datapiece_2.empty == False:
                    df_pairs = pairing.filter_outside(datapiece_1, datapiece_2, df_pairs, memory_list)
                else:
                    if message is None:
                        logging.warning(f'Some of the datapieces do not match with the Trainingdata, check if all Datapieces from Trainingdata were a part of the Trainingdata for the model!')
                        message = "delivered"
                    continue
    except AttributeError:
        logging.error(f'After checking and extracting the Trainingdata and Testdata an error occurred. The df_traindata (and/or df_testdata) is empty, probably because the input Trainingdata (from analysis_outside) is different from the Trainingdata used for modeling. This means that the unique_ids in the trained model do not match with the here used Trainingdataids. Check again paths and model in config.yaml.')
        print('Error occurred. Wrong model used (with different unique_ids than unique_ids in chosen trainingdata). Check logging-file for further information.')
        sys.exit(1)
        raise
    
    # Step 2: 
    # Label the new dataframe with new pairinglabels (pattern: 1_a, 1_b; 2_a, 2_b; etc...)
    df_pairs = pairing.pair_labeling(df_pairs)
    return df_pairs