# ## Manage unique_ids
# ## Imports
import math
from io import StringIO
from os.path import isfile, join
from xml.parsers import expat
import sys
import fileinput
import os
import logging
import pandas as pd
from pathlib import Path

# ## Define Variables
path_last_id = None

# ## Functions

def __get_current_hex(path_last_id: Path) -> str:
    file = open((path_last_id), "r")
    currentHex = file.read()
    file.close()
    return currentHex

def __write_current_hex(path_last_id: Path, currentHex: str):
    file = open((path_last_id), "w")
    file.write(currentHex)
    file.close()

def __generate_id(path_last_id: Path) -> str:
    # read last used hex number
    currentHex = __get_current_hex(path_last_id)

    # count hex number
    counter = int(currentHex, 16) + 1
    new_hex = hex(counter)
    currentHex = new_hex[2:].upper()

    # write current hex number in file
    __write_current_hex(path_last_id, currentHex)

    newUID = ""
    for _ in range(16 - len(currentHex)):
        newUID = newUID + "0"
    newUID = newUID + currentHex
    newUID = newUID[:4] + '-' + newUID[4:8] + '-' + newUID[8:12] + '-' + newUID[12::] 
    return newUID


def main(path_last_id: Path, df_ids: pd.DataFrame) -> pd.DataFrame:
    ''' Function to iterate over each row in passed Dataframe. Generate and give unique_ids to each row and store the new id in the column "unique_id" in DataFrame.

    Parameters
    ----------
    path_last_id: Path
        Path to the file with the stored last_unique_id
    df_ids: pd.DataFrame
        Dataframe from the Dictionary to iterate over and give unique_ids to each row
    
    Returns
    -------
    df_ids: pd.DataFrame
        Dataframe with the old data and the new unique_ids in column "unique_id"
    '''
    # Prepare column 'unique_ids' in Output
    df_ids = df_ids.drop(['unique_id'], axis=1, errors='ignore')
    # Iterate over each row in Dataframe and give unique_ids
    for row in df_ids.itertuples():
        unique_id = __generate_id(path_last_id)
        df_ids.at[row.index, 'unique_id'] = unique_id
    logging.info(f'New last unique_id is: {unique_id}.')
    return df_ids