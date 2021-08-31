# Script gets a dataframe with id handles. 
""" Rotates over data and pairs datapieces with each other where Metadata suggests the similarity.
A Datapiece will not be paired with itself and not twice with another row like a&b OR b&a
Filter can be manually adjusted in config.yaml.
FILTERS:
    * Date Ranges in a +-60 days frame AND
        * Full_texts are equal or
        * LocationNames are equal or 
        * ProfessionIscoCode is equal and advertisername is equal or part of eachother 
Later pairing_labels are given to each pair coming from the filter. """

# ## Imports
import pandas as pd
import numpy as np
import datetime
import os
import yaml
from pathlib import Path
import logging

# ## Define Variables
known_dupls = list()
counter = None
date_past = int
date_future = int
full_param = bool
location_param = bool
profisco_advname_param = bool

# ## Class Filter_Params
class Filter_Params:
    """ Checks and sets all Metadata-Filter values. If values in config.yaml are invalid, the class corrects them to a default setting."""

    def __init__(self):
        # ## Set Variables
        with open(Path("config.yaml"), "r") as yamlfile:
            cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            metdata_params = cfg['metadata_filter']
            date_params = metdata_params['date']
            date_past = date_params['past']
            date_future = date_params['future']
            full_param = metdata_params['full_text']
            location_param = metdata_params['location_name']
            profisco_advname_param = metdata_params['profisco_advname']
            solution_annotated = Path(cfg['solution_annotated'])
        logging.info(f'Before checking, the metadatafilter is set to: date_past = {date_past}, date_future = {date_future},full_param = {full_param}, location_param = {location_param}, profisco_advname_param = {profisco_advname_param}')
        self.date_past = date_past
        self.date_future = date_future
        self.full_param = full_param
        self.location_param = location_param
        self.profisco_advname_param = profisco_advname_param
        self.solution_annotated = solution_annotated

    # getter method for parameter
    def get_date_past(self): 
        return self.date_past
    def get_date_future(self): 
        return self.date_future
    def get_full_param(self): 
        return self.full_param
    def get_location_param(self): 
        return self.location_param
    def get_profisco_advname_param(self): 
        return self.profisco_advname_param
    def get_solution_annotated(self):
        return self.solution_annotated

    # setter method for parameter
    def set_date_past(self):
        date_past = self.date_past
        if type(date_past) == int:
            self.date_past = date_past
        else:
            self.date_past = 60
    def set_date_future(self):
        date_future = self.date_future
        if type(date_future) == int:
            self.date_future = date_future
        else:
            self.date_future = 60
    def set_full_param(self):
        full_param = self.full_param
        if type(full_param) == bool:
            self.full_param == full_param
        else:
            self.full_param = True
    def set_location_param(self):
        location_param = self.location_param
        if type(location_param) == bool:
            self.location_param = location_param
        else: 
            self.location_param = True
    def set_profisco_advname_param(self):
        profisco_advname_param = self.profisco_advname_param
        if type(profisco_advname_param) == bool:
            self.profisco_advname_param = profisco_advname_param
        else:
            self.profisco_advname_param = True
    def set_solution_annotated(self):
        self.solution_annotated = self.solution_annotated

# Checks, that the metadata-filter is only set once and initiates the getter and setter of the class.
def __check_filter():
    global counter
    # Set Parameter for global scope
    global date_past
    global date_future
    global full_param
    global location_param 
    global profisco_advname_param

    # Check once
    if counter == None:
        # Check if filter_settings are valid, else set defaults
        # Initiate Object
        filter_obj = Filter_Params()
        # Setter
        filter_obj.set_date_past()
        filter_obj.set_date_future()
        filter_obj.set_location_param()
        filter_obj.set_full_param()
        filter_obj.set_profisco_advname_param()
        # Getter
        date_past = filter_obj.get_date_past()
        date_future = filter_obj.get_date_future()
        full_param = filter_obj.get_full_param()
        profisco_advname_param = filter_obj.get_profisco_advname_param()
        location_param = filter_obj.get_location_param()
        logging.info(f'After checking, the metadatafilter is set to: date_past = {date_past}, date_future = {date_future},full_param = {full_param}, location_param = {location_param}, profisco_advname_param = {profisco_advname_param}')
        counter = 'filled'
        return date_past, date_future, full_param, location_param, profisco_advname_param
    else:
        return date_past, date_future, full_param, location_param, profisco_advname_param
    

# ## Functions

# *** Filtering for analysis_inside ***
def filter_inside(df_current: pd.DataFrame) -> pd.DataFrame:
    """ Uses the Metadata given in DataFrame to compare the job-ads in the data.
    Because pairs are searched inside one DataFrame, the 'groupby' operation (pandas) is useful and saves memory.
        * Date Ranges in a +-60 days frame AND
            * Full_texts are equal or
            * LocationNames are equal or 
            * ProfessionIscoCode is equal and advertisername is equal or part of eachother

    Parameters
    ----------
    df_current: pd.DataFrame
        One table from dictionary with the whole dataset. Contains metadata and full_texts.
    
    Raises
    ------
    KeyError, AttributeError, UnboundLocalError
        If errors within the dataframe occur and one filter might not work.

    Returns
    -------
    df_pairs: pd.DataFrame
        Paired dataset stored in a DataFrame.
    """

    # Checks and Sets the metadata-filter values once.
    date_past, date_future, full_param, location_param, profisco_advname_param = __check_filter()

    # Initiate Backup to check if pair is already filtered.
    memory_list = list()
    # Set Output DatFrame
    df_pairs = pd.DataFrame()


    for ind1 in df_current.index:
        memory_list.append((df_current.unique_id[ind1], df_current.unique_id[ind1]))
        
        # FILTER DATE FOR FILTER 1 FILTER 2 and FILTER 3
        try:
            current = pd.to_datetime(df_current.date[ind1])
            past = str(current - pd.DateOffset(days=date_past))
            future = str(current + pd.DateOffset(days=date_future))
            df_date= df_current.loc[df_current['date'].between(past, future, inclusive=True)]
        except (KeyError, AttributeError, UnboundLocalError) as err:
            logging.warning(f'The metadata-field(s) "date" could not be used. Continue without it. Error message:{err}')
            df_date = df_current
            pass

        # FILTER 1: FULLTEXT
        if full_param == True:
            try:
                testdf = df_date.groupby('full_text').get_group(str(df_current.full_text[ind1]))
                for ind in testdf.index:
                    memory_list, df_pairs = __searcher_inside(testdf, ind, df_current, ind1, memory_list, df_pairs)
            except (KeyError, AttributeError, UnboundLocalError) as err:
                logging.warning(f'The metadata-field(s) full_text could not be used. Continue without it. Error message:{err}')
                pass
        else:
            pass

        # FILTER 2: LOCATIONNAME
        if location_param == True:
            try:
                testdf = df_date.groupby('location_name').get_group(str(df_current.location_name[ind1]))
                for ind in testdf.index:
                    memory_list, df_pairs = __searcher_inside(testdf, ind, df_current, ind1, memory_list, df_pairs)
            except (KeyError, AttributeError, UnboundLocalError) as err:
                logging.warning(f'The metadata-field(s) location_name could not be used. Continue without it. Error message:{err}')
                pass
        else:
            pass

        # FILTER 3: PROFESSION_ISCO_CODE AND ADVERTISER_NAME
        if profisco_advname_param == True:
            try:
                testdf = df_date.groupby('profession_isco_code').get_group(str(df_current.profession_isco_code[ind1])) 
                testdf = testdf.loc[testdf['advertiser_name'].str.contains(df_current.advertiser_name[ind1], na=False, regex=False)]
                for ind in testdf.index:
                    memory_list, df_pairs = __searcher_inside(testdf, ind, df_current, ind1, memory_list, df_pairs)
            except (KeyError, AttributeError, UnboundLocalError) as err:
                logging.warning(f'The metadata-field(s) profession_isco_code and advertiser_name could not be used. Continue without it. Error message:{err}')
                pass      
        else:
            pass
        try:
            if full_param != True and location_param != True and profisco_advname_param != True:
                for ind in df_date.index:
                    memory_list, df_pairs = __searcher_inside(df_date, ind, df_current, ind1, memory_list, df_pairs)
        except (KeyError, AttributeError, UnboundLocalError) as err:
                logging.warning(f'The metadata-field(s) location_name could not be used. Continue without it.Error message:{err}')
                pass

    return df_pairs

# *** Filtering for analysis_outside ***
def filter_outside(datapiece_1: pd.DataFrame, datapiece_2: pd.DataFrame, df_pairs: pd.DataFrame, memory_list: list) -> pd.DataFrame:
    """ Uses the Metadata given in datapiece_1 and datapiece_2 to compare the job-ads.
    Because pair-pieces are directly compared the 'groupby' operation (pandas) cannot be used.
        * Date Ranges in a +-60 days frame AND
            * Full_texts are equal or
            * LocationNames are equal or 
            * ProfessionIscoCode is equal and advertisername is equal or part of eachother

    Parameters
    ----------
    datapiece_1: pd.DataFrame 
        Contains one row with the datapiece to be compared (from testdata).
    datapiece_2: pd.DataFrame
        Contains one row with the datapiece to be compared (from traindata).
    df_pairs: pd.DataFrame 
        Paired datapieces stored in a DataFrame (backup).
    memory_list: list
        List of all datapieces already compared (unique_ids).

    Raises
    ------
    KeyError, AttributeError, UnboundLocalError
        If errors within the dataframe occur and one filter might not work.

    Returns
    -------
    df_pairs: pd.DataFrame
        Paired dataset stored in a DataFrame. """


    # Checks and Sets the metadata-filter values once.
    date_past, date_future, full_param, location_param, profisco_advname_param = __check_filter()
    
    try:
        # setup for the date checkup: look if the date of datapiece_2 lies within the frame +-60 days of the datapiece_1
        dt2 = str(pd.to_datetime(datapiece_1.date.iloc[0]) + pd.DateOffset(days=date_past))
        dt1 = str(pd.to_datetime(datapiece_1.date.iloc[0]) - pd.DateOffset(days=date_future))
        date_frame = datapiece_2['date'].between(dt1, dt2, inclusive=True).any()
    except (KeyError, UnboundLocalError, AttributeError) as err:
        logging.warning(f'The metadata-field(s) "date" could not be used. Continue without it. Error message:{err}')
        date_frame = True
        pass

    # FILTER DATERANGE FOR FILTER 1 FILTER 2 and FILTER 3
    if datapiece_1.unique_id.iloc[0] != datapiece_2.unique_id.iloc[0] and date_frame:
        # if conditions are no problem, because of the memory_list. No pair will be stored as potential duplicates twice.
        try:
            # 1. Filter: Fulltext equals fulltext 
            if full_param == True and (datapiece_2.full_text.iloc[0] == datapiece_1.full_text.iloc[0]):
                    memory_list, df_pairs = __searcher_outside(datapiece_1.iloc[0], datapiece_2.iloc[0], memory_list, df_pairs)
        except (KeyError, UnboundLocalError, AttributeError) as err:
            logging.warning(f'The metadata-field(s) "full_text" could not be used. Continue without it. Error message:{err}')
            pass
        try:
            # 2. Filter: LocationName is equal 
            if location_param == True and ((datapiece_1.location_name.iloc[0] == datapiece_2.location_name.iloc[0])):
                memory_list, df_pairs = __searcher_outside(datapiece_1.iloc[0], datapiece_2.iloc[0], memory_list, df_pairs)
        except (KeyError, UnboundLocalError, AttributeError) as err:
            logging.warning(f'The metadata-field(s) "location_name" could not be used. Continue without it. Error message:{err}')
            pass
        try:
            # 3. Filter: ProfessionIscoCode is equal and advertisername is equal or part of eachother
            # First check if the fields are not empty
            if profisco_advname_param == True and  datapiece_2.advertiser_name.iloc[0] !='' and datapiece_1.advertiser_name.iloc[0] != '' and datapiece_2.advertiser_name.iloc[0] is not None and datapiece_1.advertiser_name.iloc[0] is not None and datapiece_2.profession_isco_code.iloc[0] == datapiece_1.profession_isco_code.iloc[0]:
                if (datapiece_2.advertiser_name.iloc[0] == datapiece_1.advertiser_name.iloc[0] or datapiece_2.advertiser_name.iloc[0] in datapiece_1.advertiser_name.iloc[0] or datapiece_1.advertiser_name.iloc[0] in datapiece_2.advertiser_name.iloc[0]):
                    memory_list, df_pairs = __searcher_outside(datapiece_1.iloc[0], datapiece_2.iloc[0], memory_list, df_pairs)
        except (KeyError, UnboundLocalError, AttributeError) as err:
            logging.warning(f'The metadata-field(s) "profession_isco_code" and "advertiser_name" could not be used. Continue without it. Error message:{err}')
            pass
        try:
            # No Filter, only date
            if full_param != True and location_param != True and profisco_advname_param != True:
                memory_list, df_pairs = __searcher_outside(datapiece_1.iloc[0], datapiece_2.iloc[0], memory_list, df_pairs)
        except (KeyError, UnboundLocalError, AttributeError) as err:
            logging.warning(f'The metadata filter (only date) could not be used. Continue without it. Error message:{err}')
            pass
    return df_pairs

def __searcher_inside(testdf, ind, df_current, ind1, memory_list, df_pairs):
    backup = (testdf.unique_id[ind], df_current.unique_id[ind1])
    backup2 = (df_current.unique_id[ind1], testdf.unique_id[ind])
    # check if one of them is already in the new dataframe:
    if not(memory_list.__contains__(backup)) and not(memory_list.__contains__(backup2)): 
        # Add new pairs to final DataFrame (df_pairs)
        df_pairs = df_pairs.append(df_current.iloc[ind1], ignore_index=True)
        df_pairs = df_pairs.append(df_current[df_current.unique_id == testdf.unique_id[ind]], ignore_index=True)
        memory_list.append(backup)
        memory_list.append(backup2)
    return memory_list, df_pairs

def __searcher_outside(datapiece_1, datapiece_2, memory_list, df_pairs):
    backup = (datapiece_2.unique_id, datapiece_1.unique_id)
    backup2 = (datapiece_1.unique_id, datapiece_2.unique_id)
    # check if one of them is already in the new dataframe:
    if not(memory_list.__contains__(backup)) and not(memory_list.__contains__(backup2)): 
        # Add new pairs to final DataFrame (df_pairs)
        df_pairs = df_pairs.append(datapiece_1, ignore_index=True)
        df_pairs = df_pairs.append(datapiece_2, ignore_index=True)
        memory_list.append(backup)
        memory_list.append(backup2)
    return memory_list, df_pairs

def pair_labeling(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """ Iterate over the final paired DataFrame from Step 1 (Filtering).
    Add pairing_label to all pairs (e.g. 1_a, 1_b, 2_a, 2_b).
    Important for the analysis to identify the pairs of potential duplicates. 
    Set the whole duplicate column to 0 as preparation for the next step (only needed for masterthesis).

    Parameters
    ----------
    df_pairs: pd.DataFrame
        Paired dataset stored in a DataFrame.

    Returns
    -------
    df_pairs: pd.DataFrame
        Paired dataset stored in a DataFrame with pairing_labels.
    """
    t = 1                  
    skipper = 0
    for (indx1),(indx2) in zip(df_pairs[:-1].index,df_pairs[1:].index):
        if skipper == 0:
            labela = str(t) + '_a'
            labelb = str(t) + '_b'
            df_pairs.at[indx1, 'pairing_label'] = labela
            df_pairs.at[indx2, 'pairing_label'] = labelb
            t += 1
            skipper = 1
        else:
            skipper = 0
            continue
    return df_pairs

def annotate_results(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """ ONLY needed for masterthesis --> comment function out while using new data or 
    it will skip automatically if testset_id label is missing (label was given manually beforehand for thesis)
    Opens file with known duplicates (solution_annotated.txt) and extracts labels (stores in list).
    Notes "1" in column "duplicats" in df_pairs for each known duplicate-pair form solution_annotated.txt.

    Parameters
    ----------
    df_pairs: pd.DataFrame
        Paired dataset stored in a DataFrame.

    Returns
    -------
    df_pairs: pd.DataFrame
        Paired dataset stored in a DataFrame with column duplicates and "1" for annotated duplicates.
    """
    # Check if filter_settings are valid, else set defaults
    filter_obj = Filter_Params()
    filter_obj.set_solution_annotated()
    solution_annotated = filter_obj.get_solution_annotated()

    try:
        # open file with known duplicates in testset, extract the labels and add them to the list
        with open(solution_annotated, 'r') as f:
            known_dupls = [line.strip('\n') for line in f]
        # add column 'duplicate' to DataFrame and set all values to '0'
        df_pairs['duplicate'] = 0
        # rotate over two rows at the same time and skip one with next rotation 
        for (indx1),(indx2) in zip(df_pairs[:-1].index,df_pairs[1:].index):
            # look for the labels ending with "a"
            if df_pairs.pairing_label.iloc[indx1].endswith('a'):
                for pairs in known_dupls:
                    item = pairs.split(',')
                    # check if this one and the following row are labels noted in the known_dupls list. If true: mark them with a 1 in the column duplicate
                    if (item[0] == df_pairs.testset_id.iloc[indx1] and item[1] == df_pairs.testset_id.iloc[indx2]) or (item[1] == df_pairs.testset_id.iloc[indx1] and item[0] == df_pairs.testset_id.iloc[indx2]): 
                        df_pairs.at[indx1, 'duplicate'] = 1
                        df_pairs.at[indx2, 'duplicate'] = 1
                        continue
                    else:
                        continue
            else:
                continue
    except:
        logging.warning("function annotate_results was not successful. Depending on use of program check, if you really have a not anntotated corpus or if the annotations are missing.")
        pass
    return df_pairs   