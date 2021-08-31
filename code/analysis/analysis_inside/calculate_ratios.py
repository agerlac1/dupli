# *** Calculator ***
"""Script manages the data handling to get pairwise strings and pass them to chosen method. 
In Return the score per pair is delivered and noted in result_dict."""

# ## Imports
from . import formulas
import pandas as pd
import logging

def calc(df: pd.DataFrame, step_key: str, jaccard: bool) -> dict:
    """Function iterates over dataframe and extracts each pair to be analyzed.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe contains one table from the input Dictionary to be calculated.
    step_key: str
        String with the step_key to define which parts of the program need to be used.
    jaccard: bool
        Boolean from ArgumentParser to decide if inside method Shingling_Similarity the jaccard- or cosine - similarity needs to be calculated.

    Returns
    -------
    result: dict
        Dictionary with calculated testdata -> keys: pairing_labels & values: scores """

    # Set Output Dictionary
    result_dict = dict()
    # Iterate over DataFrame two rows at a time via index
    for (indx1),(indx2) in zip(df[:-1].index,df[1:].index):
        # Select pairs (a & b)
        if df.loc[indx1].pairing_label.endswith('a') and df.loc[indx2].pairing_label.endswith('b'):
            # save the label and the fulltext depending on pairing (a or b)
            patterna = df.loc[indx1].pairing_label
            patternb = df.loc[indx2].pairing_label
            try:
                # Select preprocessed full_texts 
                input_full_a = df.loc[indx1].OneString
                input_full_b = df.loc[indx2].OneString
            except:
                logging.info('For a pair in Calculation, no full_texts could be extracted. Scores will remain empty.')
                result_dict.update({patterna : ''})     
                result_dict.update({patternb : ''})
                pass
            # CALCULATOR
            cosine = formulas.distributor(step_key, input_full_a, input_full_b, jaccard)
            # update dict with cosine for each label
            result_dict.update({patterna : cosine})     
            result_dict.update({patternb : cosine})
        else:
            continue
    return result_dict