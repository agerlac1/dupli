# ## Preprocessing
""" Script contains two functions to preprocess the full_texts. 
    1. to preprocess the data (stopwords, tokenization whitespace, lowercase)
        Returns: List with TaggedDocument objects (tokens_list, unique_id)
    2. to preprocess the data (stopwords, tokenization, lowercase, punctuation removal)
        Returns: Dictionary with keys = unique_ids and values = OneStrings """
        
# ## Imports
from nltk.corpus import stopwords 
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import pandas as pd
import logging

# ## Functions
def preprocess_labeled(df: pd.DataFrame) -> list:
    """ Function manages the preprocessing for doc2vec related actions. 
    Doc2Vec needs as return a preprocessed list of TaggedDocument-Objects.
    Preprocessing-steps: 
        * tokenization 
        * stopwords removal
        * simple_preprocess: lowercase and punctuation removal
        * Tagging

    Parameters
    ----------
    df: pd.DataFrame
        One table from dictionary with the whole dataset. Contains only unique_ids and full_texts.

    Raises
    ------
    MemoryError
        Tokens in the function simple_preprocess can be too long, depending on MemoryStorage. If that happens, program continues with next string.

    Returns
    -------
    data: list
        Data contains of TaggedDocument-objects(tokens_list, unique_id) """
        
    data = list()
    # Iterate over each datarow in DataFrame
    for row in df.itertuples(): 
        # set lists for tokenization and stopwords removal
        tokens_list = list()
        stop_words = set(stopwords.words('german'))  
        try:
            tokens_list = __exe_prepro(tokens_list, stop_words, row)
            data.append(TaggedDocument(tokens_list,[row.unique_id]))
        except MemoryError:
            logging.warning(f'Token from datarow {row.unique_id} is too long. Program continues with next datarow.')
            continue
    return data

def preprocess_strings(df: pd.DataFrame) -> dict:
    """ Function manages the preprocessing for simple preprocessing of texts. 
    Preprocessing-steps: 
        * tokenization 
        * stopwords removal
        * simple_preprocess: lowercase and punctuation removal

    Parameters
    ----------
    df: pd.DataFrame
        One table from dictionary with the whole dataset. Contains all metadata and full_texts.

    Raises
    ------
    MemoryError
        Tokens in the function simple_preprocess can be too long, depending on MemoryStorage. If that happens, program continues with next string.

    Returns
    -------
    data: dict
        Data is a Dictionary with keys = unique_ids and values = OneStrings.
    """
    preprocessed_onestrings = list()
    unqiue_id_list = list()
    # Iterate over each datarow in DataFrame
    for row in df.itertuples(): 
        # set lists for tokenization and stopwords removal
        tokens_list = list()
        stop_words = set(stopwords.words('german'))
        try:
            tokens_list = __exe_prepro(tokens_list, stop_words, row)
            # make onestring and no list
            preprocessed_onestrings.append(' '.join(tokens_list))
            unqiue_id_list.append(row.unique_id)
        except MemoryError:
            logging.warning(f'Token from datarow {row.unique_id} is too long. Program continues with next datarow.')
            continue
    data = dict(zip(unqiue_id_list, preprocessed_onestrings))
    return data

def __exe_prepro(tokens_list, stop_words, row):
    # start simple_preprocess from genism for texts
    tokens = simple_preprocess(row.full_text)
    # remove stopwords
    for token in tokens:
        if token not in stop_words:
            tokens_list.append(token)
        else:
            continue
    # remove empty elements
    tokens_list[:] = [x for x in tokens_list if x]
    return tokens_list