# *** Analysis ***
""" Script manages the two different analysis steps:
        1. analysis_inside (searches for duplicates in one dataset)
            * preprocessing
            * pairing
            * calculation
            * evaluation
             --> possible to call all steps at once or to repeat several steps separately
        2. analysis_outside (searches for duplicates for the testdata in the traindata)
            * doc2vec
            * tfidf    """

# ## Imports
from analysis import analysis_inside
from analysis import analysis_outside
import logging

# ## Define Variables
dict_testdata = dict()
dict_traindata = dict()

# ## Functions
# ----- Analysis_inside -----
def analysis_in(args: dict, dict_testdata :dict, dict_traindata: dict) -> None:
    """ analysis_inside (searches for duplicates in one dataset)
            * preprocessing
            * pairing
            * calculation
            * evaluation

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads """ 
    
    # Get values from ArgumentParser
    method_args = vars(args)
    # Depending on Arguments, call parts of the analysis_inside
    logging.info("Analysis_inside started.")
    # Preprocessing
    if method_args["preprocessing"]:
        step_key = 'preprocessing'
        logging.info("Preprocessing for the data starts.")
        dict_testdata = analysis_inside.step_preprocessing(dict_testdata, step_key)
        logging.info("Preprocessing finished.")
    # Pairing
    if method_args['pairing']:
        logging.info('Pairing starts.')
        dict_testdata = analysis_inside.step_pairing(dict_testdata)
        logging.info('Pairing finished.')
    # Calculation
    if method_args['calculation']:
        logging.info('Calculation starts.')
        dict_testdata = analysis_inside.step_analysis_calculate(dict_testdata, method_args['method_in'], method_args['jaccard'])
        logging.info('Calculation finished.')
    # Evaluation
    if method_args['evaluation']:
        logging.info('Evaluation starts.')
        analysis_inside.step_analysis_evaluate(dict_testdata, method_args['method_in'])
        logging.info('Evaluation finished.')
    if method_args['preprocessing'] == False and method_args['pairing'] == False and method_args['calculation'] == False and method_args['evaluation'] == False:
        # Preprocessing
        step_key = 'preprocessing'
        logging.info("Preprocessing for the data starts.")
        dict_testdata = analysis_inside.step_preprocessing(dict_testdata, step_key)
        logging.info("Preprocessing finished.")
        # Pairing
        logging.info('Pairing starts.')
        dict_testdata = analysis_inside.step_pairing(dict_testdata)
        logging.info('Pairing finished.')
        # Calculation
        logging.info('Calculation starts.')
        dict_testdata = analysis_inside.step_analysis_calculate(dict_testdata, method_args['method_in'], method_args['jaccard'])
        logging.info('Calculation finished.')
        # Evaluation
        logging.info('Evaluation starts.')
        analysis_inside.step_analysis_evaluate(dict_testdata, method_args['method_in'])
        logging.info('Evaluation finished.')

# ----- Analysis_outside -----
def analysis_out(args: dict, dict_testdata: dict, dict_traindata: dict) -> None:
    """ analysis_outside (searches for duplicates for the testdata in the traindata).
        Options:
            * doc2vec
            * tfidf

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads """

    # Get values from ArgumentParser
    method_args = vars(args)
    # Depending on Arguments, call parts of the analysis_outside
    logging.info('Analysis_outside started.')
    # Doc2Vec
    if method_args['method_out'] == 'doc2vec':
        logging.info('Method Doc2Vec starts.')
        analysis_outside.doc2vec_out(args, dict_testdata, dict_traindata)
        logging.info('Method Doc2Vec finished.')
    # TF-IDF
    elif method_args['method_out'] == 'tfidf':
        logging.info('Method TF-IDF starts.')
        analysis_outside.tfidf_out(args, dict_testdata, dict_traindata)
        logging.info('Method TF-IDF finished.')
    else:
        logging.info('Method Doc2Vec starts.')
        analysis_outside.doc2vec_out(args, dict_testdata, dict_traindata)
        logging.info('Method Doc2Vec finished.')