# ## MAIN-SCRIPT
""" The main script of the program. Further information for execution can be found in the file readme.md.

PURPOSE: Find jobs-ads duplicates in given database and in already known database depending on the given database. 

INPUT:
        - SQL-Databases containing job-ads (One as Trainingdata and one as Testdata)
            --> Metadata & Full_texts
OUTPUT:
        - SQL-Database containing pairs of job-ads, identified as duplicates
STRUCTURE:
        Program consists of three major parts:
                a. id_handling
                b. modeling
                c. analysis
TASKS: 
        a. id_handling: Gives unique_ids to the input data (Trainingdata and Testdata) 
        b. modeling: Trains models with the Trainingdata (with unique_ids)
        c. analysis (with unique_ids): 
            - Analyzes the Testdata to find duplicates inside the Testdata (inside-analysis)
            - Analyzes the Testdata to find duplicates for the Testdata in the Trainingdata (outside-analysis)
PATHS   --> All paths can be adjusted in the config.yaml (Important to set path for input files!).

WORKING-DIR: /code

CALLS:
    (ArgumentParser to process CLI-Commands)

ID_HANDLING:
    --> gives unique_ids to input-data (Trainingdata and Testdata) and stores them in /temp/id_test_data.db and /temp/id_train_data.db

            python main.py id_handling 
                optional: --test or --train
                default: executes both

MODELING (depends on id_handling first):
    --> trains, retrains and checks sanity of models --> models saved in folder dupl/models (one level higher than working_dir)

            python main.py modeling
                optional: --modeling_type tfidf or --modeling_type doc2vec
                default: --modeling_type doc2vec
                specification: --training and/or --retraining and/or --sanity_check
                    --> retraining only for doc2vec

            e.g. if you want to train a doc2vec model and do a sanity_check afterwards (and specifically no retraining),  call:
                    python main.py modeling --modeling_type doc2vec --training --sanity_check 
                if you want to train a tfidf model and do a sanity_check (all three do the same):
                    python main.py modeling --modeling_type doc2vec --training --sanity_check
                    python main.py modeling
                    python main.py modeling --modeling_type doc2vec
                
ANALYIS (depends on id_handling and modeling):
    - Analyzes the Testdata to find duplicates inside the Testdata (inside-analysis)
    - Analyzes the Testdata to find duplicates for the Testdata in the Trainingdata (outside-analysis)

            python main.py analyze
            (calls inside and outside analysis)
            optional: --analysis_type inside or --analysis_type outside or --analysis_type complete
            default: complete

            specification: 
                a. --analysis_type inside: following further statements can be added
                        --preprocessing
                        --pairing
                        --calculation
                        --evaluation
                        --jaccard (only relevant for method Shingling_Similarity)
                        --method_in
                            further methode choices:
                                levenshtein
                                countvec
                                tfidf
                                doc2vec
                                shingling
                b. --analysis_type outside: follwing further statements can be added
                        --method_out
                            tfidf or 
                            doc2vec (depending on which model you want to use, default: doc2vec)
                        --preprocessing
                        --mostsim
                        --pairing

ALL_IN_ONE:
    Possibility to call the whole process in one:
        python main.py all_in_one

        - gives unique_ids to trainingdata and testdata
        - trains doc2vec model with trainingdata
        - analyzes inside and outside """

# ## Imports
from modeling import modeling_dist
from analysis import analysis_in
from analysis import analysis_out
from id_handling import id_manager
import os
import pandas as pd 
import logging
import time
import argparse
from typing import Union
import sys

# ## Define Dictionaries to store the data (consisting of Dataframes)
dict_testdata = dict()
dict_traindata = dict()

# ## Initiate Logging-Module
logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.DEBUG, filename='logger.log', filemode='w+',
)

# ########## FUNCTIONS ##########

# ## ALL_IN_ONE
# Calls whole programm: id_handling, modeling and analysis
def all_in_one(args: dict, dict_testdata: dict, dict_traindata: dict) -> None:
    '''Function to call all parts of the program: id_handling (test- & train-data), modeling (doc2vec), analysis (inside & outside)

        Parameters
        ----------
        args : dict
            The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
        dict_testdata : dict
            Dictionary with testdata -> keys: table_names, values: DataFrames with job-ads
        dict_traindata: dict
            Dictionary with trainingdata -> keys: table_names, values: DataFrames with job-ads
        '''

    dict_testdata, dict_traindata = id_handling(args, dict_testdata, dict_traindata)
    modeling(args, dict_testdata, dict_traindata)
    analyze(args, dict_testdata, dict_traindata)


# ## ID_HANDLING
def id_handling(args: dict, dict_testdata: dict, dict_traindata: dict) -> Union[dict, dict]:
    '''Function to generate and give unique_ids to the Input-Data. Optional: Testdata or Trainingsdata

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
    logging.info('\n\n***** ID_HANDLING *****\n')
    dict_testdata, dict_traindata = id_manager(args, dict_testdata, dict_traindata)
    logging.info('\n\n***** /ID_HANDLING *****\n')
    return dict_testdata, dict_traindata

# ## MODELING
# Training & Retraining & Sanity_check of Models
def modeling(args:dict, dict_testdata: dict, dict_traindata: dict) -> None:
    '''Function to manage the training-process. Maintains training, retraining(only doc2vec) and sanity_check for models.

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata AND UNIQUE_IDS -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata AND UNIQUE_IDS -> keys: table_names, values: DataFrames with job-ads
    '''

    logging.info('\n\n***** MODELING *****\n')
    modeling_dist(args, dict_testdata, dict_traindata)
    logging.info('\n\n***** /MODELING *****\n')

# ## ANALYSIS
# Analysis of the Data: Including Calculation and Evaluation
def analyze(args: dict, dict_testdata: dict, dict_traindata: dict) -> None:
    '''Function to manage the analysis of the data: Optional: analysis_inside and/or analysis_outside.

    Parameters
    ----------
    args : dict
        The ArgumentParser values to manage which part(s) of the program need(s) to be addressed
    dict_testdata : dict
        Dictionary with testdata AND UNIQUE_IDS -> keys: table_names, values: DataFrames with job-ads
    dict_traindata: dict
        Dictionary with trainingdata AND UNIQUE_IDS -> keys: table_names, values: DataFrames with job-ads '''

    logging.info('\n\n***** ANALYSIS ****\n')
    method_args = vars(args)
    if method_args['analysis_type'] == 'inside':
        logging.info('\n\n***** ANALYSIS_INSIDE *****')
        analysis_in(args, dict_testdata, dict_traindata)
    elif method_args['analysis_type'] == 'outside':
        logging.info('\n\n***** ANALYSIS_OUTSIDE *****')
        analysis_out(args, dict_testdata, dict_traindata)
    else:
        logging.info('\n\n*****ANALYSIS_INSIDE *****')
        analysis_in(args, dict_testdata, dict_traindata)
        logging.info('\n\n***** ANALYSIS_OUTSIDE *****')
        analysis_out(args, dict_testdata, dict_traindata)
    logging.info('\n\n***** /ANALYSIS ****\n')

# ########## /FUNCTIONS ##########


# ########## ARGUMENT PARSER ##########

def get_application_parser() -> argparse.ArgumentParser:
    """ Function to generate an ArgumentParser and return given arguments from CLI
    Returns
    ----------
    application_parser : parser
        Parser contains the subparsers: id_handling, modeling, analysis and all_in_one """
        
    # ## create parser
    application_parser = argparse.ArgumentParser(description='Find Duplicates in job-ads')

    # ## create subparsers
    subparsers = application_parser.add_subparsers()
    subparsers.required = True

    # 1. subparser 'id_handling"
    idhandling_parser = subparsers.add_parser('id_handling', 
                                                description='gives unique_ids to input-data (Trainingdata and Testdata) and stores them in /temp/id_test_data.db and /temp/id_train_data.db')
    idhandling_parser.add_argument('--test', action="store_true")
    idhandling_parser.add_argument('--train', action="store_true")
    idhandling_parser.set_defaults(func=id_handling)

    # 2. subparser 'modeling'
    modeling_parser = subparsers.add_parser('modeling', description='trains, retrains and checks sanity of models --> models saved in folder dupl/models (one level higher than working_dir)')
    modeling_parser.add_argument('--modeling_type', choices=['tfidf', 'doc2vec'],
                                   default='doc2vec')
    modeling_parser.add_argument('--training', action="store_true")
    modeling_parser.add_argument('--retraining', action="store_true")
    modeling_parser.add_argument('--sanity_check', action="store_true")
    modeling_parser.set_defaults(func=modeling)

    # 3. subparser 'analysis'
    analysis_parser = subparsers.add_parser('analyze', description='')
    analysis_parser.add_argument('--preprocessing', action="store_true")
    analysis_parser.add_argument('--pairing', action="store_true")
    analysis_parser.add_argument('--calculation', action="store_true")
    analysis_parser.add_argument('--evaluation', action="store_true")
    analysis_parser.add_argument('--jaccard', action="store_true")
    analysis_parser.add_argument('--mostsim', action="store_true")
    analysis_parser.add_argument('--method_out', 
                                    choices=['doc2vec', 'tfidf'],
                                    default='doc2vec')
    analysis_parser.add_argument('--method_in', 
                                    choices=['levenshtein', 'countvec', 'tfidf', 'doc2vec', 'shingling'],
                                    default='doc2vec')
    analysis_parser.add_argument('--analysis_type', choices=['inside', 'outside', 'complete'],
                                   default='complete')
    analysis_parser.set_defaults(func=analyze)

    # 4. subparser 'all_in_one'
    allinone_parser = subparsers.add_parser('all_in_one', description='')
    allinone_parser.add_argument('--modeling_type', choices=['tfidf', 'doc2vec'],
                                   default='doc2vec')
    allinone_parser.add_argument('--test', action="store_true")
    allinone_parser.add_argument('--train', action="store_true")
    allinone_parser.add_argument('--preprocessing', action="store_true")
    allinone_parser.add_argument('--pairing', action="store_true")
    allinone_parser.add_argument('--calculation', action="store_true")
    allinone_parser.add_argument('--jaccard', action="store_true")
    allinone_parser.add_argument('--evaluation', action="store_true")
    allinone_parser.add_argument('--mostsim', action="store_true")
    allinone_parser.add_argument('--training', action="store_true")
    allinone_parser.add_argument('--retraining', action="store_true")
    allinone_parser.add_argument('--sanity_check', action="store_true")
    allinone_parser.add_argument('--method_out', 
                                    choices=['doc2vec', 'tfidf'],
                                    default='doc2vec')
    allinone_parser.add_argument('--method_in', 
                                    choices=['levenshtein', 'countvec', 'tfidf', 'doc2vec', 'shingling'],
                                    default='doc2vec')
    allinone_parser.add_argument('--analysis_type', choices=['inside', 'outside', 'complete'],
                                    default='complete')
    allinone_parser.set_defaults(func=all_in_one)

    return application_parser

# ########## /ARGUMENT PARSER ##########

# ########## START & FINISH PROGRAM ##########

if __name__ == '__main__':
    logging.info('\nThe Program started. More information about the process can be found in the file "logger.log".')
    logging.info('\n\n******************************** The program started. ********************************\n')
    print('\n\n******************************** The program started. ********************************\n')
    parser = get_application_parser()
    arguments = parser.parse_args()
    logging.info(f'The arguments {arguments} are given.')
    logging.info(f'The function {vars(arguments)["func"]} is called.')
    arguments.func(arguments, dict_testdata, dict_traindata)
    print('Processing done. For further information see logger-file.')
    print('\n\n******************************** The program finished. ********************************\n')
    logging.info('\n\n******************************** The program finished. ********************************\n')
    sys.exit(1)

# ########## /START & FINISH PROGRAM ##########