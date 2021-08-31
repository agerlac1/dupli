# ## Connections to SQL-Databases
""" Scripts gets called in every step of the program where the data needs to be extracted from a SQL-Database.
Establishes conncetions to the dbs and returns an sqlite3.Connection object. 
The modules of the application can call the connection_functions, but the paths are stored in the config.yaml file and dont need to be passed by functions. 
Thereby the paths only need to be adjusted manually from ONE File: config.yaml
"""

# ## Imports
import sqlite3
from sqlite3 import Error
from pathlib import Path
import yaml
import logging

# ## Set Variables
conn = None

# ## Open Configuration-File and set Paths
with open(Path('config.yaml'), 'r') as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    input_paths = cfg['input_paths']
    raw_train_data = input_paths['train_data']
    raw_test_data = input_paths['test_data']

    temp_paths = cfg['temp_paths']
    id_train = temp_paths['id_train']
    id_test = temp_paths['id_test']
    prepro_path = temp_paths['prepro_path']
    pair_path = temp_paths['pair_path']
    calc_path = temp_paths['calc_path']

    output_path = cfg['output_path']


# ## Functions
# ---------------------------------------------------------------------------------------------
# *** Connection to input-databases (without ids, the raw-input files) ***
# Different from the other functions because if databases do not exists an error-message is needed and program needs to stop!
# EXCEPTION wenn die database hier ein input sein soll, aber eben fehlt, wie bspw. die prepro daten. dann muss der step wiederholt werden.

# train data
def connect_raw_train_data() -> sqlite3.Connection: 
    """ creates a database connection to the SQLite database specified by Path from config.yaml

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """
    database = Path(raw_train_data)
    if not database.exists():
        logging.error(f'Database {database}does not exist.')
    else:
        # create a database connection
        conn = __create_connection(database)
        with conn:
            logging.info('Connection could be created. Return sqlite3.Connection object.')
            return conn

# test data
def connect_raw_test_data() -> sqlite3.Connection:
    """ creates a database connection to the SQLite database specified by Path from config.yaml

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """
    database = Path(raw_test_data)
    if not database.exists():
        logging.error(f'Database {database}does not exist.')
    else:
        # create a database connection
        conn = __create_connection(database)
        with conn:
            logging.info('Connection could be created. Return sqlite3.Connection object.')
            return conn
# ---------------------------------------------------------------------------------------------
# *** Connections to the id-databases (already got or are about to get unique_ids) ***

# train data with ids
def conn_training() -> sqlite3.Connection:
    """ creates a database connection to the SQLite database specified by Path from config.yaml

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """
    database = Path(id_train)
    conn = __check_existence(database)
    with conn:
        return conn

# test data with ids
def conn_testing() -> sqlite3.Connection:
    """ creates a database connection to the SQLite database specified by Path from config.yaml

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """
    database = Path(id_test)
    conn = __check_existence(database)
    with conn:
        return conn
# ---------------------------------------------------------------------------------------------
# *** Conncetions to the analysis_inside related databases ***

# Preprocessed data before pairing
def create_prepro_connection() -> sqlite3.Connection: 
    """ creates a database connection to the SQLite database specified by Path from config.yaml

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """
    database = Path(prepro_path)
    conn = __check_existence(database)
    with conn:
        return conn

# Paired and ids handled data
def create_pair_connection() -> sqlite3.Connection:
    """ creates a database connection to the SQLite database specified by Path from config.yaml

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """
    database = Path(pair_path)
    conn = __check_existence(database)
    with conn:
        return conn

# file used during analysis (backup file)
def create_temp_connection() -> sqlite3.Connection:
    """ creates a database connection to the SQLite database specified by Path from config.yaml

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """
    database = Path(calc_path)
    conn = __check_existence(database)
    with conn:
        return conn
# ---------------------------------------------------------------------------------------------
# *** Conncetion to the final Output-Database ***

def conn_final_output() -> sqlite3.Connection:
    """ creates a database connection to the SQLite database specified by Path from config.yaml

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """
    database = Path(output_path)
    conn = __check_existence(database)
    with conn:
        return conn
# ---------------------------------------------------------------------------------------------

# ## Private Functions
# ---------------------------------------------------------------------------------------------
# *** Creates the Conncetion depending on Path ***

def __create_connection(db_file: Path) -> sqlite3.Connection:
    """ create a database connection to the SQLite database
        specified by the db_file

    Parameters
    ----------
    db_file: Path
        Path to the Database to create a connection to

    Returns
    -------
    conn: sqlite3.Connection
        the Connection object
    """

    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        logging.error(e)
    logging.info('Connection could be created. Return sqlite3.Connection object.')
    return conn
# ---------------------------------------------------------------------------------------------
# *** Checks existence of databases and depending on result, logs message over status.

def __check_existence(database: Path) -> sqlite3.Connection:
    """ Checks existence of databases and logs accordingly."""
    if not database.exists():
        # EXCEPTION wenn die database hier ein input sein soll, aber eben fehlt, wie bspw. die prepro daten. dann muss der step wiederholt werden.
        logging.warning(f'Database {database} does not exist. If database is supposed to be the input data, it needs to be checked (error!). As output, the database will be created.')
        # create a database connection
        conn = __create_connection(database)
        with conn:
            return conn
    else:
        logging.info(f'Database {database} does exist. If used for Output, it will be overwritten.')
        # create a database connection
        conn = __create_connection(database)
        with conn:
            return conn
# ---------------------------------------------------------------------------------------------