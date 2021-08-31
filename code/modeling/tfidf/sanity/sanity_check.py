# *** TF-IDF Sanity_Check sanity_check.py***
""" Script checks the sanity of a TF-IDF model.
    Therefore the script follows the steps:
        * evaluate_sample (vocab size and gives random samples for evaluation) 
            --> Inspired by: https://nlpforhackers.io/tf-idf/
    ! Important ! Check Settings in config.yaml. Which model do you want to evaluate? Check tfidf_model path."""

# ## Imports
from modeling import tfidf
import random
import logging

# ## Set Variables
tfidf_model = None

# ## Functions
def evaluate_sample(dict_testdata_prepro: dict, dict_traindata_prepro: dict) -> None:
    """ evaluate_sample (vocab size, vocabinformation and gives random samples for evaluation)
            
    Parameters
    ----------
    dict_traindata_prepro: dict
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed OneStrings) 
    dict_testdata_prepro: dict
        Dictionary with testdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed OneStrings) 
    
    Raises
    ------
    KeyError
        Raises Exception if a word is not part of the vocab. """

    tfidf_model = tfidf.load_model('tfidf_model')

    # IN GENERAL
    logging.info(f'\n\nSanity_check shows: vocabsize of the trained model {tfidf_model} is: {len(tfidf_model.get_feature_names())}')
    logging.info(f'10 random sample words from vocab: {random.sample(tfidf_model.get_feature_names(), 10)}')

    # Zip test- and traindata in a dict for looping purposes
    data_dict = {'test_data': dict_testdata_prepro, 'train_data': dict_traindata_prepro}

    # RANDOM SAMPLES
    for name, dict_ in data_dict.items():
        # SET VALUES
        # Choose random doc from data
        key, doc = random.choice(list(dict_.items())) 
        # Choose 5 random word from doc
        key, word = random.choice(list(doc.items()))
        # Sample words in list
        random_words = random.sample(word.split(), 5)
        # Check out some frequencies
        X = tfidf_model.transform([word])
        logging.info(f'\n\nRandom sample evaluation for {name} with {tfidf_model}: \n Document with unique_id {key} was chosen.')
        for word in random_words:
            try: 
                logging.info(f'The word: \"{word}\" has the tfidf-score:   {X[0, tfidf_model.vocabulary_[word]]}')
            except KeyError:
                logging.info(f'The word: \"{word}\" is not part of the vocabulary.')
                continue