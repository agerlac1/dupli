# *** Doc2Vec Sanity_Check sanity_check.py***
""" Script checks the sanity of a Doc2Vec model.
    Therefore the script follows the steps:
        1. sanity_check (checks if data finds itself as most similar)
            --> Inspired by: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
        2. evaluate_sample (gives random sample for evaluation) 
    
    ! Important ! Check Settings in config.yaml. Which model do you want to evaluate? Check d2v_model_type."""

# ## Imports
import gensim
import os
from nltk.corpus import stopwords 
import collections
import random
from modeling import doc2vec
import yaml 
from pathlib import Path
import logging

# ## Set Variables
d2v_model = None

# ## Open Configuration-file and set type of doc2vec model. Which model do you want to check? d2v_model or d2v_remodel? Set config.yaml parameter.
with open(Path("config.yaml"), "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    d2v_model_type = cfg['d2v_model_type']
    d2v_type = d2v_model_type['type']

# ## Functions
def start_sanity_check(dict_traindata_prepro: dict) -> None:
    """ 1. sanity_check (checks if data finds itself as most similar)
            - iterate over each training-datapiece and infer it in the model
            - check the most_similar job-ads and store the ranking in ranks (has one datapiece1 itself as the most_similar (in Top 10) in the model?)
            - log a sample and the counting result in logger.log

    Parameters
    ----------
    dict_traindata_prepro: dict
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) """

    # load the model to be checked once
    d2v_model = __envoke_model()

    # ## Set Variables
    ranks = list()
    counter = 0
    
    # CHECK MOST_SIMILAR
    for name, train_corpus in dict_traindata_prepro.items():
        logging.info(f'Table {name} is now infered in itself and checked for most_similar uids.')
        for item in train_corpus:
            # infer text in model and get most_similar back
            sims = d2v_model.docvecs.most_similar([d2v_model.infer_vector(item.words)])
            counter += 1
            try:
                # check if uid of item (item.tags[0]) is in the list sims (most_similar top10)
                rank = [docid for docid, sim in sims].index(item.tags[0])
                ranks.append(rank)
            except:
                # item does not find itself in the most_similar Top10
                continue

    # log example from data (most, second-most, median, least)-similar texts for item
    __give_example(item, sims, train_corpus)

    # log ranks e.g.: Counter({1: 6, 0: 6, 2: 3, 3: 2, 5: 1, 6: 1, 9: 1}) (position 1: are 6 docs -> means: 6 documents found themselfs as most_similar)
    logging.info(f'In model {d2v_model} {counter} documents were infered. Ranks for most_similar Top10: {collections.Counter(ranks)}')
 

def evaluate_sample(dict_testdata_prepro: dict, dict_traindata_prepro: dict) -> None:
    """ 2. evaluate_sample (gives random sample for evaluation)
            a. RANDOM WORD from Testdata: get most_similar words from model
            b. RANDOM DOC from Testdata: get most_similar dos from model
            c. RANDOM WORD from Traindata: get most_similar words from model
            d. RANDOM DOC from Traindata: get most_similar words from model

    Parameters
    ----------
    dict_traindata_prepro: dict
        Dictionary with trainingdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) 
    dict_testdata_prepro: dict
        Dictionary with testdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) 
    
    Raises
    ------
    KeyError
        If no similar Text or Word is found, exception gets logged. """

    # load the model to be checked once
    d2v_model = __envoke_model()
    
    # Zip test- and traindata in a dict for looping purposes
    data_dict ={'test_data' : dict_testdata_prepro, 'train_data' : dict_traindata_prepro}
    for name, data in data_dict.items():
        # Select a random object from data
        random_obj = random.choice(list(random.choice(list(data.values()))))
        random_doc = random_obj.words
        random_word = random.choice(list(random_obj.words))
        random_unique_id = random_obj.tags
        
        logging.info(f"\n\nSample evaluation for {name} in {d2v_model}")
        try:
            mostsim_docs = d2v_model.docvecs.most_similar(positive = [d2v_model.infer_vector(random_doc)])
            logging.info(f"Random text from {name} with unique_id {random_unique_id} has the following most similar documents in traindata:\n{mostsim_docs}")
        except KeyError:
            logging.info(f"For the text {random_unique_id} from {name} no similar texts could be found, because text/words are not in vocabulary.")
            pass
        try:
            mostsim_words = d2v_model.most_similar(random_word)
            logging.info(f"Random word {random_word} from text {random_unique_id} from {name} has the following most similar words:\n{mostsim_words}")
        except KeyError:
            logging.info(f"For the word {random_word} from text {random_unique_id} from {name} no similar words could be found, because word is not in vocabulary. Better to retrain the model with more training_data")
            pass

# ## Private Functions

def __envoke_model(): 
    global d2v_model
    if d2v_model is None:
        # decide which one to use (d2v_model or d2v_remodel) depends on function (change in config!)
        d2v_model = doc2vec.load_model(d2v_type)
    return d2v_model

def __give_example(item, sims, train_corpus):
    """ Log most-similar, second-most, median and least documents (uid and text) for one document."""
    logging.info(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:' % d2v_model)
    logging.info('Document ({}): «{}»\n'.format(item.tags[0], ' '.join(item.words)))
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        try:
            logging.info(u'%s %s: «%s»' % (label, sims[index], ' '.join(train_corpus[([item.tags[0] for item in train_corpus].index(sims[index][0]))].words)))
            continue
        except:
            logging.info(f'For {label} Element not found, skip it.')
            continue 