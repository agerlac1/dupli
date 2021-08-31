# *** TF-IDF - Find most similar ***
""" Script to vectorize the passed datasets, compute cosine between the datasets and get the most_similar documents.
Inspired by: https://goodboychan.github.io/chans_jupyter/python/datacamp/natural_language_processing/2020/07/17/04-TF-IDF-and-similarity-scores.html"""

# ## Imports
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from modeling import tfidf
from typing import Union
import itertools
import logging

# ## Set Variables
tfidf_model = None

def most_similar(dict_testdata_prepro: dict, dict_traindata_prepro: dict) -> Union[dict, list, list]:
    """ Uses the preprocessed test- and traindata (dict with unique_ids and OneStrings) to compute cosine between those with the TF-IDF model 
        and finds the most_similar job-ads.
    
    Parameters
    ----------
    dict_testdata_prepro: dict
        Dictionary with preprocessed testdata -> keys: table_names, values: Dict (unique_ids and Onestrings)
    dict_traindata_prepro: dict
        Dictionary with preprocessed traindata -> keys: table_names, values: Dict (unique_ids and Onestrings)

    Returns
    -------
    sims_dict: dict
        Dict contains each testdata unique_id and the depending traindata_unique_id -> keys: testdata unique_id, values: most similar unique_ids from traindata
    testdata_sims_ids: list
        List contains all testdata unique_ids from sims_dict keys
    traindata_sims_ids: list 
        List contains all traindata unique_ids from sims_dict values """

    # load the model
    tfidf_model = tfidf.load_model("tfidf_model")

    # set vars for most_similar unique_ids
    sims_dict = dict()
    testdata_sim_ids = list()
    traindata_sim_ids = list()

    # write all ids and OneStrings in one list and map them in a DataFrame for train- and testdata
    training_data = pd.DataFrame([(i, j) for i, j in [dic.items() for dic in dict_traindata_prepro.values()][0]], 
                columns=['unique_id','OneString'])

    testing_data = pd.DataFrame([(i, j) for i, j in [dic.items() for dic in dict_testdata_prepro.values()][0]], 
                columns=['unique_id','OneString'])
    
    # Set indices to find the needed data
    indices_train = pd.Series(training_data.index, index=training_data['unique_id'])
    indices_test = pd.Series(testing_data.index, index=testing_data['unique_id'])
    indices = indices_train.append(indices_test)

    # Construct the TF-IDF Matrix with the two input sets
    train_vecs = tfidf_model.transform(training_data['OneString'])
    test_vecs = tfidf_model.transform(testing_data['OneString'])

    # Generate the cosine similarity matrix
    cosine_sim = linear_kernel(test_vecs, train_vecs)

    # Extract most_similar traindata-ids for each testdata-id and store them in sims_dict
    for uid in testing_data['unique_id']:
        sims = __get_most_similar(uid, cosine_sim, indices, training_data)
        sims_dict[uid] = sims.to_list()

    # Store unique_ids from testdata on list testdata_sim_ids and uids from traindata on list traindata_sim_ids 
    testdata_sim_ids = list(sims_dict.keys())
    traindata_sim_ids = list(itertools.chain.from_iterable(sims_dict.values()))

    # Log random sample
    logging.info('\n\n*** Random sample from Analysis_outside for Evaluation ***\n')
    logging.info('Document ({}): «{}»\nhas the following most similar unique_id matches\n{}'.format(uid, testing_data.loc[testing_data['unique_id'] == uid, 'OneString'], sims.to_list()))
    
    return sims_dict, testdata_sim_ids, traindata_sim_ids

def __get_most_similar(uid, cosine_sim, indices, training_data):
    # Get the index of the job-ad that matches the uid
    idx = indices[uid]
    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the job-ads based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[0:10]
    # Get the jobs-ads with the indices
    # sim_scores is a list of lists (indx, score), (indx, score)... with the indx we can extract the uid form the indices (above)
    data_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return training_data['unique_id'].iloc[data_indices]