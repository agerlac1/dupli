# *** Doc2Vec - Find most similar***
""" Script to infer the passed TaggedDocument-objects and extract the most_similar documents from model."""

# ## Imports
import os
from nltk.corpus import stopwords 
import collections
from modeling import doc2vec
from typing import Union
import yaml
from pathlib import Path
import logging
import itertools

# ## Set Variables
dict_testdata_prepro = dict()

# ## Open Configuration-file and set type of doc2vec model
with open(Path("config.yaml"), "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    d2v_model_type = cfg['d2v_model_type']
    d2v_type = d2v_model_type['type']


# ## Function
def finder(dict_testdata_prepro: dict) -> Union[dict, list, list]:
    """ Uses the preprocessed testdata (as taggedobjects) to infer them in vectors and find the most_similar job-ads saved in the model.
    
    Parameters
    ----------
    dict_testdata_prepro: dict
        Dictionary with preprocessed and tagged testdata -> keys: table_names, values: list(TaggedDocument-objects(tokens_list, unique_id))

    Returns
    -------
    sims_dict: dict
        Dict contains each testdata unique_id and the depending traindata_unique_id -> keys: testdata unique_id, values: most similar unique_ids from traindata
    testdata_sims_ids: list
        List contains all testdata unique_ids from sims_dict keys
    traindata_sims_ids: list 
        List contains all traindata unique_ids from sims_dict values """
    
    # pass type of d2v_model you want to use to find most similar (d2v_model or d2v_remodel, change in config.yaml manually)
    d2v_model = doc2vec.load_model(d2v_type)

    # Set Variables
    sims_dict = dict()
    testdata_sim_ids = list()
    traindata_sim_ids = list()

    def __infer_texts():
        # Iterate over tables in dict
        for table_content in dict_testdata_prepro.items():
            # Iterate over each datapiece in table
            for item in table_content[1]:
                # Infer_vector with token and find most_similar in docvecs from model
                sims = d2v_model.docvecs.most_similar([d2v_model.infer_vector(item.words)])
                # Store most_similars for each item (job-ad) in the sims_dict (keys: item id, values: sims_ids)
                sims_dict[item.tags[0]] = [i[0] for i in sims]
        try:
            logging.info('\n\n*** Random sample from Analysis_outside for Evaluation ***\n')
            logging.info('Document ({}): «{}»\nhas the following most similar unique_id matches\n {}'.format(item.tags[0], ' '.join(item.words), sims))
        except:
            pass
    __infer_texts()

    # Store unique_ids from testdata on list testdata_sim_ids and uids from traindata on list traindata_sim_ids 
    testdata_sim_ids = list(sims_dict.keys())
    traindata_sim_ids = list(itertools.chain.from_iterable(sims_dict.values()))

    return sims_dict, testdata_sim_ids, traindata_sim_ids