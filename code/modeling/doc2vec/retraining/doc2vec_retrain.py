# *** Doc2Vec Retraining***
""" Script retrains a Doc2Vec model."""

# ## Imports
from modeling import doc2vec
import logging

# ## Functions
def retrain_data(dict_testdata_prepro: dict):
    """ Method retrains a given Doc2Vec model and infers new vocab:
        - No parameter setting needed, uses old parameters (saved in model)
        - Load old model (to be updated)
        - Update Vocab with new Testdata
        - (Re)train the model
        - Save the model 

    Parameters
    ----------
    dict_testdata_prepro: dict
        Dictionary with testdata -> keys: table_names, values: Dict(keys: uid, values: preprocessed and tagged token) """
    
    # LOAD OLD MODEL
    model = doc2vec.load_model("d2v_model")

    # UPDATE VOCAB WITH NEW DATA
    data_retrain = sum(dict_testdata_prepro.values(), [])
    model.build_vocab(data_retrain, update=True) # update your vocab
    
    # RETRAIN THE MODEL
    model.train
    logging.info(f'For the model {len(model.docvecs)} DocVecs were computed.')
 
    # SAVE MODEL
    doc2vec.save_model(model, "d2v_remodel")