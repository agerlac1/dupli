# *** Doc2Vec & Cosine-Similarity ***
""" Script receives two input-strings and transforms them in vectors with the Doc2Vec-model and computes the Cosine-Similarity.
    -> Different models and methods can be used.
    -> Models need to be trained (Testdata is unseen) or retrained (Testdata is seen) beforehand.

    For TRAINED models the following methods are given:
        * Method 1: similarity_unseen_docs() & returns cosine (best working method so far)
        * Method 2: infer_vector() + cosine_similarity
    --> Check that model "dv2_model" is envoked by function __envoke_model()

    For RETRAINED models the following methods are given:
        * Method 3: n_similartiy() & returns cosine
        * Method 4: infer_vector() + cosine_similarity
    --> Check that model "dv2_remodel" is envoked by function __envoke_model() """

# ## Imports
from modeling import doc2vec
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from pathlib import Path

# ## Set Variables
d2v_model = None

# ## Open Configuration-file and set type of doc2vec model
with open(Path("config.yaml"), "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    d2v_model_type = cfg['d2v_model_type']
    d2v_type = d2v_model_type['type']

# ## Functions
def calculate_doc2veccosine(input_full_a: str, input_full_b: str) -> float:
    """ Transform two strings in vectors with the CountVectorizer() and compute cosine-similarity.

    Parameters
    ----------
    input_full_a: str
        String contains preprocessed full_text from one job-ad
    input_full_b: str
        String contains preprocessed full_text from one job-ad

    Returns
    -------
    cosine: float
        cosine is a sim_score between 0 and 1. Describes similarity between the processed strings."""

    # Load the model
    d2v_model = __envoke_model()
    
    # Tokenize strings
    text1 = input_full_a.split()
    text2 = input_full_b.split()
   
    # *** TRAIN ***
    # Method 1: similarity_unseen_docs() (best working method so far)
    def __sim_unseen():
        cosine = d2v_model.docvecs.similarity_unseen_docs(d2v_model, text1, text2)
        return cosine
    cosine = __sim_unseen()

    # Method 2: infer_vector()
    def __infer_train():
        vector1 = d2v_model.infer_vector(text1,steps=5, alpha=0.025)
        vector2 = d2v_model.infer_vector(text2,steps=5, alpha=0.025)
        cosine = cosine_similarity([vector1], [vector2])[0][0]
        return cosine
    #cosine = __infer_train()

    # *** RETRAIN ***
    # Method 3: n_similartiy()
    def __n_sim():
        cosine = d2v_model.n_similarity(text1, text2)
        return cosine
    #cosine = __n_sim()

    # Method 4: Infer vector + cosine in retrained dataset
    def __infer_retrain():
        vector1 = d2v_model.infer_vector(text1,steps=5, alpha=0.025)
        vector2 = d2v_model.infer_vector(text2,steps=5, alpha=0.025)
        cosine = cosine_similarity([vector1], [vector2])[0][0]
        return cosine
    #cosine = __infer_retrain()

    # return similarity_score for processed pair of strings
    return cosine
    
def __envoke_model(): 
    global d2v_model
    if d2v_model is None:
        # decide which one to use (d2v_model or d2v_remodel) depends on function (in config)
        d2v_model = doc2vec.load_model(d2v_type)
    return d2v_model