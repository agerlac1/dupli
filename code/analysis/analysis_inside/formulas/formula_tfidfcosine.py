# *** TF-IDF-Vectorizer & Cosine-Similarity ***
""" Script receives two input-strings and transforms them in vectors with the TF-IDF Vectorizer and computes the Cosine-Similarity.
Inspired by: https://www.kaggle.com/currie32/predicting-similarity-tfidfvectorizer-doc2vec

    * First approach: load learned vocab and transform documents to vectors -> compute cosine for vectors
        --> better results
    * Second approach: (commented out) no model needs to be loaded. fit_transform(string1, string2) -> compute cosine for vectors 
        --> only for replication of masterthesis needed """

# ## Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modeling import tfidf

# ## Set Variables
tfidf_model = None

# ## Functions

def calculate_tfidfcosine(input_full_a: str, input_full_b: str) -> float:
    """ Transform two strings in vectors with the TfidfVectorizer() and compute cosine-similarity.

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

    # ** FIRST APPROACH **:
    # Load model and get object (only loads once and only if script is used)
    tfidf_model = __envoke_model()
    # transform texts
    vectors = tfidf_model.transform([input_full_a, input_full_b])
    # calculate cosine between vectors
    cosine = cosine_similarity(vectors[0], vectors[1])[0][0]

    # ** SECOND APPROACH **:
    """ # No Tf-idf model needed. Use texts to fit the vocab and then transform them in vectors + cosine.
    # Create a Vectorizer Object 
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    # fit the vocab
    tfidf = vectorizer.fit([input_full_a], [input_full_b])
    # transform texts to vectors 
    vectors = tfidf_model.transform([input_full_a, input_full_b])
    # calculate cosine between vectors
    cosine = cosine_similarity(vectors[0], vectors[1])[0][0] """

    return cosine

def __envoke_model(): 
    global tfidf_model
    if tfidf_model is None:
        tfidf_model = tfidf.load_model('tfidf_model')
    return tfidf_model