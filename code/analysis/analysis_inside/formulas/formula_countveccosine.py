# *** CountVectorizer and Cosine-Similarity ***
""" Script receives two strings and calculates the similarity with the method CountVectorizer and computes the Cosine-Similarity """

# ## Imports
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

def calculate_countveccosine(input_full_a: str, input_full_b: str) -> float:
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
        
    # Create a Vectorizer Object 
    vectorizer = CountVectorizer() 
    # fit the vocab and transform texts
    vectors = vectorizer.fit_transform([input_full_a, input_full_b])   
    # calculate cosine for the vectors
    cosine = cosine_similarity(vectors[0], vectors[1])[0][0]
    # return cosine score
    return cosine