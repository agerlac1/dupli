# *** Formulas ***
""" Script calls the chosen function depending on step_key. 
    Options:
        a. Countvec_Similarity
        b. Levenshtein_Distance
        c. Tfidf_Similarity
        d. Doc2Vec_Similarity
        e. Shingling_Similarity """

# ## Imports
from . import formula_countveccosine
from . import formula_doc2veccosine
from . import formula_levenshtein
from . import formula_tfidfcosine
from . import formula_shinglingcosine

# ## Set Variables
sim_score = None

# ## Function
def distributor (step_key: str, input_full_a: str, input_full_b: str, jaccard: bool) -> float:
    """ Uses step_key to choose method for calculating similarity between two strings.
    Parameters
    ----------
    step_key: str
        String with the step_key to define which parts of the program need to be used.
    input_full_a: str
        String contains preprocessed full_text from one job-ad
    input_full_b: str
        String contains preprocessed full_text from one job-ad
    jaccard: bool
        Boolean from ArgumentParser to decide if inside method Shingling_Similarity the jaccard- or cosine - similarity needs to be calculated.

    Returns
    -------
    sim_score: float
        score is a sim_score between 0 and 1. Describes similarity between the processed strings."""

    """except TypeError:
            print("Cosine could not be calculated ") Das hier ansehen war bisher wegen cosine = NoneType in doc2vec"""

    if step_key == 'countvec':
        sim_score = formula_countveccosine.calculate_countveccosine(input_full_a, input_full_b)
    elif step_key == 'levenshtein':
        sim_score = formula_levenshtein.calculate_levenshtein(input_full_a, input_full_b)
    elif step_key == 'tfidf':
        sim_score = formula_tfidfcosine.calculate_tfidfcosine(input_full_a, input_full_b)
    elif step_key == 'doc2vec':
        sim_score = formula_doc2veccosine.calculate_doc2veccosine(input_full_a, input_full_b)
    elif step_key == 'shingling':
        sim_score = formula_shinglingcosine.calculate_shinglingcosine(input_full_a, input_full_b, jaccard)

    return sim_score