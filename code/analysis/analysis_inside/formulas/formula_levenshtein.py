# *** Levenshtein-Ratio ***
"""Script receives two strings and computes Levenshtein-Ratio"""

# ## Imports
import Levenshtein as lev

# ## Function
def calculate_levenshtein(input_full_a: str, input_full_b: str) -> float:
    """ Compute Levenshtein-Ratio for the two input-strings.

    Parameters
    ----------
    input_full_a: str
        String contains preprocessed full_text from one job-ad
    input_full_b: str
        String contains preprocessed full_text from one job-ad

    Returns
    -------
    ratio: float
        ratio is a sim_score between 0 and 1. Describes similarity between the processed strings."""
    
    # Compute similarity between two strings
    ratio = lev.ratio(input_full_a,input_full_b)
    # Return ratio
    return ratio