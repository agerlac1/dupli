# *** W-Shingling and Cosine/Jaccard ***
""" Scripts receives two strings and a boolean. 
    Transforms both strings in Shingle-Sets and computes Cosine- or Jaccard-Similarity depending on passed boolean.

    Inspired by https://www.bogotobogo.com/python/python_sets_union_intersection.php 
    and https://gist.github.com/gaulinmp/da5825de975ed0ea6a24186434c24fe4 """

# ## Imports
import re
import math
from collections import Counter

# One Shingle is a string with K-words
K = 2

# ## Functions
def calculate_shinglingcosine(input_full_a: str, input_full_b: str, jaccard: bool) -> float:
    """ Generate two sets of Shingles for each received input string. 
    Use passed jaccard-value to compute the similarity between the two sets.
    if jaccard == true: take shingle-sets and compute jaccard
    if jaccard == false: transform shingle-sets in countvectors and compute cosine.
    

    Parameters
    ----------
    input_full_a: str
        String contains preprocessed full_text from one job-ad
    input_full_b: str
        String contains preprocessed full_text from one job-ad
    jaccard: bool
        Boolean from ArgumentParser to decide if inside method Shingling_Similarity the jaccard- or cosine - similarity needs to be calculated.

    Returns
    -------
    ratio: float
        ratio is a sim_score between 0 and 1. Describes similarity between the processed strings."""
    
    # Set shingles list
    shingles = list()
    # Generate two shingle-sets from input texts
    sh_set1 = __gen_shingle_set(input_full_a)
    sh_set2 = __gen_shingle_set(input_full_b)
    
    # Decide which Similarity-measure to use.
    # JACCARD
    if jaccard == True:
        # shingles : list of sets (sh)
        shingles.append(sh_set1)
        shingles.append(sh_set2)
        # compare each pair in combinations tuple of shingles
        ratio = __jaccard_set(shingles[0], shingles[1])
        return ratio
    # COSINE
    else:
        # generate countvectors for shingle-sets
        vec1 = Counter(sh_set1)
        vec2 = Counter(sh_set2)    
        # calculate cosine for the vectors
        ratio = __cosine(vec1, vec2)
        return ratio

# Compute Cosine for two vectors
def __cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

# Compute Jaccard for two sets of shingles (k-token)
def __jaccard_set(s1, s2):
    # takes two sets and returns Jaccard coefficient
    u = s1.union(s2)
    i = s1.intersection(s2)
    return len(i)/len(u)

# Generate shingle sets with K-token
def __gen_shingle_set(doc):
    # replace non-alphanumeric char with a space, and then split
    tokens = re.sub(r"\W", " ",  doc).split()
    sh = set()
    for i in range(len(tokens)-K):
        t = tokens[i]
        for x in tokens[i+1:i+K]:
            t += ' ' + x 
        sh.add(t)
    return sh