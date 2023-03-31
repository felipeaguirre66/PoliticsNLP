import numpy as np

def semantic_words_distance(u,v):
    """
    Author: Franco Ferrante
    
    Given two vectors, it calculates the semantic word distance as the 
    cosine of the angle between the two vectors as (Sanz et al. 2021)

    Parameters
    ----------
    u : float list
        A word embedding.
    v : float list
        A word embedding.

    Returns
    -------
    float
        Cosine Distance: the semantic word distance between u and v.

    """
    return 1-(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))

def ongoing_semantic_variability(vector_distances):
    """
    Author: Franco Ferrante
    
    Given a vector, it calculates the ongoing semantic variability as defined at
    (Sanz et al. 2021)

    Parameters
    ----------
    vector_distances : float list
        A list where each float element represents the semantic distance 
        between two words.

    Returns
    -------
    average : float list
        A list where each float element represents the ongoing semantic
        variability.

    """
    if (len(vector_distances) < 2):
        return np.nan
    summation = sum((vector_distances-np.mean(vector_distances))**2)
    average = summation/(len(vector_distances))
    return average


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

def add_accents(word:str):
    # Create list of all posible accents for a word
    accents = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
    accented_words = []
    for i, char in enumerate(word):
        if char in accents:
            accented_words.append(word[:i] + accents[char] + word[i+1:])
    return accented_words

def is_noun(word):
    # Returns True only if word is noun
    accented_words = add_accents(word)
    accented_words.append(word)
    return any(lemma.synset().pos() == 'n' for aw in accented_words for lemma in wn.lemmas(aw, lang='spa'))

def get_word_granularity(word):
    # Returns the word granularity
    # if not is_noun(word): return None
    synsets = wordnet.synsets(word, lang='spa')
    if not synsets:
        found = False
        accented_words = add_accents(word) #probar agregando acento
        for aw in accented_words:
            synsets = wordnet.synsets(aw, lang='spa')
            if synsets:
                found = True
                break
        if not found: return None
    
    max_depth = 0
    for synset in synsets:
        for path in synset.hypernym_paths():
            depth = len(path)
            if depth > max_depth:
                max_depth = depth
    return max_depth