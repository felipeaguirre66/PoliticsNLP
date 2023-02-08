# Some imports
import pandas as pd
import regex as re
import unidecode
import nltk
import spacy




"""
    1. Preprocess functions

"""

import stop_words as sw
stop_words = sw.get_stop_words('es')


class Preprocess():

    """
        Preprocess: keep only alphanumeric, lemmatized and drop stop words.
        Join: returns words in a str

        Input: str, list of str or column with str

    """
    
    def __init__(self, lemma=True, decode=True, alphanumeric=True, stopwords=True, join=True):
        self.nlp = spacy.load('es_core_news_lg')
        
        self.lemma = lemma
        self.decode = decode
        self.alphanumeric = alphanumeric
        self.stopwords = stopwords
        self.join = join
    
    def preprocess(self, textos):

        if type(textos) == pd.core.series.Series: # si es una columna, lo pasa a lista
            textos = textos.astype(str).to_list()

        elif type(textos) == str: # si es un str, lo pasa a lista
            textos = [textos]
    

        pre_processed = []
        for text in textos:
            if self.lemma:
                text = self.nlp(text)
                text = [word.lemma_ for word in text]
                text = ' '.join(text)
            if self.decode:
                text = unidecode.unidecode(text.lower().strip())
            if self.alphanumeric:
                text = re.findall('\w+', text)
            if self.stopwords:
                text = [i for i in text if i not in stop_words]
            if self.join:
                text = ' '.join(text)

            pre_processed.append(text)

        return pre_processed




"""
    2. Count and plot words functions.

"""

import matplotlib.pyplot as plt
from collections import Counter

def count_words(words_list):
    
    """ Count words in a document and return its elements and frequencies. """
    
    # Flatten the list of lists into a single list
    if type(words_list[-1]==list):
        words_list = [item for sublist in words_list for item in sublist]
    
    counts = Counter(words_list)
    
    # Sort the counts in descending order
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract the elements and their frequencies from the dictionary
    elements, frequencies = zip(*counts)

    return elements, frequencies


def plot_word(elements, frequencies, plot_title = 'Plot', N = 20):
    plt.barh(elements[:N][::-1], frequencies[:N][::-1])
    plt.title(plot_title)
    plt.show()
    
    
def string_to_tuple(myStr):
    """
    Converts string '(1,2)' into tuple (1,2) (useful when reading xlsx files)'
    """
    myStr = myStr.replace("(", "")
    myStr = myStr.replace(")", "")
    myStr = myStr.replace(",", " ")
    myTuple = myStr.split()
    myTuple = tuple(map(int, myTuple))
    return myTuple