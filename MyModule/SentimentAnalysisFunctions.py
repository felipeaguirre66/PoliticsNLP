# Some imports
import pandas as pd
import regex as re
import unidecode
import nltk
import spacy


from MyModule.GeneralFunctions import *


"""
    3. Sentiment analysis functions.

"""

from pysentimiento import create_analyzer
from sentiment_analysis_spanish import sentiment_analysis

class sentiment_analyzer_3d():
    
    def __init__(self):
        self.analyzer = create_analyzer(task="sentiment", lang="es")
        pass
    
    def predict_sentiment_3d(self, phrase, sent = ['POS', 'NEG', 'NEU'], print_=False):
        prediction = self.analyzer.predict(phrase)
        
        emotions = []
        for s in sent:
            emotions.append(prediction.probas[s])
            if print_: print(f'{s}: {prediction}')
        
        return emotions
        
        
        
def analyse_sentiment_1d(phrase): 
    
    """ Predicts sentiment of phrase in one dimension (-1 to 1) for NEG, NEU, POS. """
    
    analyzer = sentiment_analysis.SentimentAnalysisSpanish()     
    return analyzer.sentiment(phrase)