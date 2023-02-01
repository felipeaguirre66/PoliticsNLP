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

def analyse_sentiment_3d(phrase, sent = ['POS', 'NEG', 'NEU']):
    
    """ Predicts sentiment of phrase in three dimensions: POS, NEG, NEU. """
    
    analyzer = create_analyzer(task="sentiment", lang="es")
    
    predictions = []
    for s in sent:
        prediction = analyzer.predict(phrase).probas[s]
        predictions.append(prediction)
        print(f'{s}: {prediction}')
        
        
        
def analyse_sentiment_1d(phrase): 
    
    """ Predicts sentiment of phrase in one dimension (-1 to 1) for NEG, NEU, POS. """
    
    analyzer = sentiment_analysis.SentimentAnalysisSpanish()     
    return analyzer.sentiment(phrase)