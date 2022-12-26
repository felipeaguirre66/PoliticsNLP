
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



"""
    4. Topic Modeling functions.

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

def words_wheights(model, documents, n_components):
    
    """
    Applys model on documents.
    
    n_components: number of components or topics.
    
    Output: Every topic's wheight for each word + each word label
    
    """
    
    # Create the TF-IDF matrix
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # Perform Model
    model = model(n_components=n_components)
    model.fit(X)

    # Exchange key - value order
    terms = dict([(index, word) for word, index in vectorizer.vocabulary_.items()])

    all_words = []
    all_wheights = []

    for topic_idx, topic in enumerate(model.components_):
        
        # Labels of words (ordered by importance)
        words = [terms[i] for i in topic.argsort()[::-1]]
        all_words.append(words)
    
        # Wheights of words (ordered by importance)
        wheights = list(topic)
        wheights.sort(reverse=True)
        all_wheights.append(wheights)
        
    return all_words, all_wheights



"""
    5. Topic Modeling: evaluate performance functions.

"""

from numpy.linalg import norm
import numpy as np

def cos_similarity(A, B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine


def evaluate_coherence(model, documents, n_topics):
    
    """
    Evaluate Model's Topic Coherence by calculating mean cosine similarity between
    every pair of topic's vectors.
    
    -------
    
    Input: model (LSA or LDA), documents (list of str), n_topics (number of topics)
    Output: average cosine similarity
    
    """
    
    # Create the TF-IDF matrix
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # create an LSA model
    model = model(n_components=n_topics)

    # fit the LSA model to the preprocessed text data
    model.fit(X)

    # extract the topics from the LSA model
    topics = model.components_

    # initialize an empty list to store the topic coherence scores
    topic_coherence = []

    # cosine similarity between every pair of topics
    for i in range(topics.shape[0]):
        for i2 in range(topics.shape[0]):
            cos_sim = cos_similarity(topics[i], topics[i2])
            topic_coherence.append(cos_sim)
            
    return sum(topic_coherence)/len(topic_coherence)



from sklearn.decomposition import PCA

def visualize_topics(model, documents, num_top_words, n_components, des = ''):
    
    """
    Allows to visualize in 2D the topic distribution.
    
    This applys PCA=2 to the topic vectors, and then plots them
    
    Input: 
    a. model (LSA or LDA)
    b. documents: list of strings
    c. num_top_words (print Top K words more relevant to each topic)
    d. n_components (number of topics the model must find)
    
    """
    
    # Create the TF-IDF matrix
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # create an LSA model
    model = model(n_components=n_components)

    # fit the LSA model to the preprocessed text data
    model.fit(X)
    
    # reduce dimensions using PCA
    pca = PCA(n_components=2)
    topic_vectors = pca.fit_transform(model.components_)
    
    # plot the topic vectors
    print('Every topic with its most relevant words.\n\n')
    plt.figure(figsize=(8, 6))
    for i in range(n_components):
        plt.scatter(topic_vectors[i, 0], topic_vectors[i, 1])
        word_weight = dict([(index, word) for word, index in vectorizer.vocabulary_ .items()])
        top_words = [word_weight[j] for j in model.components_[i].argsort()[:-num_top_words - 1:-1]]
        label = f' Topic {i}'
        print(f'Topic {i} = {", ".join(top_words)}\n')
        plt.annotate(label, xy=(topic_vectors[i, 0], topic_vectors[i, 1]))
    
    explained_var = pca.explained_variance_ratio_ * 100
    
    if des:
        plt.title(f'Des {des} Topic Distribution')
    else:
        plt.title('Topic Distribution')
    plt.xlabel(f'PCA 1 %{round(explained_var[0])}')
    plt.ylabel(f'PCA 2 %{round(explained_var[1])}')
    plt.show()
    


"""
    6. Semantic Similarity: TF-IDF

"""

from sklearn.metrics.pairwise import cosine_similarity

class TF_IDF_Model():

    """
        Create a TF-IDF matrix. Find top N most similars, for new or existing (in df) documents.
        
        -----
        Input: 
        a. df: pandas DataFrame with at least this columns: 
            - ID (int): to identify the document
            - Data (str)
        
        b. Preprocess object
        
        ----

    """
    
    def __init__(self, df, pp_object = None):
        self.df = df
        self.pp_object = pp_object
    
    
    def train(self, column = 'texto'):

        """
        Necessary for predict_new and predict_old.

        ----
        Input:
        a. column: column to perform TF-IDF on (training data).
        
        Ouput:
        a. Dataframe with cosine similarities between TF-IDF doc vectors.
        
        """
        
        training_data = self.df[column]
        
        if self.pp_object:
            training_data = self.pp_object.preprocess(training_data)

        self.vectorizer = TfidfVectorizer()
        self.training_vectors = self.vectorizer.fit_transform(training_data)

        similarity = cosine_similarity(self.training_vectors)

        self.df_similarity = pd.DataFrame(similarity, columns=self.df.ID.values, index=self.df.ID.values)
        
    
        return self.df_similarity


    def predict_old(self, ID, N=10):

        """
        Finds top N most similar phrases using cosine similarity.

        ----
        Input:
        a. ID: existing phrase (from column 'texto') or ID.
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """

        # check if input is ID or phrase
        if ID not in self.df['ID'].values:
            ID = int(self.df[self.df['texto']==ID].ID)

        original_phrase = self.df[self.df['ID']==ID].texto.values

        index_and_scores = self.df_similarity[ID].nlargest(N+1)[1:]

        top_scores = index_and_scores.values

        top_indexes = index_and_scores.index

        top_texts = self.df[self.df['ID'].isin(top_indexes)].texto.tolist()

        print(f'Comparing:\n{original_phrase}\n')

        simil_dict = {}

        for i in range(N):

            # create dict
            simil_dict[top_indexes[i]] = [top_texts[i], top_scores[i]]

            print(f'----\n{i}.\n{top_texts[i]}\nScore: {top_scores.tolist()[i]}')

        return simil_dict
    
    
    def predict_new(self, new_document, N=10):
        
    
        """

        Finds top N most similar phrases for a new phrase given the training data using cosine similarity.

        ----
        Input:
        b. new_document: str
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """
        
        nd = new_document
        
        original_documents = self.df['texto'].tolist()
        
        if self.pp_object:
            new_document = self.pp_object.preprocess(new_document)

        # Use trained TF-IDF
        new_vector = self.vectorizer.transform(new_document)

        similarities = cosine_similarity(new_vector,  self.training_vectors)

        # Sort similarities in descending order + return top N most similar doc
        top_n = similarities.argsort()[:, -N:][0][::-1]


        print(f'Comparing:\n{nd}\n')

        simil_dict = {}

        for i in range(N):

            ID = self.df['ID'].tolist()[top_n[i]]

            text = original_documents[top_n[i]]

            score = similarities[0][top_n[i]]

            # Create dict
            simil_dict[ID] = [text, score]

            print(f'----\n{i}.\n{text}\nScore: {score}')

        return simil_dict
    
    

"""
    7. Semantic Similarity: LSA
    
"""


class LSA_Model():

    """
        Create a TF-IDF matrix. 
        Find top N most similars, for new or existing (in df) documents using LSA.
        
        -----
        Input: 
        a. df: pandas DataFrame with at least this columns: 
            - ID (int): to identify the document
            - data_column (str)
        
        b. Preprocess object
        
        ----

    """
    
    def __init__(self, df, pp_object = None):
        self.df = df
        self.pp_object = pp_object
        

    def train(self, data_column = 'texto', n_components=5):

        """
        Necessary for predict_old and predict_new.

        ----
        Input:
        a. data_column: column to perform TF-IDF on (training data).
        b. n_components: LSA number of topics.

        """
        self.column = data_column
        
        self.training_data = self.df[self.column].tolist()
        
        if self.pp_object:
            preprocessed_corpus = self.pp_object.preprocess(self.training_data)
        else:
            preprocessed_corpus = self.training_data

        self.vectorizer = TfidfVectorizer()
        X_tfidf = self.vectorizer.fit_transform(preprocessed_corpus)

        self.lsa = TruncatedSVD(n_components=n_components)
        self.X_lsa = self.lsa.fit_transform(X_tfidf)
        

    def predict_old(self, ID, N=10):

        """
        Finds top N most similar phrases using cosine similarity.

        ----
        Input:
        a. ID: existing phrase (from column 'texto') or ID (from column 'ID').
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """

        # check if input is ID or phrase
        if ID not in self.df['ID'].values:
            ID = int(self.df[self.df[self.column]==ID].ID)

        original_phrase = self.df[self.df['ID']==ID].texto.values
        
        if self.pp_object:
            preprocessed_phrase = self.pp_object.preprocess(original_phrase)
        else:
            preprocessed_phrase = original_phrase


        input_vector = self.vectorizer.transform(preprocessed_phrase)

        similarity_scores = cosine_similarity(self.lsa.transform(input_vector), self.X_lsa)[0]

        sorted_indexes = similarity_scores.argsort()[::-1]

        result = [value for index, value in enumerate(self.training_data) if index in sorted_indexes[:N]]

        simil_dict = {}

        print(f'Comparing:\n{original_phrase}\n')

        for i,x in enumerate(sorted_indexes[:N]):

            original_ID = self.df['ID'].tolist()[x]
            original_text = self.training_data[x]
            score = similarity_scores[x]

            simil_dict[original_ID] = [original_text, score]

            print(f'----\n{i}.\n{original_text}\nScore: {score}')

        return simil_dict 
    
    
    def predict_new(self, new_document, N=10):

        """
        Finds top N most similar phrases using cosine similarity.

        ----
        Input:
        a. new_document (str): new phrase
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """
        
        if self.pp_object:
            pp_data = self.pp_object.preprocess(new_document)
        else:
            pp_data = new_document

        input_vector = self.vectorizer.transform(pp_data)

        similarity_scores = cosine_similarity(self.lsa.transform(input_vector), self.X_lsa)[0]

        sorted_indexes = similarity_scores.argsort()[::-1]

        result = [value for index, value in enumerate(self.training_data) if index in sorted_indexes[:N]]

        simil_dict = {}

        print(f'Comparing:\n{new_document}\n')

        for i,x in enumerate(sorted_indexes[:N]):

            original_ID = self.df['ID'].tolist()[x]
            original_text = self.training_data[x]
            score = similarity_scores[x]

            simil_dict[original_ID] = [original_text, score]

            print(f'----\n{i}.\n{original_text}\nScore: {score}')


        return simil_dict
    
    
"""
    8. Semantic Similarity: LDA
"""

class LDA_Model():

    """
        Create a TF-IDF matrix. 
        Find top N most similars, for new or existing (in df) documents using LDA.
        
        -----
        Input: 
        a. df: pandas DataFrame with at least this columns: 
            - ID (int): to identify the document
            - data_column (str)
        
        b. Preprocess object
        
        ----

    """
    
    def __init__(self, df, pp_object = None):
        self.df = df
        self.pp_object = pp_object
        

    def train(self, data_column = 'texto', n_components=5):

        """
        Necessary for predict_old and predict_new.

        ----
        Input:
        a. data_column: column to perform TF-IDF on (training data).
        b. n_components: LDA number of topics.

        """
        self.column = data_column
        
        self.training_data = self.df[self.column].tolist()
        
        if self.pp_object:
            preprocessed_corpus = self.pp_object.preprocess(self.training_data)
        else:
            preprocessed_corpus = self.training_data
            
        self.vectorizer = TfidfVectorizer()
        X_tfidf = self.vectorizer.fit_transform(preprocessed_corpus)

        self.lda = LatentDirichletAllocation(n_components=n_components)
        self.X_lda = self.lda.fit_transform(X_tfidf)
        

    def predict_old(self, ID, N=10):

        """
        Finds top N most similar phrases using cosine similarity.

        ----
        Input:
        a. ID: existing phrase (from column 'texto') or ID (from column 'ID').
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """

        # check if input is ID or phrase
        if ID not in self.df['ID'].values:
            ID = int(self.df[self.df[self.column]==ID].ID)

        original_phrase = self.df[self.df['ID']==ID].texto.values
        
        if self.pp_object:
            preprocessed_phrase = self.pp_object.preprocess(original_phrase)
        else:
            preprocessed_phrase = original_phrase

        input_vector = self.vectorizer.transform(preprocessed_phrase)

        similarity_scores = cosine_similarity(self.lda.transform(input_vector), self.X_lda)[0]

        sorted_indexes = similarity_scores.argsort()[::-1]

        result = [value for index, value in enumerate(self.training_data) if index in sorted_indexes[:N]]

        simil_dict = {}

        print(f'Comparing:\n{original_phrase}\n')

        for i,x in enumerate(sorted_indexes[:N]):

            original_ID = self.df['ID'].tolist()[x]
            original_text = self.training_data[x]
            score = similarity_scores[x]

            simil_dict[original_ID] = [original_text, score]

            print(f'----\n{i}.\n{original_text}\nScore: {score}')

        return simil_dict 
    
    
    def predict_new(self, new_document, N=10):

        """
        Finds top N most similar phrases using cosine similarity.

        ----
        Input:
        a. new_document (str): new phrase
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """
        
        if self.pp_object:
            pp_data = self.pp_object.preprocess(new_document)
        else:
            pp_data = new_document

        input_vector = self.vectorizer.transform(pp_data)

        similarity_scores = cosine_similarity(self.lda.transform(input_vector), self.X_lda)[0]

        sorted_indexes = similarity_scores.argsort()[::-1]

        result = [value for index, value in enumerate(self.training_data) if index in sorted_indexes[:N]]

        simil_dict = {}

        print(f'Comparing:\n{new_document}\n')

        for i,x in enumerate(sorted_indexes[:N]):

            original_ID = self.df['ID'].tolist()[x]
            original_text = self.training_data[x]
            score = similarity_scores[x]

            simil_dict[original_ID] = [original_text, score]

            print(f'----\n{i}.\n{original_text}\nScore: {score}')


        return simil_dict
    
    
"""
    9. Semantic Similarity: LDA
"""

import itertools

import itertools

class Spacy_Model():

    """
        Find top N most similars, for new or existing (in df) documents using Spacy's es_core_news_lg model.
        
        -----
        Input: 
        a. df: pandas DataFrame with at least this columns: 
            - ID (int): to identify the document
            - data_column (str)
        
        b. Preprocess object
        
        ----

    """
    
    def __init__(self, df, pp_object = None):
        self.df = df
        self.pp_object = pp_object
        self.nlp = spacy.load('es_core_news_lg')
        self.trained = False
        
        
    def preprocess_training_data(self, data_column = 'texto'):
        
        self.training_data = self.df[data_column].tolist()
        
        if self.pp_object:
            self.preprocessed_corpus = self.pp_object.preprocess(self.training_data)
        else:
            self.preprocessed_corpus = self.training_data
        
        
    def train(self, data_column = 'texto'):

        """
        Only necessary for predict_old_trained (faster than predict_old_non_trained).

        ----
        Input:
        a. data_column: column to train on (that is, calculate similarities).
        
        ----
        Output:
        a. df_simil: matrix with similarities between every possible pair of documents.

        """
        self.trained = True
        
        self.column = data_column
        
        self.preprocess_training_data(self.column)
        
        # Create a list of tuples containing all possible pairs of strings and ID's
        doc_pairs = list(itertools.combinations(self.preprocessed_corpus, 2))
        
        # Calculate similarity for each pair
        similarities = []
        for i, pair in enumerate(doc_pairs):
            doc1 = self.nlp(pair[0])
            doc2 = self.nlp(pair[1])
            sim = doc1.similarity(doc2)
            similarities.append(sim)
            if i%15000==0:print(f'Training {round(len(similarities)/len(doc_pairs)*100)}%\n')
                
        # Construct dataframe
        self.IDs = self.df['ID'].tolist()
        
        self.df_simil = pd.DataFrame(index=self.IDs, columns=self.IDs)
        
        ID_pairs = list(itertools.combinations(self.IDs, 2))

        # Fill the dataframe with the calculated similarities
        for i, pair in enumerate(ID_pairs):
            self.df_simil.loc[pair[0], pair[1]] = similarities[i]
            self.df_simil.loc[pair[1], pair[0]] = similarities[i]
        self.df_simil.fillna(1.0, inplace=True)
    
        return self.df_simil
    
    
    def predict_old_trained(self, ID, N=10):

        """
        Finds top N most similar existing phrases with trained model.

        ----
        Input:
        a. ID: existing phrase (from column 'texto') or ID (from column 'ID').
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """

        # check if input is ID or phrase
        if ID not in self.df['ID'].values:
            ID = int(self.df[self.df[self.column]==ID].ID)

        original_phrase = self.df[self.df['ID']==ID].texto.values
        
        if self.pp_object:
            preprocessed_phrase = self.pp_object.preprocess(original_phrase)
        else:
            preprocessed_phrase = original_phrase

        
        index_and_scores = self.df_simil[ID].nlargest(N+1)[1:]

        top_scores = index_and_scores.values

        top_indexes = index_and_scores.index

        top_texts = self.df[self.df['ID'].isin(top_indexes)].texto.tolist()

        print(f'Comparing:\n{original_phrase}\n')

        simil_dict = {}

        for i in range(N):

            # create dict
            simil_dict[top_indexes[i]] = [top_texts[i], top_scores[i]]

            print(f'----\n{i}.\n{top_texts[i]}\nScore: {top_scores.tolist()[i]}')

        return simil_dict 
    
    
    def predict_old_non_trained(self, ID, data_column = 'texto', N=10):

        """
        Finds top N most similar existing phrases with non trained model.

        ----
        Input:
        a. ID: existing phrase (from column 'texto') or ID (from column 'ID').
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """
        
        self.column = data_column
        
        # check if input is ID or phrase
        if ID not in self.df['ID'].values:
            ID = int(self.df[self.df[self.column]==ID].ID)
            
        original_phrase = self.df[self.df['ID']==ID].texto.values
        
        
        if self.pp_object:
            preprocessed_phrase = self.pp_object.preprocess(original_phrase)[0]
        else:
            preprocessed_phrase = original_phrase

        preprocessed_phrase_nlp = self.nlp(preprocessed_phrase)
        
        best_simil = {}
        for i, doc in enumerate(self.preprocessed_corpus):
            doc_nlp = self.nlp(doc)
            original_phrase = self.df['texto'].tolist()[i]
            best_simil[original_phrase] = preprocessed_phrase_nlp.similarity(doc_nlp)
        
        sorted_best_simil = sorted(best_simil.items(), key=lambda x: x[1], reverse=True)

        print(f'Comparing:\n{original_phrase}\n')
        
        simil_dict = {}

        for i, (phrase, score) in enumerate(sorted_best_simil[:N]):
            
            original_ID = self.df['ID'].tolist()[i]

            simil_dict[original_ID] = [phrase, score]

            print(f'----\n{i}.\n{phrase}\nScore: {score}')


        return simil_dict
    
    
    def predict_old(self, ID, data_column = 'texto', N=10):
        
        if self.trained:
            print('Predicting with fast model.')
            self.predict_old_trained(ID = ID, N=N)
            
        else:
            print('Predicting with slow model.')
            self.preprocess_training_data(data_column)
            self.predict_old_non_trained(ID = ID, data_column = data_column, N=N)
            
    
    def predict_new(self, new_document, N=10):

        """
        Finds top N most similar phrases.

        ----
        Input:
        a. new_document (str): new phrase
        b. N: int (Top N)

        ------
        Ouput:
        a. Dict of N most similar phrases, with ID as keys and texts + score as values.

        """
        
        if self.pp_object:
            pp_data = self.pp_object.preprocess(new_document)
        else:
            pp_data = new_document

        new_document_nlp = self.nlp(new_document)

        best_simil = {}
        for i, doc in enumerate(self.preprocessed_corpus):
            doc_nlp = self.nlp(doc)
            original_phrase = self.df['texto'].tolist()[i]
            best_simil[original_phrase] = new_document_nlp.similarity(doc_nlp)
        
        sorted_best_simil = sorted(best_simil.items(), key=lambda x: x[1], reverse=True)

        print(f'Comparing:\n{new_document}\n')
        
        simil_dict = {}

        for i, (phrase, score) in enumerate(sorted_best_simil[:N]):
            
            original_ID = self.df['ID'].tolist()[i]

            simil_dict[original_ID] = [phrase, score]

            print(f'----\n{i}.\n{phrase}\nScore: {score}')


        return simil_dict