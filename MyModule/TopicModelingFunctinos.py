"""
    4. Topic Modeling functions.

"""
# Some imports
import pandas as pd
import regex as re
import matplotlib.pyplot as plt

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
    
    explained_var = pca.explained_variance_ratio_
    
    if des:
        plt.title(f'Des {des} Topic Distribution')
    else:
        plt.title('Topic Distribution')
    plt.xlabel(f'PCA 1 %{round(explained_var[0] * 100)}')
    plt.ylabel(f'PCA 2 %{round(explained_var[1] * 100)}')
    plt.show()
    
"""
Bertopic plot

"""    

from IPython.display import display, Markdown

def bertopic_plots(model, documents, des):
    
    """
    This function plots every relevant BERTopic plot. 
    
    Input:
    a. model: BERTopic
    b. documents: preprocessed
    c. des: name of desafio
    
    Output: none
    
    """
    
    
    topics, probs = model.fit_transform(documents)
    
    
    
    # Define the text to display
    text = f'<div style="text-align:center;"><span style="font-size:48px;color:blue;">Desafío {des}</span></div>'

    # Get the unique Topics and their frequencies
    unique_elements, counts = np.unique(topics, return_counts=True)
    
    plt.bar(unique_elements, counts)
    
    # Add axis labels and a title
    plt.xlabel('Topic')
    plt.ylabel('Frequency (N of documents)')
    plt.xticks(unique_elements)
    plt.title(f'Frequency of each Topic\nDes {des}')

    # Visualize topics
    fig_vt = model.visualize_topics()

    # Importance of each word
    fig_w = model.visualize_barchart(top_n_topics=len(model.get_topics()))
    
    # Vistualize documents
    fig_vd = model.visualize_documents(documents, hide_annotations=True)
    
    # Heatmap topics
    fig_hm = model.visualize_heatmap()

    #Show images
    print('\n\n')
    display(Markdown(text))
    
    plt.show()
    fig_vt.show()
    fig_w.show()
    fig_vd.show()
    fig_hm.show()
    
    try:
        # Most representative documents
        pp = Preprocess(lemma=False)
        original_texts = df['texto'].values.tolist()
        prepro_texts = pp.preprocess(df['texto'])
        prepro_texts = [' '.join(d) for d in prepro_texts]

        for key, value in model.get_representative_docs().items():
            print('\n'*4),print(f'TOPIC {key} most representative documents:\n')
            for i, doc in enumerate(value):
                indice = prepro_texts.index(doc)
                print(f'{i+1}.\n {original_texts[indice]}\n\n')
    except:
        pass