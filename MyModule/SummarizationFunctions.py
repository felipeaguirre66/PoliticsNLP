from MyModule.GeneralFunctions import *

import warnings

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



class T5Summary():
   
   """
   Generate abstractive summaries utilizing T5 model.
   ---------
   Input: 
      Text: str
   
   Output: 
      Summarized text: str
   """
   
   def __init__(self):
      t5_ckpt = 'josmunpen/mt5-small-spanish-summarization'
      print(f'ckpt:\n{t5_ckpt}')
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.tokenizer = AutoTokenizer.from_pretrained(t5_ckpt)
      self.model = AutoModelForSeq2SeqLM.from_pretrained(t5_ckpt)

   
   def generate_summary(self, text, max_new_tokens=1000, num_beams=10, num_return_sequences=2):
      
      self.text=text
      self.max_new_tokens=max_new_tokens
      self.num_beams=num_beams
      self.num_return_sequences=num_return_sequences
      
      inputs = self.tokenizer([self.text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
      input_ids = inputs.input_ids.to(self.device)
      attention_mask = inputs.attention_mask.to(self.device)
      output = self.model.generate(input_ids,
                              max_new_tokens=self.max_new_tokens, #cantidad maxima de palabras en la respuesta.
                              num_beams = self.num_beams, #grado en el que explora otras alternativas antes de converger 
                                                      #(mas grande = mejor resultado y más tiempo de ejecuciuon)
                              do_sample = True, #enhances performance but increases runtime
                              num_return_sequences=self.num_return_sequences, #number of summarizations to return              
                              attention_mask=attention_mask)
      summarizations = []
      for res in output:
         summary = self.tokenizer.decode(res, skip_special_tokens=True)
         summarizations.append(summary)
         
      return summarizations


from transformers import BertTokenizerFast, EncoderDecoderModel

class BETOSummary():
   
   """
   Generate abstractive summaries utilizing BETO model.
   ---------
   Input: 
      Text: str
   
   Output: 
      Summarized text: str
   """
   
   def __init__(self):
      beto_ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
      print(f'ckpt:\n{beto_ckpt}')
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.tokenizer = BertTokenizerFast.from_pretrained(beto_ckpt)
      self.model = EncoderDecoderModel.from_pretrained(beto_ckpt)

   
   def generate_summary(self, text, max_new_tokens=1000, num_beams=10, num_return_sequences=2):
      
      self.text=text
      self.max_new_tokens=max_new_tokens
      self.num_beams=num_beams
      self.num_return_sequences=num_return_sequences
      
      inputs = self.tokenizer([self.text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
      input_ids = inputs.input_ids.to(self.device)
      attention_mask = inputs.attention_mask.to(self.device)
      output = self.model.generate(input_ids,
                                max_new_tokens=self.max_new_tokens, #cantidad maxima de palabras en la respuesta.
                                num_beams = self.num_beams, #grado en el que explora otras alternativas antes de converger 
                                                        #(mas grande = mejor resultado y más tiempo de ejecuciuon)
                                do_sample = True, #enhances performance but increases runtime
                                num_return_sequences=self.num_return_sequences, #number of summarizations to return              
                                attention_mask=attention_mask)
      summarizations = []
      for res in output:
         summary = self.tokenizer.decode(res, skip_special_tokens=True)
         summarizations.append(summary)
         
      return summarizations



from sklearn.cluster import KMeans
import hdbscan

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import numpy as np

class MostRepresentativeDocs():
    
    """
    Finds the most representative documents of a dataset as follows:
    
    1. Creates sentence embbeding for each text
    2. Finds clusters with K-means (K is an hyperparameter)
    3. Creates a mean vector of each cluster
    4. Utilices cosine similarity to return the most similar documents to each cluster's mean vector.
    
    """
    
    def __init__(self, cluster_algorithm='kmeans', n_pca=None, **kwargs):
        """
        Input: 
            cluster_algorithm: Kmeans('kmeans'), HDBScan ('hdbscan')
            
        """
        self.model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
        self.cluster_algorithm = cluster_algorithm
        self.n_pca = n_pca
        self.kwargs = kwargs if kwargs is not None else None
        
        self.found_clusters = False
        
    def preprocess_and_encode(self):
        if self.pp_object:
            self.documents = self.pp_object.preprocess(self.documents)
        
        self.emb_docs = self.model.encode(self.documents)
        
        if self.n_pca:
            self.n_pca = min(self.n_pca, self.emb_docs.shape[0], self.emb_docs.shape[1])
            pca = PCA(n_components=self.n_pca)
            self.emb_docs = pca.fit_transform(self.emb_docs)
            print(f'Reducing embedding dimensions to {self.emb_docs.shape[-1]}')
        
    def fit_clustering_model(self):  

        if self.cluster_algorithm == 'kmeans':
            self.cluster_model = KMeans(
                init="k-means++",
                n_clusters=self.n_clusters,
                n_init=100,
                max_iter=1000,
                **self.kwargs)
            
        elif self.cluster_algorithm == 'hdbscan':
              self.cluster_model = hdbscan.HDBSCAN(**self.kwargs)
        
        self.cluster_model.fit(self.emb_docs)
    
    def label_docs_emb(self):
        labeled_embs = {}
        for i in range(len(self.emb_docs)):
            current_label = self.cluster_model.labels_[i]
            if current_label in labeled_embs.keys():
                labeled_embs[current_label].append(self.emb_docs[i])
            else:
                labeled_embs[current_label] = [self.emb_docs[i]]
        
        self.labeled_embs = labeled_embs
        
    def label_original_docs(self):
        labeled_original_docs = {}
        for i, kmeans_label in enumerate(self.cluster_model.labels_):
            if kmeans_label in labeled_original_docs.keys():
                labeled_original_docs[kmeans_label].append(self.original_documents[i])
            else:
                labeled_original_docs[kmeans_label] = [self.original_documents[i]]
                
        self.labeled_original_docs = labeled_original_docs
    
    def find_labels_means(self):
        self.label_means = {}
        for label, vectors in self.labeled_embs.items():
            self.label_means[label] = sum(vectors)  / len(vectors)
        
    def find_representatives(self):
        best_doc_per_label = {}

        for label_iter, label_mean in self.label_means.items():

            sentences = self.labeled_embs[label_iter]

            best_simil = {}
            for emb_doc_labeled in sentences:
                
                index = self.emb_docs.tolist().index(emb_doc_labeled.tolist())
                phrase = self.original_documents[index]#.tolist()[index]
                
                sim = cosine_similarity(emb_doc_labeled.reshape(1, -1), label_mean.reshape(1, -1))[0][0]
                best_simil[phrase] = sim

            sorted_best_simil = sorted(best_simil.items(), key=lambda x: x[1], reverse=True)
            
            best_doc_per_label[label_iter] = sorted_best_simil
            
            self.best_doc_per_label = best_doc_per_label
        
        return self.best_doc_per_label
    
    def elbow_method(self, documents, k_range=[1, 10], pp_object=None):
        
        """
        Plot elbow method (sum of squared distance for each K) to find optimum K for K means
        -----------------------------
        Input:
            documents: list of str
            k_range: k range to try
            pp_object: pre process object
        
        """
        
        self.pp_object = pp_object
        self.documents = documents
        
        self.preprocess_and_encode()
        
        list_k = list(range(k_range[0],k_range[1]))
        sse = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in list_k:
                km = KMeans(n_clusters=k)
                km.fit(self.emb_docs)
                sse.append(km.inertia_)
            
        plt.figure(figsize=(5, 5))
        plt.plot(list_k, sse, '-o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('SSD between each point and its centroid')
    
    
    def get_average_silhouette_score(self):
        return silhouette_score(self.emb_docs, self.cluster_model.labels_)
    
    def get_cluster_silhouette_score(self, cluster_index=0):
    
        cluster_mask = (self.cluster_model.labels_ == cluster_index)  # boolean mask for selecting points in the cluster

        # calculate the Silhouette score for each point in the data
        silhouette_scores = silhouette_samples(self.emb_docs, self.cluster_model.labels_)

        # select the Silhouette scores for the points in the cluster
        cluster_scores = silhouette_scores[cluster_mask]

        # compute the average Silhouette score for the cluster
        return cluster_scores.mean()
    
    def plot_elbow_silhouette_score(self, range_n_clusters=[2,20]):
        
        range_n_clusters = range(range_n_clusters[0],range_n_clusters[1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            silhouette_avg = []
            for num_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(self.emb_docs)
                cluster_labels = kmeans.labels_
                silhouette_avg.append(silhouette_score(self.emb_docs, cluster_labels))
        
        # add a label for the maximum X value
        max_index = np.argmax(silhouette_avg)
        max_x = range_n_clusters[max_index]
        plt.axvline(x=max_x, color='r', linestyle='--')
        plt.text(max_x, 1.1*np.max(silhouette_avg), f'Maximum X value: {max_x:.2f}', ha='center')
           
        plt.plot(range_n_clusters,silhouette_avg,'bx-')    
        plt.xlabel('Values of K') 
        plt.ylabel('Silhouette score') 
        plt.title('Silhouette analysis For Optimal k')
        plt.show()
    
    def elbow_pca_explained_variance(self, documents, pp_object=None):
        
        self.pp_object = pp_object
        self.documents = documents
        
        self.preprocess_and_encode()
        
        range_n_dimensions = range(2, min(self.emb_docs.shape[0], self.emb_docs.shape[1]))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explained_var = []
            for num_d in range_n_dimensions:
                pca = PCA(n_components=num_d)
                pca.fit(self.emb_docs)
                explained_var.append(sum(pca.explained_variance_ratio_))
        
        # add a label for the maximum X value    
        plt.plot(range_n_dimensions, explained_var,'bx-')    
        plt.xlabel('Number of dimensions') 
        plt.ylabel('Explained variance percentage') 
        plt.title('PCA Explained variance')
        plt.show()
   
    def visualize_documents_kmeans(self, documents, n_clusters, pp_object=None):
        
        """
        Visualize document distribution with it's kmens cluster assigment
        
        Input:
            documents: list of str
            n_clusters: number of clusters to find
        
        Output: 
            2D visualization
        
        WARNING: true distribution might be different given that PCA is applied, encoding 768 dimension into just 2.
        """
        self.n_clusters = n_clusters
        self.pp_object = pp_object
        self.documents = documents
        
        self.preprocess_and_encode()
        
        pca = PCA(n_components=2)
        self.emb_docs = pca.fit_transform(self.emb_docs)
        
        self.fit_clustering_model()
        self.label_docs_emb()
        
        color_map = plt.cm.get_cmap('tab10', len(self.labeled_embs))

        legend_labels = []
        for i, (key, pca_data) in enumerate(self.labeled_embs.items()):
            color = color_map(i)
            x = [d[0] for d in pca_data]
            y = [d[1] for d in pca_data]
            plt.scatter(x, y, color=color, label=key)
            legend_labels.append(f"Cluster{key}: {len(pca_data)} documents")
        pca1, pca2 = pca.explained_variance_ratio_
        plt.title('Documents distribution.')
        plt.xlabel(f'Explained variance: {round(pca1*100)}%')
        plt.ylabel(f'Explained variance: {round(pca2*100)}%')
        plt.legend(legend_labels, loc='best', bbox_to_anchor=(1, 0.5))
        plt.show()
        
    def plot_word_counts(self, documents, n_clusters, pp_object_transformers=None, pp_object_word_count=None):
        """
        Plot word frequencia by each desafio's cluster.
        -----------------------------------------------
        Input:
            Documents: list of str
            n_clusters: to find KMeans
            pp_object_transformers: preprocess docs before embedding
            pp_object_word_count: preprocess docs before word count
        """ 
        
        # Find clusters ONLY if not allready done (so clusters are the same across all model functions)
        if not self.found_clusters:
            result_dict = self.get_representatives(documents=documents, n_clusters=n_clusters, pp_object=pp_object_transformers)
        else:
            result_dict = self.best_doc_per_label
            
        for key_clus, value_clus in result_dict.items():
            only_text = [v[0] for v in value_clus]
            pp_only_text = pp_object_word_count.preprocess(' '.join(only_text))[0]
            words_desafio = pp_only_text.split(' ')
            elements, frequencies = count_words(words_desafio)
            plot_word(elements, frequencies, plot_title = f'Cluster {key_clus} (N={len(only_text)}).')
    
    def cluster_and_label_original_docs(self, documents, n_clusters=1, pp_object=None):
        """
        Input:
            documents: list of str
            n_clusters: to find
        
        Output: 
            Dictionary with cluster label (keys) and original document (values)
        """
        self.documents = documents
        self.original_documents = documents
        self.n_clusters = n_clusters
        self.pp_object = pp_object

        self.preprocess_and_encode()
        self.fit_clustering_model()
        self.label_original_docs()
        
        return self.labeled_original_docs
        
    def get_representatives(self, documents, n_clusters=1, pp_object=None):
        """
        Input:
            -documents: list of str
            -pp_object: preprocessing object
            -n_clusters = number of clusters to find representative docs on (if n_clusters=1, it gets the most rep doc for all the documents as a group)
        
        Output: 
            -dictionary with each K-mean's cluster label as keys and each document with its similarity coefficient to the cluster's mean (descending).
        """
       
        self.documents = documents
        self.original_documents = documents
        self.n_clusters = n_clusters
        self.pp_object = pp_object
        
        self.found_clusters = True
        

        self.preprocess_and_encode()
        self.fit_clustering_model()
        self.label_docs_emb()
        self.find_labels_means()
        return self.find_representatives()
       