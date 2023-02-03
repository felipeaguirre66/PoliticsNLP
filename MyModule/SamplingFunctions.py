from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SampleRepresentatives():
    
    """
    Sample documents of a corpus by clustering them in N groups and selecting one of each cluster.
    
    1. Encode each document with Transformers
    2. Fit K-Means (k = n_clusters)
    3. Label each document with its cluster and return them
    
    """
    
    def __init__(self):
        self.model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
        
    def preprocess_and_encode(self):
        if self.pp_object:
            self.documents = self.pp_object.preprocess(self.documents)
        
        self.emb_docs = self.model.encode(self.documents)
        
    def fit_kmeans(self):   
        self.kmeans = KMeans(
            init="k-means++",
            n_clusters=self.n_clusters,
            n_init=10,
            max_iter=300)
        
        self.kmeans.fit(self.emb_docs)
   
    def label_docs(self):
        labeled_docs = {}
        for i in range(len(self.emb_docs)):
            current_label = self.kmeans.labels_[i]
            if current_label in labeled_docs.keys():
                labeled_docs[current_label].append(self.original_documents[i])
            else:
                labeled_docs[current_label] = [self.original_documents[i]]
        
        return labeled_docs

    def get_sample(self, documents, n_clusters=1, pp_object=None,):
        self.documents = documents
        self.original_documents = documents
        self.n_clusters = n_clusters
        self.pp_object = pp_object
        
        """
        Input:
            -documents: list of str
            -pp_object: preprocessing object
            -n_clusters = number of clusters to find (it would be reasonable to set n_clusters=number of samples you want displayed.)
        
        Output:
            -dictionary with each K-mean's cluster label as keys and each document corresponding to it as values.
        
        """
        self.preprocess_and_encode()
        self.fit_kmeans()
        return self.label_docs()