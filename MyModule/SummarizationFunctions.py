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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class MostRepresentativeDocs():
    
    """
    Finds the most representative documents of a dataset as follows:
    
    1. Creates sentence embbeding for each text
    2. Finds clusters with K-means (K is an hyperparameter)
    3. Creates a mean vector of each cluster
    4. Utilices cosine similarity to return the most similar documents to each cluster's mean vector.
    
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
        labeled_embs = {}
        for i in range(len(self.emb_docs)):
            current_label = self.kmeans.labels_[i]
            if current_label in labeled_embs.keys():
                labeled_embs[current_label].append(self.emb_docs[i])
            else:
                labeled_embs[current_label] = [self.emb_docs[i]]
        
        self.labeled_embs = labeled_embs
    
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
                phrase = self.original_documents.tolist()[index]
                
                sim = cosine_similarity(emb_doc_labeled.reshape(1, -1), label_mean.reshape(1, -1))[0][0]
                best_simil[phrase] = sim

            sorted_best_simil = sorted(best_simil.items(), key=lambda x: x[1], reverse=True)
            
            best_doc_per_label[label_iter] = sorted_best_simil
        
        return best_doc_per_label
    
    def get_representatives(self, documents, n_clusters=1, pp_object=None):
        self.documents = documents
        self.original_documents = documents
        self.n_clusters = n_clusters
        self.pp_object = pp_object
        
        """
        Input:
            -documents: list of str
            -pp_object: preprocessing object
            -n_clusters = number of clusters to find representative docs on (if n_clusters=1, it gets the most rep doc for all the documents as a group)
        
        Output: 
            -dictionary with each K-mean's cluster label as keys and each document with its similarity coefficient to the cluster's mean (descending).
        """
        self.preprocess_and_encode()
        self.fit_kmeans()
        self.label_docs()
        self.find_labels_means()
        return self.find_representatives()
       