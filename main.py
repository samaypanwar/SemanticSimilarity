import os
import sys
import pickle
from datetime import datetime
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.random_projection import gaussian_random_matrix
import pandas as pd
import annoy
import warnings


if not sys.warnoptions:
    warnings.simplefilter("ignore")

class Semantic:
    
    def __init__(self):
        """
        This init function initialises the tensorflow.hub module that we are going to use
        to generate all our embeddings
        """
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.module = hub.load(self.module_url)
        
        
    def generate_embeddings(self,text):
        return self.module([text]).numpy()
    
    def search_companyName(self,listofDescriptions):
        
        """"
        Since the queried result is a list of descriptions and not company names,
        this function locates the company name that matches the resulant description.
        
        It requires that we have the companyNames.csv file located in the current working directory
        """
    
        companyName = pd.read_csv('companyNames.csv',index_col='BusinessDesc')
        listofNames = []

        for name in listofDescriptions:
            try:
                listofNames.append(companyName.loc[name,'companyName'])
            except:
                continue

        return listofNames
    
    
    
    class cosine_similarity:
        """
        This function uses cosine similiarity to query results.
        """
        
        def __init__(self):
            
            self.super_embeddings = pickle.load(open('Business_embeddings.pkl','rb'))
            data = pd.read_csv(r'companyNames.csv',index_col='companyName')
            self.corpus = data.BusinessDesc.to_list()
            self.semantic = Semantic()
        
        
        def query(self,query,topN=5,verbose=False):
            """"
            This function searches for the relevant results using cosine similarity, it requires the query,
            and the number of closest matches you want (topN). If verbose=True, it prints the queried results.
            It requires that we have the companyNames.csv file located in the current working directory.
            """
            
            embed_query = self.semantic.generate_embeddings(query)
            sim = cosine_similarity(embed_query,self.super_embeddings)

            bestN = list(sim.argsort()[0,-topN:])
            bestN.reverse()

            listofDescriptions = []
            for el in bestN :
                listofDescriptions.append(self.corpus[el]) #uses the pandas dataframe 'corpus' to 
                                                      #locate the appropriate description matching the resultant indices

            names = self.semantic.search_companyName(listofDescriptions)
            if verbose:
                print("Results = ")
                print("---------------------------------------")
                for name in names:
                    print(name)
                print("---------------------------------------")

            return names
        
        
        
    class annoy_similarity:
        """"
        This function uses the annoy method for querying results. The default metric
        to be used is the manhattan distance which gives us the best accuracy.
        """
        
        def __init__(self,metric='manhattan',verbose=False):
            
            #self.metric = metric
            #self.verbose = verbose
            original_dim = 512
            self.embedding_dimension = original_dim  #projected_dim incase a random_projection_matrix is used
            index_filename = "indices/" + metric +"_index"  #CWD/indices/metric_index
            
            self.index = annoy.AnnoyIndex(self.embedding_dimension,metric=metric)
            self.index.load(index_filename)
            
            if verbose:
                print('Annoy index is loaded.')
       
            
            with open(index_filename + '.mapping', 'rb') as handle:
                self.mapping = pickle.load(handle)
        
            if verbose:
                print('Mapping file is loaded.')
            
            self.semantic = Semantic()
            
            
        def find_similar_items(self, embedding, topN=5):
            """
            Finds similar items to a given embedding in the ANN index
            """
            ids = self.index.get_nns_by_vector(
            embedding, topN, search_k=-1, include_distances=False)
            items = [self.mapping[i] for i in ids]
            return items
        
        
        def query(self,query,topN = 5,verbose=False):
            """
            This function queries the resultant descriptions using the ANN algorithm.
            topN is the number of closest neighbours we want.
            If verbose=True, it prints out the results.
            """
            
            query_embedding = self.semantic.generate_embeddings(query)[0]
            items = self.find_similar_items(query_embedding,topN)
            names = self.semantic.search_companyName(items)

            if verbose:
                print("")
                print("Results:")

                print("---------------------")
                for name in names:
                    print(name)
                print("---------------------")

            return names
        
        
        def build_annoy_index():
            """
            This function builds the annoy index incase it hasn't been built already
            """
            randomStatement = True
            

        