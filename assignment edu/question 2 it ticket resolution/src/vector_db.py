import faiss
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class TicketKnowledgeBase:
    def __init__(self, dimension: int):
        """
        Initializes an in-memory Vector Database using FAISS.
        This allows us to search millions of records in milliseconds.
        """
       
        self.index = faiss.IndexFlatL2(dimension) 
        self.metadata = pd.DataFrame() 
        
    def build_index(self, embeddings: np.ndarray, df: pd.DataFrame):
        """Populates the database with historical tickets."""
        logging.info(f"Adding {len(embeddings)} vectors to FAISS index...")
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = df
        
    def search(self, query_vector: np.ndarray, top_k: int = 3):
        """Finds the closest historical tickets to a new incoming ticket."""
        query_vector = np.array([query_vector]).astype('float32')
        
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1: 
                match = self.metadata.iloc[idx].to_dict()
                match['similarity_score'] = 1 / (1 + dist) 
                results.append(match)
                
        return results