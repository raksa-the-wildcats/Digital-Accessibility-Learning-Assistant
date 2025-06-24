"""
Simple vector store that doesn't rely on ChromaDB for deployment compatibility.
Uses basic similarity search with sentence-transformers.
"""
import os
import json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embeddings_model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def add_documents(self, chunks: List[Dict[str, str]]):
        """Add document chunks to the vector store."""
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Adding {len(chunks)} chunks to simple vector store...")
        
        # Extract texts
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        new_embeddings = self.embeddings_model.encode(texts)
        
        # Store everything
        self.documents.extend(texts)
        self.embeddings.extend(new_embeddings)
        self.metadata.extend([
            {'source': chunk['source'], 'chunk_id': chunk['chunk_id']} 
            for chunk in chunks
        ])
        
        print(f"Successfully added {len(chunks)} chunks to simple vector store")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'distance': 1 - similarities[idx]  # Convert similarity to distance
            })
        
        return results
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return len(self.documents)
    
    def reset_collection(self):
        """Reset the collection (delete all documents)."""
        self.documents = []
        self.embeddings = []
        self.metadata = []
        print("Reset simple vector store collection")