import os
# Set environment variable for protobuf compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import chromadb
from typing import List, Dict
from config import Config

# Try to import sentence_transformers, but don't fail if it's not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers module not available. Install with: pip install sentence-transformers")

class VectorStore:
    def __init__(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers is required. Install with: pip install sentence-transformers")
        
        # Clean environment of any proxy settings
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Initialize ChromaDB client with fallback strategy
        self.client = None
        
        # Try different initialization methods
        try:
            # Method 1: Try PersistentClient
            os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
            self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
            print(f"Successfully created PersistentClient at {Config.CHROMA_DB_PATH}")
        except Exception as e:
            print(f"PersistentClient failed: {e}")
            try:
                # Method 2: Try in-memory client
                self.client = chromadb.Client()
                print("Using in-memory ChromaDB client")
            except Exception as e2:
                print(f"In-memory client failed: {e2}")
                try:
                    # Method 3: Try with basic settings
                    import chromadb.config
                    settings = chromadb.config.Settings(anonymized_telemetry=False)
                    self.client = chromadb.Client(settings)
                    print("Using ChromaDB client with basic settings")
                except Exception as e3:
                    print(f"All ChromaDB initialization methods failed: {e3}")
                    print("Falling back to simple vector store...")
                    from .simple_vector_store import SimpleVectorStore
                    self.client = SimpleVectorStore()
                    self.use_simple_store = True
        
        self.collection_name = Config.COLLECTION_NAME
        self.use_simple_store = getattr(self, 'use_simple_store', False)
        
        if not self.use_simple_store:
            self.embeddings = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.collection = None
            self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, chunks: List[Dict[str, str]]):
        """Add document chunks to the vector store."""
        if not chunks:
            print("No chunks to add")
            return
        
        if self.use_simple_store:
            # Use simple vector store
            self.client.add_documents(chunks)
        else:
            # Use ChromaDB
            print(f"Adding {len(chunks)} chunks to vector store...")
            
            # Extract texts and metadata
            texts = [chunk['content'] for chunk in chunks]
            metadatas = [{'source': chunk['source'], 'chunk_id': chunk['chunk_id']} for chunk in chunks]
            ids = [chunk['chunk_id'] for chunk in chunks]
            
            # Generate embeddings
            print("Generating embeddings...")
            embeddings = self.embeddings.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully added {len(chunks)} chunks to vector store")
    
    def similarity_search(self, query: str, k: int = None) -> List[Dict]:
        """Search for similar documents."""
        if k is None:
            k = Config.TOP_K_DOCUMENTS
        
        if self.use_simple_store:
            # Use simple vector store
            return self.client.similarity_search(query, k)
        else:
            # Use ChromaDB
            # Generate query embedding
            query_embedding = self.embeddings.encode([query])[0].tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Format results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    })
            
            return documents
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        if self.use_simple_store:
            return self.client.get_collection_count()
        else:
            return self.collection.count()
    
    def reset_collection(self):
        """Reset the collection (delete all documents)."""
        if self.use_simple_store:
            self.client.reset_collection()
        else:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"Reset collection: {self.collection_name}")