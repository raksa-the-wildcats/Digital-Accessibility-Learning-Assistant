import os
from dotenv import load_dotenv

load_dotenv()

# Try to import streamlit for secrets, fallback to environment variables
try:
    import streamlit as st
    def get_openai_api_key():
        try:
            return st.secrets["OPENAI_API_KEY"]
        except:
            return os.getenv("OPENAI_API_KEY")
except ImportError:
    def get_openai_api_key():
        return os.getenv("OPENAI_API_KEY")

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = get_openai_api_key()
    
    # Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using sentence-transformers model
    CHAT_MODEL = "gpt-4o"
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "accessibility_docs"
    
    # PDF Processing Configuration
    PDF_DIRECTORY = "./data/pdfs"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration
    TOP_K_DOCUMENTS = 5
    
    # UI Configuration
    APP_TITLE = "Digital Accessibility Learning Assistant"
    APP_DESCRIPTION = "🎓 **For Education Students**: Learn about digital accessibility and inclusive design. Understand how to create accessible content and advocate for students with disabilities in educational settings!"
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True