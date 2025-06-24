import os
from langchain.prompts import PromptTemplate
from typing import List, Dict
from config import Config
from .vector_store import VectorStore

# Try to import openai, but fall back to requests
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai module not available. Will use requests as fallback")

import requests
import json

class QAChain:
    def __init__(self):
        # Clear proxy environment variables that might interfere
        import os
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Initialize OpenAI client
        self.openai_client = None
        self.use_requests_fallback = False
        
        if Config.OPENAI_API_KEY:
            if OPENAI_AVAILABLE:
                try:
                    self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                    print("OpenAI client initialized successfully")
                except Exception as e:
                    print(f"Error initializing OpenAI client: {e}")
                    print("Falling back to requests-based API calls")
                    self.use_requests_fallback = True
            else:
                print("Using requests-based OpenAI API calls")
                self.use_requests_fallback = True
        
        # Check if dependencies are available
        if not OPENAI_AVAILABLE:
            print("Warning: openai module not available. Install with: pip install openai")
        
        try:
            self.vector_store = VectorStore()
        except ImportError as e:
            print(f"Warning: {str(e)}")
            self.vector_store = None
            
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for Q&A."""
        template = """You are an expert digital accessibility instructor helping education students understand accessibility and inclusive design. Your students are future teachers, administrators, and education professionals who need to create accessible content and advocate for students with disabilities.

Context from accessibility documentation:
{context}

Student Question: {question}

Instructions for your response:
1. **Use education-friendly language** - Explain concepts clearly without assuming technical background
2. **Connect to educational settings** - Relate accessibility to classroom materials, online learning, and student needs
3. **Explain the "why"** - Help students understand how accessibility barriers affect learners with disabilities
4. **Provide practical examples** - Give real classroom scenarios and educational content examples
5. **Focus on universal design** - Show how accessible design benefits all students, not just those with disabilities
6. **Include actionable guidance** - Offer specific steps they can take in their future educational roles
7. **Reference standards when relevant** - Mention WCAG, ADA, or Section 508 in educational contexts

Answer as a supportive instructor preparing future educators to be accessibility advocates who create inclusive learning environments. Emphasize that accessibility is about equity and ensuring all students can fully participate in education."""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def get_answer(self, question: str) -> Dict[str, any]:
        """Get an answer to a question using the knowledge base."""
        try:
            # Check if OpenAI is available (either client or requests fallback)
            if not Config.OPENAI_API_KEY:
                return {
                    "answer": "❌ OpenAI API key not set. Please set OPENAI_API_KEY environment variable",
                    "sources": [],
                    "error": "API key not set"
                }
            
            if not self.openai_client and not self.use_requests_fallback:
                return {
                    "answer": "❌ The OpenAI API is not available. Please install the required dependency: pip install openai and set OPENAI_API_KEY",
                    "sources": [],
                    "error": "openai module not available or API key not set"
                }
            
            if not self.vector_store:
                return {
                    "answer": "❌ Vector store is not available. Please install the required dependency: pip install sentence-transformers",
                    "sources": [],
                    "error": "vector store not available"
                }
            
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question)
            
            if not docs:
                return {
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "error": None
                }
            
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Document: {doc['metadata'].get('source', 'Unknown')}\n{doc['content']}"
                for doc in docs
            ])
            
            # Generate answer using OpenAI
            prompt = self.prompt_template.format(context=context, question=question)
            
            if self.openai_client:
                # Use official OpenAI client
                response = self.openai_client.chat.completions.create(
                    model=Config.CHAT_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content.strip()
            else:
                # Use requests fallback
                headers = {
                    "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": Config.CHAT_MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result['choices'][0]['message']['content'].strip()
                else:
                    return {
                        "answer": f"❌ OpenAI API error: {response.status_code} - {response.text}",
                        "sources": [],
                        "error": f"API error: {response.status_code}"
                    }
            
            # Prepare sources
            sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in docs]))
            
            return {
                "answer": answer,
                "sources": sources,
                "error": None
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def initialize_knowledge_base(self, force_rebuild: bool = False):
        """Initialize the knowledge base from PDFs."""
        try:
            if not self.vector_store:
                print("❌ Vector store is not available. Please install sentence-transformers")
                return False
            
            # Check if collection already has documents
            if not force_rebuild and self.vector_store.get_collection_count() > 0:
                print(f"Knowledge base already contains {self.vector_store.get_collection_count()} documents")
                return True
            
            # Import here to avoid circular imports
            from .pdf_processor import PDFProcessor
            
            # Process PDFs
            processor = PDFProcessor()
            chunks = processor.process_all_pdfs()
            
            if not chunks:
                print("No chunks created from PDFs")
                return False
            
            # Reset collection if force rebuild
            if force_rebuild:
                self.vector_store.reset_collection()
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            print(f"Knowledge base initialized with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error initializing knowledge base: {str(e)}")
            return False