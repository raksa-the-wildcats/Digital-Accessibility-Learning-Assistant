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

**Response Length Guidelines:**
- For simple factual questions (shortcuts, specific values, definitions): Provide a brief, direct answer (1-2 sentences)
- For "how-to" questions: Give step-by-step instructions with brief context
- For conceptual questions: Provide moderate explanation with educational examples
- For complex topics: Give comprehensive guidance with practical applications

**Response Structure:**
1. **Direct Answer First** - Start with the specific answer to their question
2. **Brief Context** - Add educational relevance only when helpful
3. **Practical Application** - Include classroom examples only for broader topics
4. **Actionable Guidance** - Offer next steps only for complex questions

**Language Style:**
- Use clear, education-friendly language
- Avoid unnecessary technical jargon
- Keep responses proportional to question complexity
- Focus on practical value for future educators

Answer as a helpful instructor who respects students' time by providing appropriately-sized responses. For simple questions, be concise. For complex topics, provide comprehensive guidance."""

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
                print("No chunks created from PDFs - creating default accessibility content")
                # Create some default accessibility content for demo purposes
                chunks = self._create_default_content()
                if not chunks:
                    print("Failed to create default content")
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
    
    def _create_default_content(self):
        """Create default accessibility content when PDFs are not available."""
        default_content = [
            {
                "content": "Digital accessibility ensures that educational content and technology can be used by students with disabilities. This includes visual, auditory, motor, and cognitive disabilities. The goal is to create inclusive learning environments where all students can participate fully.",
                "source": "Default Content",
                "chunk_id": "default_001"
            },
            {
                "content": "WCAG (Web Content Accessibility Guidelines) provides standards for making web content accessible. The guidelines are organized around four principles: Perceivable, Operable, Understandable, and Robust (POUR). These principles help ensure content works with assistive technologies.",
                "source": "Default Content", 
                "chunk_id": "default_002"
            },
            {
                "content": "Universal Design for Learning (UDL) is an educational framework that guides the design of learning experiences to meet the needs of all learners. It emphasizes providing multiple means of representation, engagement, and action/expression.",
                "source": "Default Content",
                "chunk_id": "default_003"
            },
            {
                "content": "When creating accessible documents, use proper heading structure, provide alternative text for images, ensure good color contrast, and use descriptive link text. These practices help students using screen readers and other assistive technologies.",
                "source": "Default Content",
                "chunk_id": "default_004"
            },
            {
                "content": "Section 508 and the Americans with Disabilities Act (ADA) require educational institutions to provide accessible technology and content. This includes websites, learning management systems, and digital course materials.",
                "source": "Default Content",
                "chunk_id": "default_005"
            }
        ]
        return default_content