import os
import PyPDF2
from typing import List, Dict
from config import Config

class PDFProcessor:
    def __init__(self):
        self.pdf_directory = Config.PDF_DIRECTORY
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
        return text
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, str]]:
        """Split text into chunks with metadata."""
        chunks = []
        text = text.replace('\n', ' ').strip()
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append({
                    'content': chunk.strip(),
                    'source': filename,
                    'chunk_id': f"{filename}_chunk_{len(chunks)}"
                })
        
        return chunks
    
    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """Process all PDFs in the directory and return chunks."""
        all_chunks = []
        
        if not os.path.exists(self.pdf_directory):
            print(f"PDF directory {self.pdf_directory} not found")
            return all_chunks
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            return all_chunks
        
        print(f"Processing {len(pdf_files)} PDF files...")
        
        for filename in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, filename)
            print(f"Processing: {filename}")
            
            text = self.extract_text_from_pdf(pdf_path)
            if text.strip():
                chunks = self.chunk_text(text, filename)
                all_chunks.extend(chunks)
                print(f"  - Created {len(chunks)} chunks")
            else:
                print(f"  - No text extracted from {filename}")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks