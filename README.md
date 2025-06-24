# Web Accessibility Q&A Chatbot

A chatbot that answers web accessibility questions using your PDF knowledge base, built with OpenAI, LangChain, ChromaDB, and Streamlit.

## Features

- **PDF Knowledge Base**: Automatically processes PDF documents and creates a searchable knowledge base
- **Intelligent Q&A**: Uses OpenAI's GPT models to provide accurate answers based on your documents
- **Web Interface**: Clean, user-friendly chat interface built with Streamlit
- **Source Attribution**: Shows which documents were used to generate each answer
- **Persistent Storage**: ChromaDB stores your knowledge base locally for fast retrieval

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Add PDF Documents

Place your accessibility PDF documents in the `data/pdfs/` directory.

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will start on http://localhost:8501

### 5. Initialize the Knowledge Base

1. Open the web interface
2. Click "Initialize Chatbot" 
3. Wait for the PDF processing to complete
4. Start asking questions!

## Usage

### Example Questions

- "What are the WCAG 2.1 guidelines for color contrast?"
- "How do I make images accessible?"
- "What is the proper way to use ARIA labels?"
- "How can I make forms more accessible?"
- "What are the requirements for keyboard navigation?"

### Features

- **Initialize Chatbot**: Processes your PDFs and creates the knowledge base
- **Refresh Knowledge Base**: Reprocesses all PDFs (useful when you add new documents)
- **Clear Chat**: Clears the conversation history
- **Source Attribution**: Each answer shows which PDF documents were referenced

## Project Structure

```
pdf-chat/
├── streamlit_app.py          # Main Streamlit application
├── app.py                    # Original Gradio application (deprecated)
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── data/
│   └── pdfs/                # Place your PDF files here
├── utils/
│   ├── pdf_processor.py     # PDF text extraction and chunking
│   ├── vector_store.py      # ChromaDB operations
│   └── qa_chain.py          # LangChain Q&A chain
└── chroma_db/               # ChromaDB storage (created automatically)
```

## Configuration

You can modify settings in `config.py`:

- **Models**: Change OpenAI models for embeddings and chat
- **Chunking**: Adjust chunk size and overlap for PDF processing
- **Retrieval**: Modify number of documents retrieved per query
- **UI Settings**: Customize app title and description

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY environment variable is required"**
   - Make sure you've created a `.env` file with your API key

2. **"No PDF files found"**
   - Ensure your PDF files are in the `data/pdfs/` directory

3. **"Failed to initialize knowledge base"**
   - Check that your PDFs contain readable text (not just images)
   - Verify your OpenAI API key is valid and has sufficient credits

4. **Slow responses**
   - This is normal for the first run as it processes all PDFs
   - Subsequent runs will be faster as the knowledge base is cached

## Cost Considerations

- **Embeddings**: ~$0.0001 per 1K tokens for processing your PDFs
- **Chat**: ~$0.001 per 1K tokens for each question/answer
- **Typical usage**: $5-20/month for moderate usage

## Next Steps

- Add more PDF documents to expand the knowledge base
- Experiment with different chunk sizes and retrieval parameters
- Customize the UI and prompts for your specific use case
- Deploy to cloud platforms for wider access

## Support

For issues and questions, please check the configuration and ensure all dependencies are installed correctly.