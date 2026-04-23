# DocMind AI — Financial Document Intelligence System

An AI-powered document chatbot that combines RAG (Retrieval-Augmented Generation) 
with live financial data to answer questions from your PDFs and the web.

## Live Demo
[Click here to try it live](YOUR_STREAMLIT_URL_HERE)

## Features
- Multi-PDF upload and indexing
- Semantic search with confidence scoring
- Web-augmented answers via DuckDuckGo
- Cross-document comparison with similarity scoring
- OCR support for scanned PDFs
- Live stock charts with candlestick visualization
- Financial metrics extraction from documents
- Page-level source citation

## Tech Stack
| Component | Technology |
|---|---|
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| LLM | Groq LLaMA 3.1 8B |
| PDF Parsing | PyMuPDF + pymupdf4llm |
| Web Search | DuckDuckGo Search |
| Stock Data | yfinance |
| Charts | Plotly |
| UI | Streamlit |
| Cost | $0 |

## Installation

1. Clone the repository
   git clone https://github.com/YOUR_USERNAME/docmind-ai.git
   cd docmind-ai

2. Create virtual environment
   python3.11 -m venv venv
   source venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

4. Run the app
   streamlit run app.py

## Usage
1. Enter your Groq API key in the sidebar
2. Upload one or more PDF files
3. Ask questions in the chat
4. Switch to Compare Documents to compare two PDFs
5. Switch to Financial Analysis for live stock charts

## Architecture
User uploads PDF → PyMuPDF extracts text → LangChain splits into 
500-token chunks → HuggingFace converts to embeddings → ChromaDB 
stores vectors → User asks question → Cosine similarity retrieves 
top-4 chunks → Groq LLaMA generates grounded answer → 
Confidence score displayed

## Get a Free Groq API Key
https://console.groq.com