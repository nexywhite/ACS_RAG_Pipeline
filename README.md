Original task for Engineering Teamwork I: Applied Computer Science:

Task 3 — Build Your RAG Pipeline (Week 6) 
Connect all steps: 
1. Prepare a simple user Inteface (similar to that of Gemini, OpenAI, or DeepSeek) 
2. Let user enters a question (put a prompt) 
3. Vector search searches the relevant chunks (similarity search) 
  - Must use at least 3 different data format (pdf, video, image, text, web-content, etc.) 
4. Retrieved text is inserted into a prompt template for LLM 
5. Prompt is sent to an LLM 
6. A final answer is produced 
Required Output: 
Show one question that fails without retrieval but succeeds with RAG.

Retrieval-Augmented Generation (RAG) Pipeline — PDF + Text + Image OCR + Ollama + ChromaDB
A complete RAG pipeline using PDFs, text files, images (OCR), sentence transformers, ChromaDB,
Ollama, and Streamlit.

Overview:
This repository contains a fully functioning RAG system capable of loading documents, extracting text,
chunking, embedding, storing in a vector database, retrieving relevant context, and generating
grounded answers through a local LLM.
Features:
- Multiple data formats (PDF, text, images)
- OCR support via Tesseract
- Chunking & semantic embeddings
- ChromaDB vector storage
- Local LLM (llama3.2:1b) via Ollama
- Streamlit UI
- Base vs RAG answer comparison

Architecture:
PDF/Text/Image → Text Extraction → Chunking → Embeddings → ChromaDB → Retriever → LLM → Answer

Project Structure:
- RAG_Chatbot.py          –    Streamlit UI
- rag_query.py            –    CLI test
- index_documents.py      –    Build vector DB
- preprocess_loaders.py   –    PDF/Text/Image loaders + OCR + chunker
- requirements.txt        –    Dependencies
- chroma_db/              –    Generated vector DB
- data/                   –    Knowledge base files

Installation:
1. Clone repository:
   - git clone <your-repo-url>
   - cd ACS_RAG_Pipeline
   
2. Create virtual environment and activate it:
   - python -m venv .venv
   - .\.venv\Scripts\activate   # Windows
   - source .venv/bin/activate  # Mac/Linux

3. Install dependencies:
   - pip install -r requirements.txt

4. Install Tesseract OCR (separate):
   - https://github.com/UB-Mannheim/tesseract/wiki   # Windows
   - brew install tesseract                          # Mac
   - sudo apt install tesseract-ocr                  # Linux

5. Install FFmpeg (separate):
   - winget install Gyan.FFmpeg                      # Windows
   - brew install ffmpeg                             # Mac
   - sudo apt install ffmpeg                         # Linux

6. Install Ollama (separate):
   - https://ollama.com/download
   
7. Prepare knowledge base:
   - data/pdfs/     - PDF documents
   - data/text/     - TXT / MD files
   - data/images/   - Images (OCR)
   - data/audio/    - Audio files (MP3, WAV, etc.)
   - data/video/    - Video files (MP4, MKV, etc.)

8. Run indexer, RAG, and RAG Chatbot:
    - Indexer:
    - python index_documents.py
      - extracts text
      - chunks documents
      - encodes embeddings
      - saves everything into ChromaDB

    - RAG Queries:
    - python rag_query.py
      - gives base LLM answer (no retrieval)
      - gives RAG answer (with retrieval)
      - gives top chunks retrieved

    - RAG Chatbot:
    - streamlit run RAG_Chatbot.py
      - shows input box
      - shows base LLM answer
      - shows RAG answer
      - shows retrieved chunk explorer

Example Where RAG Beats Base LLM:
- Question: “What are the opening hours of the Evergreen Public Library?”
- Base LLM: “I'm not aware of any specific information about an "Evergreen Public Library".”
- RAG Answer: “According to the context, the opening hours of the Evergreen Public Library are:
               Monday – Friday: 9:00 AM – 7:00 PM Saturday: 10:00 AM – 5:00 PM Sunday: Closed.”

Technologies Used:
- Python
- LangChain
- Ollama
- ChromaDB
- Sentence Transformers
- Tesseract OCR
- Streamlit

Packages used (see requirements.txt):
- langchain>=0.2.0
- langchain-community>=0.2.0
- langchain-openai>=0.1.0
- langchain-ollama>=0.1.0
- chromadb>=0.4.22
- sentence-transformers>=2.5.2
- huggingface_hub>=0.20.0
- pytesseract
- pillow
- subprocess
- tempfile
- pypdf>=4.0.1
- openai-whisper
- python-dotenv>=1.0.0
- tiktoken>=0.5.2
- streamlit>=1.29.0
