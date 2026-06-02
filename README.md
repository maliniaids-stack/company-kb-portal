# KB Portal — Setup Guide

## Folder structure
```
kb_portal/
├── index.html      ← open this in browser (or serve with any static server)
├── app.py          ← FastAPI backend
├── .env            ← add your GROQ_API_KEY here
├── uploaded_docs/  ← auto-created, stores uploaded files
├── chroma_db/      ← auto-created, vector store
└── articles.json   ← auto-created, article metadata
```

## 1. Install dependencies
```bash
pip install fastapi uvicorn python-multipart langchain langchain-community \
            langchain-groq chromadb sentence-transformers pypdf docx2txt \
            python-dotenv
```

## 2. Add your Groq API key
Create a `.env` file next to `app.py`:
```
GROQ_API_KEY=your_key_here
```

## 3. Start the backend
```bash
uvicorn app:app --reload --port 8000
```

## 4. Open the frontend
Just open `index.html` in your browser (double-click, or serve with):
```bash
python -m http.server 5500
# then visit http://localhost:5500
```

## How it works

### Explore Hub
- Anyone can search, filter by category/tag, and click articles to read them.
- Articles posted by admin are shown here with file download links.

### Admin Console
- Create articles with title, category, tags, author, body text, AND/OR an uploaded file.
- Edit → change any field, or upload a new file to replace the old one.
- Delete → removes article + file from disk.
- Every file upload automatically rebuilds the ChromaDB vector store (Steps 1–5).

### AI Chat Room
Calls `POST /ask` on the backend which runs the full RAG pipeline:
1. **PDF Upload** → stored in `uploaded_docs/` via Admin Console
2. **Text Extraction** → PyPDF / Docx2txt / TextLoader
3. **Chunking** → 500 chars, 50 overlap
4. **Embeddings** → sentence-transformers/all-MiniLM-L6-v2
5. **Vector Storage** → ChromaDB (rebuilt on every file add/remove)
6. **Retrieval** → MMR retrieval, k=4
7. **Generation** → LLaMA-3.3-70b via Groq

### Analytics
Live counts of articles, categories, tags, and a bar chart by category.
