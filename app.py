"""
KB Portal — FastAPI Backend
Run:  uvicorn app:app --reload --port 8000
Requires: pip install fastapi uvicorn python-multipart langchain langchain-community
          langchain-groq chromadb sentence-transformers pypdf docx2txt python-dotenv
"""
import os
import shutil
import gc
import time
import json
import logging
from typing import Optional
from datetime import date

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── LangChain / Vector / LLM imports ──
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Setup terminal logging for visual debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("KB_Portal")

load_dotenv()

app = FastAPI(title="KB Portal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ──
UPLOAD_DIR = "uploaded_docs"
CHROMA_DIR = "chroma_db"
ARTICLES_DB = "articles.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Articles store (JSON file as simple DB) ──
def load_articles():
    if not os.path.exists(ARTICLES_DB):
        return []
    try:
        with open(ARTICLES_DB, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except Exception as e:
        logger.error(f"Error loading articles JSON: {e}")
        return []


def save_articles(articles):
    try:
        with open(ARTICLES_DB, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving articles JSON: {e}")


# ── Embedding model (Loaded globally once) ──
embedding = None

def get_embedding():
    global embedding
    if embedding is None:
        # Offloads tokenization and vector calculations to Hugging Face servers
        embedding = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=os.getenv("HF_TOKEN") # Or "HUGGINGFACEHUB_API_TOKEN"
        )
    return embedding

# ── LLM (Groq LLaMA) ──
logger.info("Initializing ChatGroq client...")
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# ── RAG System Prompt ──
PROMPT_TEMPLATE = """
You are a company knowledge base assistant.
Answer ONLY using the provided context.

Rules:
1. Do not use outside knowledge.
2. If information is missing, say: "Information not found in uploaded documents."
3. If multiple questions are asked, answer each separately.
4. Keep answers concise and professional.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


# ── Build / Rebuild Vector Store ──
def build_vector_store():
    docs = []

    # 1. Load all files from the directory
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)

        try:
            if fname.endswith(".txt"):
                loader = TextLoader(fpath, encoding="utf-8")
            elif fname.endswith(".pdf"):
                loader = PyPDFLoader(fpath)
            elif fname.endswith(".docx"):
                loader = Docx2txtLoader(fpath)
            else:
                continue

            loaded = loader.load()

            for d in loaded:
                d.metadata["source"] = fname

            docs.extend(loaded)

        except Exception as e:
            logger.error(f"Error loading {fname}: {e}")

    if not docs:
        return None

    # 2. Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split documents into {len(chunks)} textual chunks.")

    if len(chunks) == 0:
        logger.warning("Extracted text generated 0 usable vectors. Check if PDFs are image-only scans.")
        return None

    # 3. Memory Cleanup & Clear Old DB (Crucial for Render's 512MB limit)
    gc.collect()
    time.sleep(0.5)
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        except Exception as e:
            logger.error(f"Could not purge directory cleanly: {e}")

    # 4. Build single fresh database instance
    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding(),
        persist_directory=CHROMA_DIR
    )
    
    logger.info("ChromaDB indexing complete. Vector store successfully built!")
    return db


def get_retriever():
    if os.path.exists(CHROMA_DIR):
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        return db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 15})
    return None


# ══════════════════════════════════════
#  FASTAPI ROUTE HANDLERS
# ══════════════════════════════════════

# ── Startup Lifecycle Hook ──
@app.on_event("startup")
def startup_event():
    """Initializes vector store on system spinup if files already exist."""
    logger.info("System startup: Checking and initializing available data caches...")
    build_vector_store()


# ── GET all articles ──
@app.get("/articles")
def get_articles():
    return load_articles()


# ── CREATE article ──
@app.post("/articles")
async def create_article(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    category: str = Form("Other"),
    tags: str = Form(""),
    author: str = Form(""),
    content: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    articles = load_articles()
    new_id = max((a["id"] for a in articles), default=0) + 1
    filename = None

    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".txt", ".pdf", ".docx"):
            raise HTTPException(status_code=400, detail="Unsupported file type structure.")
        
        filename = f"{new_id}_{file.filename}"
        fpath = os.path.join(UPLOAD_DIR, filename)
        
        with open(fpath, "wb") as f:
            f.write(await file.read())
            
        logger.info(f"New file saved to disk: {filename}. Triggering indexing task.")
        background_tasks.add_task(build_vector_store)

    article = {
        "id": new_id,
        "title": title,
        "category": category,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "author": author,
        "content": content,
        "filename": filename,
        "updated": str(date.today()),
    }
    articles.append(article)
    save_articles(articles)
    return article


# ── UPDATE article ──
@app.put("/articles/{article_id}")
async def update_article(
    article_id: int,
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    category: str = Form("Other"),
    tags: str = Form(""),
    author: str = Form(""),
    content: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    articles = load_articles()
    idx = next((i for i, a in enumerate(articles) if a["id"] == article_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Requested Article ID not found.")

    filename = articles[idx].get("filename")
    file_changed = False

    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".txt", ".pdf", ".docx"):
            raise HTTPException(status_code=400, detail="Unsupported file type structure.")
        
        # Scrub out previous variant file
        if filename:
            old_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(old_path):
                os.remove(old_path)
                
        filename = f"{article_id}_{file.filename}"
        fpath = os.path.join(UPLOAD_DIR, filename)
        
        with open(fpath, "wb") as f:
            f.write(await file.read())
            
        file_changed = True

    articles[idx].update({
        "title": title,
        "category": category,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "author": author,
        "content": content,
        "filename": filename,
        "updated": str(date.today()),
    })
    
    save_articles(articles)
    
    if file_changed:
        logger.info(f"File altered for Article {article_id}. Queueing vector updates.")
        background_tasks.add_task(build_vector_store)
        
    return articles[idx]


# ── DELETE article ──
@app.delete("/articles/{article_id}")
def delete_article(article_id: int, background_tasks: BackgroundTasks):
    articles = load_articles()
    article = next((a for a in articles if a["id"] == article_id), None)
    if not article:
        raise HTTPException(status_code=404, detail="Requested Article ID not found.")

    filename = article.get("filename")
    if filename:
        fpath = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(fpath):
            os.remove(fpath)
        logger.info(f"Associated file '{filename}' removed. Queueing vector sync updates.")
        background_tasks.add_task(build_vector_store)

    articles = [a for a in articles if a["id"] != article_id]
    save_articles(articles)
    return {"deleted": article_id}


# ── SERVE FILE ──
@app.get("/file/{article_id}")
def serve_file(article_id: int):
    articles = load_articles()
    article = next((a for a in articles if a["id"] == article_id), None)
    if not article or not article.get("filename"):
        raise HTTPException(status_code=404, detail="File metadata registration missing.")
        
    fpath = os.path.join(UPLOAD_DIR, article["filename"])
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Target file not found on system storage.")
    return FileResponse(fpath, filename=article["filename"])


# ── ASK question (RAG pipeline) ──
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question payload cannot be empty.")

    retriever = get_retriever()
    if retriever is None:
        logger.warning("RAG Query executed, but retriever context is completely offline.")
        return {
            "answer": "No documents have been uploaded yet. Please upload files in the Admin Console first.",
            "confidence": 0.0,
            "sources": [],
            "mode": "no_docs"
        }

    # Fetch document contexts from DB
    retrieved_docs = retriever.invoke(question)
    logger.info(f"Query: '{question}' yielded {len(retrieved_docs)} matching contexts.")
    
    # Calculate extraction query confidence score
    confidence = min(len(retrieved_docs) / 4, 1.0)

    if not retrieved_docs:
        return {
            "answer": "Information not found in uploaded documents.",
            "confidence": 0.0,
            "sources": [],
            "mode": "no_match"
        }

    # Construct and send strict context prompt instructions to LLaMA
    context = "\n\n".join([d.page_content[:1000] for d in retrieved_docs])
    prompt_text = PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt_text).content

    sources = list({d.metadata.get("source", "Unknown") for d in retrieved_docs})

    return {
        "answer": response,
        "confidence": round(confidence, 2),
        "sources": sources,
        "mode": "rag"
    }


# ── FORCE REFRESH vector store ──
@app.post("/refresh")
def refresh_kb():
    """Synchronously rebuilds store to clear memory bugs on demand."""
    gc.collect()
    time.sleep(0.5)
    db = build_vector_store()
    return {"status": "refreshed", "has_docs": db is not None}


# ── Health check / SPA entry point ──
@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "message": "KB Portal Backend API running successfully"}