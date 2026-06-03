import os
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Lightweight LangChain / Chroma imports (No Torch, No Local Transformers)
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# CRITICAL FOR 512MB RAM: Stop ChromaDB from spinning up background telemetry threads
os.environ["ANONYMIZED_TELEMETRY"] = "False"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_docs"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------------------------------------------
# LIGHTWEIGHT API EMBEDDINGS (CONSUMES ~0MB LOCAL RAM)
# ------------------------------------------------------------------
class LightHuggingFaceEmbeddings(Embeddings):
    """
    Ultra-lightweight wrapper for Hugging Face Inference Providers API.
    Bypasses local PyTorch/Transformers to fit within strict memory limits.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is missing from your configuration!")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.api_url, 
            headers=self.headers, 
            json={"inputs": texts}
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"HuggingFace API Error: {response.text}"
            )
        return response.json()

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0]

def get_embedding():
    return LightHuggingFaceEmbeddings()

# ------------------------------------------------------------------
# CORE RAG LOGIC
# ------------------------------------------------------------------
def build_vector_store():
    """Scans UPLOAD_DIR and builds/rebuilds Chroma vector store in-place."""
    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        print("No documents found in uploaded_docs. Skipping indexing.")
        return None

    all_docs = []
    for file in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith((".docx", ".doc")):
                loader = Docx2txtLoader(file_path)
            else:
                continue
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {file}: {e}")

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    embeddings = get_embedding()
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("Vector store initialized successfully.")
    return db

@app.on_event("startup")
def startup_event():
    build_vector_store()

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    build_vector_store()
    return {"message": f"Successfully uploaded and indexed {file.filename}"}

@app.post("/query")
async def query_bot(request: dict):
    user_query = request.get("query")
    if not user_query:
        raise HTTPException(status_code=400, detail="Query string is required.")
    
    embeddings = get_embedding()
    
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload files first.")
        
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    template = """You are a helpful assistant. Use the following context to answer the question.
    If you don't know the answer, say you don't know.
    
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "question": user_query})
    return {"response": response.content}