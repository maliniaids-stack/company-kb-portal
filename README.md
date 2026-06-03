---
title: Company KB Portal
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🏢 Company KB Portal

### ⚡ AI-Powered Knowledge Base with RAG (Retrieval Augmented Generation)

A modern **internal company knowledge system** that lets users:
- 🔎 Search company documents instantly
- 🤖 Ask AI questions (ChatGPT-like)
- 📂 Upload PDFs / DOCX and auto-index them
- 📊 View analytics on documents & categories

Built with **FastAPI + ChromaDB + Groq LLaMA 3.3**

---

## 🚀 Live Demo

👉 Try it now:  
:contentReference[oaicite:0]{index=0}

---

## 📸 Product Preview

![Dashboard](assets/screenshot2.png)

![RAG Architecture](assets/flowchart.jpeg)

---

## ✨ Key Features

### 🔎 Smart Search (Explore Hub)
- Search across all company documents
- Filter by category, tags, and metadata
- Instant document preview + download

---

### 🤖 AI Chat Assistant
Ask anything like:
- “What is the HR policy?”
- “Summarize data security rules”
- “Show reimbursement process”

Powered by:
- LLaMA 3.3 (Groq)
- Vector similarity search (ChromaDB)

---

### 🛠 Admin Dashboard
- Upload new documents
- Edit / delete articles
- Auto re-index vector database
- File version handling

---

### 📊 Analytics Dashboard
- Total documents
- Category distribution
- Tag insights
- Live usage stats

---

## 🧠 System Architecture

```text id="saas_arch"
User Query
   ↓
Frontend (index.html)
   ↓
FastAPI Backend (/ask)
   ↓
Text Extraction (PDF/DOCX)
   ↓
Chunking (Recursive Splitter)
   ↓
Embeddings (MiniLM)
   ↓
ChromaDB Vector Search
   ↓
MMR Retrieval (Top K = 4)
   ↓
Groq LLaMA 3.3 Response
