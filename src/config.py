"""
config.py - Central configuration for the Logistics RAG system.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "Docs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")

# ─── Document Source Directories ─────────────────────────
INCOTERMS_DIR = os.path.join(DOCS_DIR, "Icoterms")
DGFT_DIR = os.path.join(DOCS_DIR, "DGFT Trade Policy")
HS_CODES_DIR = os.path.join(DOCS_DIR, "ITC-HS Codes")
WTO_DIR = os.path.join(DOCS_DIR, "WTO")

# ─── Structured Data Files ──────────────────────────────
INCOTERMS_JSON = os.path.join(DATA_DIR, "incoterms_2020.json")
HS_CODES_JSON = os.path.join(DATA_DIR, "hs_codes_top100.json")

# ─── Embedding Model ────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Groq LLM ───────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 2048

# ─── Chunking Parameters ────────────────────────────────
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# ─── Retrieval Parameters ────────────────────────────────
TOP_K_RETRIEVAL = 6
