"""
ingest.py - Document ingestion, parsing, chunking, embedding, and FAISS index building.

Parses:
  - INCOTERMS 2020 PDFs + structured JSON
  - DGFT Foreign Trade Policy 2023 PDFs
  - ITC-HS Code PDFs + structured JSON
  - WTO Trade Policy Review PDFs

Outputs a persisted FAISS vector store at faiss_index/.
"""

import os
import json
import re
import logging
from typing import Optional

import warnings
import pdfplumber
import PyPDF2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    DOCS_DIR, DATA_DIR, FAISS_INDEX_DIR,
    INCOTERMS_DIR, DGFT_DIR, HS_CODES_DIR, WTO_DIR,
    INCOTERMS_JSON, HS_CODES_JSON,
    EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy pdfplumber/pdfminer font warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")

# ═══════════════════════════════════════════════════════════
# 1. PDF Extraction Utilities
# ═══════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF using pdfplumber with per-page error handling.
    Falls back to PyPDF2 if pdfplumber completely fails.
    Returns a list of dicts: [{page: int, text: str}, ...]
    """
    pages_data = _extract_with_pdfplumber(pdf_path)

    # Fallback to PyPDF2 if pdfplumber yielded nothing
    if not pages_data:
        logger.warning(f"pdfplumber yielded no pages for {os.path.basename(pdf_path)}, falling back to PyPDF2...")
        pages_data = _extract_with_pypdf2(pdf_path)

    return pages_data


def _extract_with_pdfplumber(pdf_path: str) -> list[dict]:
    """Extract text using pdfplumber with per-page error handling."""
    pages_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"  Parsing {os.path.basename(pdf_path)} ({total_pages} pages) with pdfplumber...")
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text_parts = []

                    # Extract regular text
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                    # Extract tables as Markdown
                    try:
                        tables = page.extract_tables()
                        for table in tables:
                            md_table = _table_to_markdown(table)
                            if md_table:
                                text_parts.append(md_table)
                    except Exception:
                        pass  # Tables are optional

                    full_text = "\n\n".join(text_parts).strip()
                    if full_text:
                        pages_data.append({"page": page_num, "text": full_text})

                except Exception as e:
                    logger.warning(f"  Skipping page {page_num}/{total_pages} of {os.path.basename(pdf_path)}: {type(e).__name__}")
                    continue

    except Exception as e:
        logger.error(f"pdfplumber failed on {os.path.basename(pdf_path)}: {e}")

    return pages_data


def _extract_with_pypdf2(pdf_path: str) -> list[dict]:
    """Fallback extraction using PyPDF2."""
    pages_data = []
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        total_pages = len(reader.pages)
        logger.info(f"  Parsing {os.path.basename(pdf_path)} ({total_pages} pages) with PyPDF2...")
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({"page": page_num, "text": text.strip()})
            except Exception as e:
                logger.warning(f"  PyPDF2 skipping page {page_num}: {type(e).__name__}")
                continue
    except Exception as e:
        logger.error(f"PyPDF2 also failed on {os.path.basename(pdf_path)}: {e}")

    return pages_data


def _table_to_markdown(table: list[list]) -> Optional[str]:
    """Convert a pdfplumber table (list of rows) to Markdown table format."""
    if not table or len(table) < 2:
        return None

    # Clean cells
    cleaned = []
    for row in table:
        cleaned_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
        cleaned.append(cleaned_row)

    # Build markdown
    header = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
    rows = "\n".join("| " + " | ".join(row) + " |" for row in cleaned[1:])

    return f"{header}\n{separator}\n{rows}"


# ═══════════════════════════════════════════════════════════
# 2. Document Loaders (PDF → LangChain Documents)
# ═══════════════════════════════════════════════════════════

def load_pdfs_from_directory(directory: str, source_label: str, chapter_prefix: str = "") -> list[Document]:
    """
    Load all PDFs from a directory and create LangChain Document objects
    with metadata (source, filename, page, chapter).
    """
    documents = []
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return documents

    pdf_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(".pdf")])
    logger.info(f"Found {len(pdf_files)} PDFs in {directory}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        pages = extract_text_from_pdf(pdf_path)

        # Try to extract chapter info from filename
        chapter = _extract_chapter_from_filename(pdf_file, chapter_prefix)

        for page_data in pages:
            doc = Document(
                page_content=page_data["text"],
                metadata={
                    "source": source_label,
                    "filename": pdf_file,
                    "page": page_data["page"],
                    "chapter": chapter,
                    "doc_type": "pdf",
                },
            )
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} pages from {source_label}")
    return documents


def _extract_chapter_from_filename(filename: str, prefix: str) -> str:
    """Try to extract chapter number/name from PDF filename."""
    # Match patterns like "Chapter01", "Chapter 2", "chapter-05", "Chapter+10"
    match = re.search(r"[Cc]hapter[\s_+\-]*(\d+)", filename)
    if match:
        return f"{prefix}Chapter {match.group(1)}"

    return prefix if prefix else ""


# ═══════════════════════════════════════════════════════════
# 3. Structured JSON Loaders
# ═══════════════════════════════════════════════════════════

def load_incoterms_json() -> list[Document]:
    """Load INCOTERMS 2020 structured JSON into LangChain Documents."""
    documents = []
    try:
        with open(INCOTERMS_JSON, "r", encoding="utf-8") as f:
            incoterms = json.load(f)

        for term in incoterms:
            # Build a rich text representation for embedding
            text = (
                f"INCOTERM Rule: {term['rule']} — {term['full_name']}\n"
                f"Transport Mode: {term['mode']}\n\n"
                f"Seller's Obligations: {term['seller_obligations']}\n\n"
                f"Buyer's Obligations: {term['buyer_obligations']}\n\n"
                f"Risk Transfer Point: {term['risk_transfer_point']}\n\n"
                f"Cost Transfer Point: {term['cost_transfer_point']}\n\n"
                f"Insurance Requirement: {term['insurance']}\n\n"
                f"Export Clearance: {term['export_clearance']}\n"
                f"Import Clearance: {term['import_clearance']}\n\n"
                f"Key Notes: {term['key_notes']}"
            )

            doc = Document(
                page_content=text,
                metadata={
                    "source": "INCOTERMS 2020",
                    "doc_type": "incoterm_rule",
                    "rule": term["rule"],
                    "full_name": term["full_name"],
                    "mode": term["mode"],
                    "risk_transfer_point": term["risk_transfer_point"],
                    "seller_obligations": term["seller_obligations"],
                    "buyer_obligations": term["buyer_obligations"],
                    "insurance": term["insurance"],
                    "export_clearance": term["export_clearance"],
                    "import_clearance": term["import_clearance"],
                },
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} INCOTERMS rules from JSON")
    except Exception as e:
        logger.error(f"Error loading INCOTERMS JSON: {e}")

    return documents


def load_hs_codes_json() -> list[Document]:
    """Load HS Code structured JSON into LangChain Documents."""
    documents = []
    try:
        with open(HS_CODES_JSON, "r", encoding="utf-8") as f:
            hs_codes = json.load(f)

        for entry in hs_codes:
            text = (
                f"HS Code: {entry['hs_code']}\n"
                f"Product Description: {entry['description']}\n"
                f"Unit of Measurement: {entry['unit']}\n"
                f"Basic Customs Duty (BCD): {entry['bcd_percent']}%\n"
                f"Integrated GST (IGST): {entry['igst_percent']}%\n"
                f"Section: {entry['section']}, Chapter: {entry['chapter']}"
            )

            doc = Document(
                page_content=text,
                metadata={
                    "source": "Indian Customs Tariff Schedule",
                    "doc_type": "hs_code",
                    "hs_code": entry["hs_code"],
                    "description": entry["description"],
                    "unit": entry["unit"],
                    "bcd_percent": entry["bcd_percent"],
                    "igst_percent": entry["igst_percent"],
                    "section": entry["section"],
                    "chapter": entry["chapter"],
                },
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} HS code entries from JSON")
    except Exception as e:
        logger.error(f"Error loading HS Codes JSON: {e}")

    return documents


# ═══════════════════════════════════════════════════════════
# 4. Chunking with Metadata Preservation
# ═══════════════════════════════════════════════════════════

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split large documents into smaller chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunked_docs = []
    for doc in documents:
        # Structured JSON entries (INCOTERMS, HS Codes) are kept whole —
        # they are already well-sized semantic units.
        if doc.metadata.get("doc_type") in ("incoterm_rule", "hs_code"):
            chunked_docs.append(doc)
            continue

        # Split longer PDF pages
        if len(doc.page_content) > CHUNK_SIZE:
            chunks = splitter.split_text(doc.page_content)
            for i, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "chunk_index": i},
                )
                chunked_docs.append(chunk_doc)
        else:
            chunked_docs.append(doc)

    logger.info(f"Chunking complete: {len(documents)} pages → {len(chunked_docs)} chunks")
    return chunked_docs


# ═══════════════════════════════════════════════════════════
# 5. Build & Persist FAISS Index
# ═══════════════════════════════════════════════════════════

def build_faiss_index(documents: list[Document]) -> FAISS:
    """Embed documents and build a FAISS vector store."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"Building FAISS index from {len(documents)} chunks...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Persist to disk
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    logger.info(f"FAISS index saved to {FAISS_INDEX_DIR}")

    return vectorstore


# ═══════════════════════════════════════════════════════════
# 6. Main Ingestion Pipeline
# ═══════════════════════════════════════════════════════════

def run_ingestion():
    """Run the full ingestion pipeline: parse → chunk → embed → persist."""
    logger.info("=" * 60)
    logger.info("Starting Logistics RAG Ingestion Pipeline")
    logger.info("=" * 60)

    all_documents = []

    # --- 1. Load INCOTERMS PDFs ---
    incoterm_pdf_docs = load_pdfs_from_directory(
        INCOTERMS_DIR, source_label="INCOTERMS 2020 PDF"
    )
    all_documents.extend(incoterm_pdf_docs)

    # --- 2. Load INCOTERMS structured JSON ---
    incoterm_json_docs = load_incoterms_json()
    all_documents.extend(incoterm_json_docs)

    # --- 3. Load DGFT Foreign Trade Policy PDFs ---
    dgft_docs = load_pdfs_from_directory(
        DGFT_DIR, source_label="DGFT Foreign Trade Policy 2023", chapter_prefix="DGFT FTP "
    )
    all_documents.extend(dgft_docs)

    # --- 4. Load ITC-HS Code PDFs ---
    hs_pdf_docs = load_pdfs_from_directory(
        HS_CODES_DIR, source_label="Indian Customs Tariff Schedule PDF"
    )
    all_documents.extend(hs_pdf_docs)

    # --- 5. Load HS Codes structured JSON ---
    hs_json_docs = load_hs_codes_json()
    all_documents.extend(hs_json_docs)

    # --- 6. Load WTO Trade Policy Review PDFs ---
    wto_docs = load_pdfs_from_directory(
        WTO_DIR, source_label="WTO Trade Policy Review"
    )
    all_documents.extend(wto_docs)

    logger.info(f"Total raw documents loaded: {len(all_documents)}")

    # --- 7. Chunk all documents ---
    chunked_documents = chunk_documents(all_documents)

    # --- 8. Build FAISS index ---
    vectorstore = build_faiss_index(chunked_documents)

    logger.info("=" * 60)
    logger.info("Ingestion pipeline completed successfully!")
    logger.info(f"Total chunks indexed: {len(chunked_documents)}")
    logger.info("=" * 60)

    return vectorstore


if __name__ == "__main__":
    run_ingestion()
