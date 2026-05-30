"""
rag_chain.py - RAG pipeline with Groq LLM, structured outputs, and duty computation.

Provides three query modes:
  1. INCOTERMS Query - Returns structured rule details with Pydantic schema
  2. HS Code Lookup  - Semantic product search → HS code + duty computation
  3. General Trade Policy Query - Free-form regulatory Q&A with citations
"""

import os
import json
import logging
from typing import Optional

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.documents import Document

from config import (
    FAISS_INDEX_DIR, EMBEDDING_MODEL_NAME,
    GROQ_API_KEY, GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS,
    TOP_K_RETRIEVAL, HS_CODES_JSON,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. Pydantic Schemas for Structured Outputs
# ═══════════════════════════════════════════════════════════

class IncotermsResponse(BaseModel):
    """Structured output schema for INCOTERMS queries."""
    rule_name: str = Field(description="The INCOTERM rule abbreviation (e.g., FOB, CIF, DDP)")
    full_name: str = Field(description="Full name of the INCOTERM rule (e.g., Free on Board)")
    transport_mode: str = Field(description="Applicable transport mode (e.g., Any mode, Sea and inland waterway only)")
    obligation_party: str = Field(description="Detailed description of which party (buyer/seller) bears the primary obligation for the queried scenario")
    risk_transfer_point: str = Field(description="The exact geographical/logistical point where risk transfers from seller to buyer")
    cost_allocation: str = Field(description="Summary of how costs are split between buyer and seller")
    insurance_requirement: str = Field(description="Insurance obligations under this rule")
    policy_citation: str = Field(description="Specific reference to INCOTERMS 2020 ICC rules, section, or page number")
    key_notes: str = Field(description="Additional critical distinctions or 2020 updates relevant to this rule")


class HSCodeResponse(BaseModel):
    """Structured output schema for HS Code lookup."""
    hs_code: str = Field(description="The matched 8-digit HS/ITC code")
    product_description: str = Field(description="Official product description from the tariff schedule")
    bcd_rate: float = Field(description="Basic Customs Duty rate as a percentage")
    igst_rate: float = Field(description="Integrated GST rate as a percentage")
    unit: str = Field(description="Unit of measurement for the product")
    match_confidence: str = Field(description="Confidence level of the match: High, Medium, or Low")
    reasoning: str = Field(description="Brief explanation of why this HS code was matched to the user's query")


class TradeQueryResponse(BaseModel):
    """Structured output for general trade policy queries."""
    answer: str = Field(description="Detailed answer to the trade compliance query")
    source_documents: str = Field(description="Source document names and chapters referenced")
    policy_citations: str = Field(description="Specific policy clause, chapter, or section citations")
    key_regulations: str = Field(description="Key regulatory provisions or rules that apply")
    practical_implications: str = Field(description="Practical implications for the exporter/importer")


class UnifiedQueryResponse(BaseModel):
    """Structured output for unified trade queries across all 4 sources."""
    description: str = Field(description="A concise summary or direct answer to the user's query.")
    incoterms_context: str = Field(description="Detailed, comprehensive explanation of relevance and rules from INCOTERMS 2020. Put 'Not applicable' if irrelevant.")
    dgft_context: str = Field(description="Detailed, comprehensive explanation of relevance and policies from DGFT Foreign Trade Policy 2023. Put 'Not applicable' if irrelevant.")
    hs_code_context: str = Field(description="Detailed, comprehensive explanation of relevance and details from ITC-HS Customs Tariff Codes, including duty rates if applicable. Put 'Not applicable' if irrelevant.")
    wto_context: str = Field(description="Detailed, comprehensive explanation of relevance and insights from WTO Trade Policy Reviews. Put 'Not applicable' if irrelevant.")
    citations: list[str] = Field(description="List of specific document citations.")


# ═══════════════════════════════════════════════════════════
# 2. Duty Computation Engine (deterministic Python)
# ═══════════════════════════════════════════════════════════

def compute_customs_duty(
    assessable_value: float,
    bcd_percent: float,
    igst_percent: float,
    sws_percent: float = 10.0,
) -> dict:
    """
    Compute the cascading Indian customs duty structure.
    
    Formula:
      BCD = Assessable Value × BCD%
      SWS = BCD × SWS% (Social Welfare Surcharge, typically 10% of BCD)
      IGST = (Assessable Value + BCD + SWS) × IGST%
      Total Duty = BCD + SWS + IGST
      Total Landed Cost = Assessable Value + Total Duty
    """
    bcd_amount = assessable_value * (bcd_percent / 100)
    sws_amount = bcd_amount * (sws_percent / 100)
    igst_base = assessable_value + bcd_amount + sws_amount
    igst_amount = igst_base * (igst_percent / 100)
    total_duty = bcd_amount + sws_amount + igst_amount
    total_landed = assessable_value + total_duty

    return {
        "assessable_value": round(assessable_value, 2),
        "bcd_rate": f"{bcd_percent}%",
        "bcd_amount": round(bcd_amount, 2),
        "sws_rate": f"{sws_percent}% of BCD",
        "sws_amount": round(sws_amount, 2),
        "igst_rate": f"{igst_percent}%",
        "igst_base": round(igst_base, 2),
        "igst_amount": round(igst_amount, 2),
        "total_duty": round(total_duty, 2),
        "total_landed_cost": round(total_landed, 2),
    }


# ═══════════════════════════════════════════════════════════
# 3. RAG Pipeline Core
# ═══════════════════════════════════════════════════════════

class TradeRAGPipeline:
    """Main RAG pipeline for trade compliance queries."""

    def __init__(self):
        """Initialize the pipeline: load FAISS index, embedding model, and Groq LLM."""
        logger.info("Initializing Trade RAG Pipeline...")

        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load FAISS index
        if not os.path.exists(FAISS_INDEX_DIR):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_DIR}. "
                "Run `python src/ingest.py` first to build the index."
            )
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RETRIEVAL},
        )

        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
        )

        # Load HS codes for direct lookup
        with open(HS_CODES_JSON, "r", encoding="utf-8") as f:
            self.hs_codes_data = json.load(f)

        logger.info("Trade RAG Pipeline initialized successfully.")

    # ─── INCOTERMS Query ────────────────────────────────
    def query_incoterms(self, user_query: str) -> dict:
        """
        Query INCOTERMS rules with structured output.
        Returns a dict matching the IncotermsResponse schema.
        """
        # Retrieve relevant context, filtering for INCOTERMS docs
        docs = self.vectorstore.similarity_search(
            user_query,
            k=TOP_K_RETRIEVAL,
            filter=lambda meta: meta.get("source", "").startswith("INCOTERMS"),
        )

        # Fallback to general retrieval if filter returns nothing
        if not docs:
            docs = self.retriever.invoke(user_query)

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert international trade compliance advisor specializing in INCOTERMS 2020 rules published by the ICC.

Based ONLY on the provided context, answer the user's query with precise, structured information.

IMPORTANT RULES:
- Only use information present in the context
- If the context does not contain the answer, say so clearly
- Be precise about risk transfer points and obligation parties
- Cite specific INCOTERMS 2020 rules and sections

CONTEXT:
{context}"""),
            ("human", "{query}"),
        ])

        structured_llm = self.llm.with_structured_output(IncotermsResponse)
        chain = prompt | structured_llm

        try:
            result = chain.invoke({"context": context, "query": user_query})
            return result.model_dump()
        except Exception as e:
            logger.error(f"Structured output failed, falling back: {e}")
            # Fallback to unstructured
            fallback_chain = prompt | self.llm | StrOutputParser()
            text_result = fallback_chain.invoke({"context": context, "query": user_query})
            return {"answer": text_result, "error": "Structured output unavailable"}

    # ─── HS Code Lookup ────────────────────────────────
    def lookup_hs_code(self, product_description: str, assessable_value: Optional[float] = None) -> dict:
        """
        Semantic HS Code lookup: match a product description to the closest HS code,
        then compute customs duties deterministically.
        """
        # Step 1: Semantic search against HS code entries
        docs = self.vectorstore.similarity_search(
            product_description,
            k=4,
            filter={"doc_type": "hs_code"},
        )

        if not docs:
            # Fallback to general search
            docs = self.retriever.invoke(f"HS code for {product_description}")

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Indian customs classification expert. 
Based on the HS code entries in the context below, identify the BEST matching HS code for the user's product description.

CONTEXT:
{context}

Return the exact HS code, product description, BCD rate, IGST rate, and unit from the context.
If no exact match exists, return the closest match and indicate low confidence."""),
            ("human", "Find the HS code for: {query}"),
        ])

        structured_llm = self.llm.with_structured_output(HSCodeResponse)
        chain = prompt | structured_llm

        try:
            result = chain.invoke({"context": context, "query": product_description})
            response = result.model_dump()
        except Exception as e:
            logger.error(f"HS Code structured output failed: {e}")
            # Try to extract from metadata directly
            if docs and docs[0].metadata.get("doc_type") == "hs_code":
                meta = docs[0].metadata
                response = {
                    "hs_code": meta.get("hs_code", "N/A"),
                    "product_description": meta.get("description", "N/A"),
                    "bcd_rate": meta.get("bcd_percent", 0),
                    "igst_rate": meta.get("igst_percent", 0),
                    "unit": meta.get("unit", "N/A"),
                    "match_confidence": "Medium",
                    "reasoning": "Matched from vector store metadata (LLM fallback)",
                }
            else:
                return {"error": str(e)}

        # Step 2: Compute customs duties (deterministic Python math)
        if assessable_value and assessable_value > 0:
            bcd = response.get("bcd_rate", 0)
            igst = response.get("igst_rate", 0)
            if isinstance(bcd, str):
                bcd = float(bcd.replace("%", ""))
            if isinstance(igst, str):
                igst = float(igst.replace("%", ""))
            duty_computation = compute_customs_duty(assessable_value, bcd, igst)
            response["duty_computation"] = duty_computation

        # Attach the retrieved context sources
        response["source_chunks"] = [
            {"content_preview": doc.page_content[:200], "metadata": doc.metadata}
            for doc in docs[:3]
        ]

        return response

    # ─── General Trade Policy Query ─────────────────────
    def query_trade_policy(self, user_query: str) -> dict:
        """
        General trade policy query with citations.
        Searches across all document types (DGFT, WTO, etc.).
        """
        docs = self.retriever.invoke(user_query)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Build source citation string
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            chapter = doc.metadata.get("chapter", "")
            sources.append(f"{src} — {chapter} (Page {page})")
        source_str = "\n".join(set(sources))

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert trade compliance advisor specializing in Indian foreign trade regulations, DGFT policies, customs procedures, and WTO trade rules.

Answer the user's question based ONLY on the provided context.

IMPORTANT:
- Cite specific chapters, sections, and policy names
- If the answer involves duty rates, provide the exact rates from the context
- If the context doesn't contain the answer, say so clearly
- Provide practical implications for exporters/importers

RETRIEVED CONTEXT:
{context}

SOURCES:
{sources}"""),
            ("human", "{query}"),
        ])

        structured_llm = self.llm.with_structured_output(TradeQueryResponse)
        chain = prompt | structured_llm

        try:
            result = chain.invoke({
                "context": context,
                "query": user_query,
                "sources": source_str,
            })
            response = result.model_dump()
        except Exception as e:
            logger.error(f"Trade policy structured output failed: {e}")
            fallback_chain = prompt | self.llm | StrOutputParser()
            text_result = fallback_chain.invoke({
                "context": context,
                "query": user_query,
                "sources": source_str,
            })
            response = {"answer": text_result, "sources": source_str}

        response["retrieved_sources"] = [
            {
                "source": doc.metadata.get("source"),
                "chapter": doc.metadata.get("chapter"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:150],
            }
            for doc in docs
        ]

        return response

    # ─── Unified Trade Query ──────────────────────────────
    def unified_trade_query(self, user_query: str) -> dict:
        """
        Queries across all 4 domains (INCOTERMS, DGFT, HS Codes, WTO) and structures the output.
        """
        # Increase K to ensure we capture context across different domains if possible
        docs = self.retriever.invoke(user_query)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            chapter = doc.metadata.get("chapter", "")
            sources.append(f"{src} — {chapter} (Page {page})")
        source_str = "\n".join(set(sources))

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior international trade and customs compliance expert.
Your task is to provide an exhaustive, highly detailed, and professionally formatted analysis of the user's query using ONLY the provided context.

Context may include:
- INCOTERMS 2020 (ICC Rules)
- DGFT Foreign Trade Policy 2023
- ITC-HS Customs Tariff Codes
- WTO Trade Policy Reviews

INSTRUCTIONS FOR OUTPUT:
1. Your Direct Summary MUST be elaborate. Explain the complete scenario tying together the product, duty, shipping term, and regulatory policies into a cohesive 2-3 paragraph professional summary.
2. For EACH of the 4 domains, you MUST provide a detailed, comprehensive explanation (at least 3-4 sentences per domain). Do NOT provide just a single sentence.
3. Cite exact clauses, chapter numbers, duty percentages, and page numbers where available.
4. If a domain is not relevant, explicitly explain WHY it is not applicable based on the context, rather than just saying "Not applicable".
5. CRITICAL ANTI-HALLUCINATION RULE: If the retrieved context does not contain the specific answer to the user's query, you MUST abort the detailed structure and output EXACTLY and ONLY the phrase "I do not have the information regarding this in my knowledge base." Do NOT attempt to summarize unrelated context or guess.

RETRIEVED CONTEXT:
{context}

SOURCES:
{sources}"""),
            ("human", "{query}"),
        ])

        structured_llm = self.llm.with_structured_output(UnifiedQueryResponse)
        chain = prompt | structured_llm

        try:
            result = chain.invoke({
                "context": context,
                "query": user_query,
                "sources": source_str,
            })
            response = result.model_dump()
        except Exception as e:
            logger.error(f"Unified query structured output failed: {e}")
            fallback_chain = prompt | self.llm | StrOutputParser()
            text_result = fallback_chain.invoke({
                "context": context,
                "query": user_query,
                "sources": source_str,
            })
            import re
            
            # Check if it triggered the anti-hallucination rule
            if "I do not have the information regarding this in my knowledge base" in text_result:
                response = {
                    "description": "I do not have the information regarding this in my knowledge base.",
                    "incoterms_context": "Not applicable.",
                    "dgft_context": "Not applicable.",
                    "hs_code_context": "Not applicable.",
                    "wto_context": "Not applicable.",
                    "citations": []
                }
            else:
                # Helper to extract sections based on headers
                def extract_section(text, header):
                    match = re.search(rf"{header}.*?(?=(?:INCOTERMS 2020|ITC-HS Customs|DGFT Foreign|WTO Trade|$))", text, re.IGNORECASE | re.DOTALL)
                    return match.group(0).strip() if match else "Not explicitly mentioned in the unstructured response. See summary."

                incoterms = extract_section(text_result, "INCOTERMS 2020")
                hs_code = extract_section(text_result, "ITC-HS Customs")
                dgft = extract_section(text_result, "DGFT Foreign")
                wto = extract_section(text_result, "WTO Trade")
                
                # The summary is anything before the first major header
                summary_match = re.split(r"INCOTERMS 2020|ITC-HS Customs|DGFT Foreign|WTO Trade", text_result, maxsplit=1, flags=re.IGNORECASE)
                summary = summary_match[0].strip() if summary_match else text_result

                response = {
                    "description": summary if summary else "See detailed domains below.",
                    "incoterms_context": incoterms,
                    "dgft_context": dgft,
                    "hs_code_context": hs_code,
                    "wto_context": wto,
                    "citations": sources
                }
        response["retrieved_sources"] = [
            {
                "source": doc.metadata.get("source"),
                "chapter": doc.metadata.get("chapter"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:150],
            }
            for doc in docs
        ]

        return response


    # ─── Raw Retrieval (for RAGAS evaluation) ──────────
    def retrieve_context(self, query: str, k: int = None) -> list[Document]:
        """Retrieve raw documents for evaluation purposes."""
        if k is None:
            k = TOP_K_RETRIEVAL
            
        query_lower = query.lower()
        # If the query asks about duties or HS codes, explicitly fetch HS code chunks
        if any(term in query_lower for term in ["hs code", "customs duty", "import duty", "landed cost"]):
            # Strip generic conversational words to focus the embedding on the product
            clean_q = query.replace("What is the customs duty on importing ", "").replace(" into India?", "").replace("Calculate the total landed cost for importing ", "").replace(" into India.", "").replace("What is the import duty on ", "")
            
            general_docs = self.vectorstore.similarity_search(query, k=k)
            hs_docs = self.vectorstore.similarity_search(
                clean_q, 
                k=3, 
                filter={"doc_type": "hs_code"}
            )
            
            # Combine, putting the HS docs first for higher precision rank
            combined = hs_docs + general_docs
            seen = set()
            unique_docs = []
            for doc in combined:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
            return unique_docs[:k]
            
        return self.vectorstore.similarity_search(query, k=k)
