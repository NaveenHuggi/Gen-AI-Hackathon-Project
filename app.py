"""
app.py — Streamlit Dashboard for the Logistics Regulation & INCOTERMS RAG Knowledge Base.

Features:
  - Regulatory Query Interface (INCOTERMS + Trade Policy)
  - HS Code & Customs Duty Lookup
  - Premium dark-themed UI with glassmorphism
"""

import sys
import os
import json
import streamlit as st

# Add src directory to path for Streamlit runtime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.rag_chain import TradeRAGPipeline, compute_customs_duty

# ═══════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TradeComply AI — Logistics RAG Knowledge Base",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# Custom CSS — Premium Dark Theme with Glassmorphism
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ─── Import Google Font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Global ─── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1040 30%, #0d1b2a 60%, #0a0e27 100%);
        color: #e0e6f0;
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 20, 50, 0.95) 0%, rgba(10, 14, 39, 0.98) 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc !important;
    }

    /* ─── Header Hero ─── */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 50%, rgba(59, 130, 246, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-title-text {
        background: linear-gradient(135deg, #a5b4fc, #818cf8, #6366f1, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.6;
    }

    /* ─── Glass Cards ─── */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(16px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }

    /* ─── Result Cards ─── */
    .result-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a5b4fc;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-field {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(71, 85, 105, 0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .field-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .field-value {
        font-size: 0.95rem;
        color: #e2e8f0;
        line-height: 1.6;
    }

    /* ─── Duty Table ─── */
    .duty-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .duty-table th {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 0.8rem 1rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .duty-table td {
        background: rgba(15, 23, 42, 0.4);
        padding: 0.7rem 1rem;
        border-top: 1px solid rgba(71, 85, 105, 0.2);
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    .duty-table tr:last-child td {
        background: rgba(99, 102, 241, 0.1);
        font-weight: 700;
        color: #a5b4fc;
    }

    /* ─── Badges ─── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-high { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-medium { background: rgba(234, 179, 8, 0.2); color: #fbbf24; border: 1px solid rgba(234, 179, 8, 0.3); }
    .badge-low { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* ─── Tabs styling ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #a5b4fc !important;
        border-bottom-color: #6366f1 !important;
    }

    /* ─── Buttons ─── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
    }

    /* ─── Input fields ─── */
    .stTextInput > div > div {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(71, 85, 105, 0.4) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stTextInput > div > div:focus-within {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
    }
    .stTextArea > div > div {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(71, 85, 105, 0.4) !important;
        border-radius: 10px !important;
    }
    .stNumberInput > div > div {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(71, 85, 105, 0.4) !important;
        border-radius: 10px !important;
    }

    /* ─── Expander ─── */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 10px !important;
        color: #a5b4fc !important;
    }

    /* ─── Metrics ─── */
    [data-testid="stMetricValue"] {
        color: #a5b4fc !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.4); border-radius: 10px; }

    /* ─── Source citation ─── */
    .source-chip {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #93c5fd;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin: 0.15rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Initialize Pipeline (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached across reruns)."""
    try:
        return TradeRAGPipeline()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.info("Please run `python src/ingest.py` from the project root to build the FAISS index first.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.stop()


# ═══════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚢 TradeComply AI")
    st.markdown("---")
    st.markdown("""
    **Knowledge Sources:**
    - 📘 INCOTERMS 2020 (ICC Rules)
    - 📋 DGFT Foreign Trade Policy 2023
    - 📊 ITC-HS Customs Tariff Codes
    - 🌐 WTO Trade Policy Reviews
    """)
    st.markdown("---")
    st.markdown("""
    **How to Use:**
    1. Enter a query 
    2. Enter your trade compliance question
    3. Get precise, citation-backed answers
    """)
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #64748b;'>
    Powered by LangChain + Groq + FAISS<br>
    Embeddings: all-MiniLM-L6-v2<br>
    Model: Llama 3.3 70B
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Hero Header
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🚢 <span class="hero-title-text">TradeComply AI</span></div>
    <div class="hero-subtitle">
        Intelligent Logistics Regulation & INCOTERMS Knowledge Base — Powered by RAG.<br>
        Query trade compliance rules, compute customs duties, and navigate HS codes with AI precision.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Load pipeline
# ═══════════════════════════════════════════════════════════
pipeline = load_pipeline()


# ═══════════════════════════════════════════════════════════
# Helper Functions for Rendering
# ═══════════════════════════════════════════════════════════

def render_field(label: str, value: str):
    """Render a styled field card."""
    st.markdown(f"""
    <div class="result-field">
        <div class="field-label">{label}</div>
        <div class="field-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_detailed_field(label: str, value: str):
    """Render a full-width styled card for detailed text."""
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.4); border-left: 4px solid #6366f1; border-radius: 4px 10px 10px 4px; padding: 1.2rem; margin-bottom: 1rem;">
        <div style="font-size: 0.9rem; font-weight: 700; color: #818cf8; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;">{label}</div>
        <div style="font-size: 1rem; color: #e2e8f0; line-height: 1.7; white-space: pre-wrap;">{value}</div>
    </div>
    """, unsafe_allow_html=True)



def render_badge(confidence: str) -> str:
    """Return HTML for a confidence badge."""
    level = confidence.lower() if confidence else "low"
    css_class = f"badge-{level}" if level in ("high", "medium", "low") else "badge-medium"
    return f'<span class="badge {css_class}">{confidence}</span>'


def render_duty_table(duty: dict):
    """Render the customs duty computation as a styled table."""
    st.markdown(f"""
    <table class="duty-table">
        <tr>
            <th>Component</th>
            <th>Rate / Base</th>
            <th>Amount (₹)</th>
        </tr>
        <tr>
            <td>Assessable Value (CIF)</td>
            <td>—</td>
            <td>₹{duty['assessable_value']:,.2f}</td>
        </tr>
        <tr>
            <td>Basic Customs Duty (BCD)</td>
            <td>{duty['bcd_rate']}</td>
            <td>₹{duty['bcd_amount']:,.2f}</td>
        </tr>
        <tr>
            <td>Social Welfare Surcharge (SWS)</td>
            <td>{duty['sws_rate']}</td>
            <td>₹{duty['sws_amount']:,.2f}</td>
        </tr>
        <tr>
            <td>IGST Base</td>
            <td>AV + BCD + SWS</td>
            <td>₹{duty['igst_base']:,.2f}</td>
        </tr>
        <tr>
            <td>Integrated GST (IGST)</td>
            <td>{duty['igst_rate']}</td>
            <td>₹{duty['igst_amount']:,.2f}</td>
        </tr>
        <tr>
            <td><strong>Total Customs Duty</strong></td>
            <td>—</td>
            <td><strong>₹{duty['total_duty']:,.2f}</strong></td>
        </tr>
        <tr>
            <td><strong>Total Landed Cost</strong></td>
            <td>—</td>
            <td><strong>₹{duty['total_landed_cost']:,.2f}</strong></td>
        </tr>
    </table>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Main Content — Unified Query Interface
# ═══════════════════════════════════════════════════════════

st.markdown("### Unified Trade Compliance Query")
st.markdown("Enter your query below. The system will search across INCOTERMS, DGFT, HS Codes, and WTO policies to provide a comprehensive, structured response.")

unified_query = st.text_area(
    "Enter your trade query or product description:",
    placeholder="e.g., Under CIF terms, at what point does risk transfer from seller to buyer? Or, What is the HS code for smartphones and what are the duties?",
    height=120,
    key="unified_input",
)

col1, col2, col3 = st.columns([2, 3, 5])
with col1:
    unified_submit = st.button("🔍 Analyze", key="unified_submit", use_container_width=True)
with col2:
    target_language = st.selectbox(
        "Response Language", 
        ["English", "Hindi", "French", "Spanish", "Mandarin"], 
        index=0, 
        label_visibility="collapsed"
    )

if unified_submit and unified_query:
    with st.spinner(f"Analyzing across all knowledge bases in {target_language}..."):
        result = pipeline.unified_trade_query(unified_query, target_language=target_language)

    st.markdown('<div class="result-header" style="font-size: 1.5rem; border-bottom: 2px solid rgba(99, 102, 241, 0.5); padding-bottom: 0.5rem; margin-bottom: 1.5rem;">📑 Comprehensive Analysis</div>', unsafe_allow_html=True)
    
    # Description
    st.markdown("#### Direct Answer / Summary")
    st.info(result.get("description", "N/A"))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4 Domains (Stacked full-width for detailed text)
    st.markdown("#### Detailed Domain Context")
    render_detailed_field("📘 INCOTERMS 2020", result.get("incoterms_context", "Not applicable"))
    render_detailed_field("📋 DGFT Foreign Trade Policy", result.get("dgft_context", "Not applicable"))
    render_detailed_field("📊 HS Code & Customs Duty", result.get("hs_code_context", "Not applicable"))
    render_detailed_field("🌐 WTO Trade Policy", result.get("wto_context", "Not applicable"))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Citations
    if result.get("citations"):
        st.markdown("#### 📎 View Retrieved Source Documents")
        with st.expander("Click to view full citations"):
            for cit in result["citations"]:
                st.markdown(f"> {cit}")
                
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="display: inline-block; padding: 0.4rem 1rem; background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 20px; color: #a5b4fc; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.5px;">🌐 Response generated in: {target_language}</div>',
        unsafe_allow_html=True
    )
            
    st.markdown('<br>', unsafe_allow_html=True)

    # Source documents expander
    if result.get("retrieved_sources"):
        with st.expander("📎 View Retrieved Source Documents", expanded=False):
            for i, src in enumerate(result["retrieved_sources"], 1):
                with st.container(border=True):
                    st.markdown(f"##### 📄 Source {i}: {src.get('source', 'Unknown')}")
                    
                    # Metadata row using columns
                    col_meta1, col_meta2 = st.columns(2)
                    with col_meta1:
                        if src.get('chapter'):
                            st.caption(f"**Chapter:** {src.get('chapter')}")
                    with col_meta2:
                        if src.get('page') and src.get('page') != "N/A":
                            st.caption(f"**Page:** {src.get('page')}")
                            
                    st.divider()
                    
                    # Preview text as a blockquote
                    preview_text = src.get("preview", "").strip()
                    if preview_text:
                        st.markdown(f"> {preview_text}")
                    else:
                        st.caption("No preview text available.")
                        
elif unified_submit:
    st.warning("Please enter a query.")


# ═══════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #475569; font-size: 0.8rem; padding: 1rem;'>
    <strong>TradeComply AI</strong> — Logistics Regulation & INCOTERMS RAG Knowledge Base<br>
    Built with LangChain • FAISS • Groq • Sentence Transformers • Streamlit<br>
    <em>Gen AI Hackathon 2026</em>
</div>
""", unsafe_allow_html=True)
