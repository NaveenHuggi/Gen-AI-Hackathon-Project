"""
app.py — Streamlit Dashboard for the Logistics Regulation & INCOTERMS RAG Knowledge Base.

Features:
  - Regulatory Query Interface (INCOTERMS + Trade Policy)
  - HS Code & Customs Duty Lookup
  - Premium dark-themed UI with glassmorphism
"""

import sys
import os
import streamlit as st

# Add src directory to path for Streamlit runtime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from src.rag_chain import TradeRAGPipeline, compute_customs_duty
except ImportError:
    # Fallback for UI testing if src.rag_chain is not found or fails
    TradeRAGPipeline = None


# ═══════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TradeComply AI | Intelligent Logistics",
    page_icon="🛳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# Custom CSS — Premium Dark Theme with Glassmorphism
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ─── Import Google Font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* ─── Global ─── */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: radial-gradient(circle at top left, #0e122b 0%, #060913 100%);
        color: #e2e8f0;
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 36, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* ─── Header Hero ─── */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 50%);
        z-index: 0;
        pointer-events: none;
    }

    .hero-content {
        position: relative;
        z-index: 1;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-title-text {
        background: linear-gradient(135deg, #c7d2fe, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.6;
        max-width: 800px;
    }

    /* ─── Cards & Containers ─── */
    .glass-panel {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .domain-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.5));
        border-left: 4px solid #6366f1;
        border-top: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        border-radius: 4px 16px 16px 4px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .domain-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }

    .domain-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #818cf8;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .domain-content {
        font-size: 1.05rem;
        color: #f1f5f9;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    /* ─── Buttons ─── */
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* ─── Inputs ─── */
    .stTextArea > div > div > textarea {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-size: 1.05rem !important;
        padding: 1rem !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        padding: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        color: #94a3b8;
        font-weight: 500;
        padding: 1rem 0.5rem;
        margin-right: 1rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #e2e8f0 !important;
        border-bottom: 2px solid #6366f1 !important;
    }

    /* ─── Status Tags ─── */
    .status-tag {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Hide top padding */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Initialize Pipeline (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached across reruns)."""
    if TradeRAGPipeline is None:
        return None
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
# Sidebar Configuration
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚢 TradeComply AI")
    st.caption("Intelligent Logistics Assistant v1.0")
    
    st.divider()
    
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 10px;'>
        <div style='display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2em;'>📘</span> INCOTERMS 2020</div>
    </div>
    <div style='background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 10px;'>
        <div style='display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2em;'>📋</span> DGFT Foreign Trade Policy</div>
    </div>
    <div style='background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 10px;'>
        <div style='display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2em;'>📊</span> ITC-HS Customs Tariff</div>
    </div>
    <div style='background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 10px;'>
        <div style='display: flex; align-items: center; gap: 8px;'><span style='font-size: 1.2em;'>🌐</span> WTO Trade Policy Reviews</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### ⚙️ Preferences")
    target_language = st.selectbox(
        "Response Language", 
        ["English", "Hindi", "French", "Spanish", "Mandarin"], 
        index=0,
    )
    
    st.divider()
    
    st.markdown("""
    <div style='font-size: 0.8rem; color: #64748b; text-align: center;'>
        Powered by <strong>LangChain + FAISS</strong><br>
        LLM: <em>Llama 3.3 70B</em>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Hero Section
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <div class="hero-title">Welcome to <span class="hero-title-text">TradeComply AI</span> 🚢</div>
        <div class="hero-subtitle">
            Navigate global trade complexities with confidence. Query trade compliance rules, compute customs duties, and instantly resolve INCOTERMS disputes with AI precision.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Helper Functions for Rendering
# ═══════════════════════════════════════════════════════════
def render_domain_card(icon: str, label: str, value: str):
    """Render a premium domain context card."""
    import re
    # Strip raw document section headers that may have leaked through
    value = re.sub(r'\b([AB]\d{1,2})\s+(DELIVERY|TRANSFER OF RISKS|NOTICE|INSURANCE|COSTS|CARRIAGE)[:\s]', '', value)
    
    is_empty = not value or value.strip().lower() in ("not applicable", "not applicable.", "")
    
    st.markdown(f"""
    <div class="domain-card">
        <div class="domain-title">{icon} {label}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if is_empty:
        st.markdown("<span style='color: #64748b; font-style: italic;'>No specific context found in this domain.</span>", unsafe_allow_html=True)
    else:
        st.markdown(value)


# ═══════════════════════════════════════════════════════════
# Main Content — Query Interface
# ═══════════════════════════════════════════════════════════
pipeline = load_pipeline()

st.markdown("### 🔍 Unified Trade Query")
st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;'>Describe your trade scenario, product, or compliance question.</p>", unsafe_allow_html=True)

# Using a form for better UX
with st.form(key="query_form", border=False):
    unified_query = st.text_area(
        "Query",
        placeholder="e.g. Under CIF terms, at what point does risk transfer from seller to buyer? Or, What is the HS code for smartphones and what are the duties?",
        height=140,
        label_visibility="collapsed"
    )
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        submit_button = st.form_submit_button("Analyze Workflow ✨", use_container_width=True)

# ═══════════════════════════════════════════════════════════
# Results Rendering
# ═══════════════════════════════════════════════════════════
if submit_button:
    if not unified_query.strip():
        st.warning("⚠️ Please enter a query to analyze.")
    else:
        if pipeline is None:
            st.error("Pipeline not initialized. (Mocking result for demonstration)")
            # Mocking logic if pipeline not loaded
            import time
            with st.spinner("Analyzing across knowledge bases..."):
                time.sleep(2)
                result = {
                    "description": "Mock summary answer for your query.",
                    "incoterms_context": "Risk transfers at port of destination...",
                    "dgft_context": "Not applicable",
                    "hs_code_context": "HS Code 8517.12.00",
                    "wto_context": "Compliant with WTO agreements",
                    "citations": ["INCOTERMS 2020, Article A4"],
                    "retrieved_sources": [
                        {"source": "INCOTERMS_2020.pdf", "chapter": "CIF", "page": "24", "preview": "The seller delivers the goods..."}
                    ]
                }
        else:
            with st.spinner(f"Analyzing across all knowledge bases in {target_language}..."):
                result = pipeline.unified_trade_query(unified_query, target_language=target_language)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results Header
        col_res, col_lang = st.columns([3, 1])
        with col_res:
            st.markdown("### 📑 Comprehensive Analysis")
        with col_lang:
            st.markdown(
                f'<div style="text-align: right;"><span class="status-tag">🌐 {target_language}</span></div>',
                unsafe_allow_html=True
            )
            
        st.divider()
        
        # ── Front-end sanitizer: strip raw doc section labels from description
        import re
        description = result.get("description", "N/A")
        description = re.sub(r'\b([AB]\d{1,2})\s+(DELIVERY|TRANSFER OF RISKS|NOTICE|INSURANCE|COSTS|CARRIAGE)[:\s]', '', description)
        # Convert bullet chars to proper markdown
        description = description.replace("\u2022 ", "\n- ").replace("\u2022", "\n- ")

        # Summary Box
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1.5rem 1.5rem 0.5rem 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-weight: 700; color: #a5b4fc; margin-bottom: 1rem; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px;">📋 Direct Answer / Summary</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(description)
        
        # Tabs for detailed domain contexts
        tab1, tab2, tab3, tab4 = st.tabs(["📘 INCOTERMS", "📋 DGFT Policy", "📊 HS Code & Duty", "🌐 WTO Policy"])
        
        with tab1:
            render_domain_card("📘", "INCOTERMS 2020", result.get("incoterms_context", "Not applicable"))
        with tab2:
            render_domain_card("📋", "DGFT Foreign Trade Policy", result.get("dgft_context", "Not applicable"))
        with tab3:
            render_domain_card("📊", "HS Code & Customs Duty", result.get("hs_code_context", "Not applicable"))
        with tab4:
            render_domain_card("🌐", "WTO Trade Policy", result.get("wto_context", "Not applicable"))
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Citations & Sources Section
        st.markdown("### 📎 Evidence & Citations")
        
        if result.get("citations"):
            st.markdown("**Key Citations:**")
            for cit in result["citations"]:
                st.markdown(f"- 📄 `{cit}`")
        
        if result.get("retrieved_sources"):
            with st.expander("View Source Document Excerpts", expanded=False):
                for i, src in enumerate(result["retrieved_sources"], 1):
                    with st.container(border=True):
                        st.markdown(f"**Source {i}:** `{src.get('source', 'Unknown')}`")
                        
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            if src.get('chapter'):
                                st.caption(f"**Chapter:** {src.get('chapter')}")
                        with col_meta2:
                            if src.get('page') and src.get('page') != "N/A":
                                st.caption(f"**Page:** {src.get('page')}")
                                
                        preview_text = src.get("preview", "").strip()
                        if preview_text:
                            st.info(f"_{preview_text}_")
                        else:
                            st.caption("No preview text available.")


# ═══════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.9rem; padding: 1rem;'>
    <strong>TradeComply AI</strong> — Logistics Regulation & INCOTERMS Knowledge Base<br>
    <span style='font-size: 0.8rem; margin-top: 5px; display: inline-block;'>Built with ❤️ for Gen AI Hackathon 2026</span>
</div>
""", unsafe_allow_html=True)
