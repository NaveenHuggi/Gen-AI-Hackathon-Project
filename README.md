# 🚢 TradeComply AI

**Intelligent Logistics Regulation & INCOTERMS Knowledge Base — Powered by RAG**

TradeComply AI is a powerful Retrieval-Augmented Generation (RAG) system built for export managers, freight forwarders, and customs teams. It seamlessly ingests and analyzes complex international trade compliance documents—including the INCOTERMS 2020 handbook, DGFT Foreign Trade Policy 2023, ITC-HS Customs Tariff Codes, and WTO Trade Policy Reviews—to provide exact regulatory answers and customs duty computations.

## ✨ Features

- **Unified Trade Query System:** Ask complex questions about trade scenarios, and the AI cross-references 4 independent regulatory databases simultaneously.
- **Customs Duty & HS Code Calculator:** Query an item in plain language (e.g., "lithium-ion batteries"), and the system identifies the matching HS code and calculates the exact BCD, SWS, and IGST components for the total landed cost.
- **Detailed Source Citations:** Every answer links directly back to the source document, complete with chapter headers and page numbers, enabling instant compliance verification.
- **Premium Glassmorphism UI:** Built with Streamlit and custom CSS for a beautiful, responsive, modern dark-themed user experience.

---

## 🛠️ Tech Stack

- **Frameworks:** LangChain, Streamlit
- **LLM Engine:** Groq API (High-speed Llama 3 models) / Google Gemini 3.1 Pro
- **Embeddings:** HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS
- **PDF Extraction:** `pdfplumber` with `PyPDF2` fallback

---

## 🚀 Quickstart Guide (Reproducibility)

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/NaveenHuggi/Gen-AI-Hackathon-Project.git
cd Gen-AI-Hackathon-Project
```

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to prevent dependency conflicts.
```bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy the `.env.example` file to create a `.env` file and fill in your API keys.
```bash
cp .env.example .env
```
*(You will need a valid `GROQ_API_KEY` for the LLM to function.)*

### 6. Build the FAISS Vector Index (Ingestion)
Before running the dashboard, the system needs to process the regulatory PDFs and build the vector database. Put your PDFs in `Docs/Icoterms` and `Docs/trade` or use the ones provided.
```bash
python src/ingest.py
```
*This script will parse the PDFs, chunk the text, compute embeddings, and save the index into the `faiss_index/` directory.*

### 7. Run the Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to interact with TradeComply AI.

---

## 📊 RAGAS Evaluation Pipeline (Optional)
This project includes an automated evaluation pipeline using RAGAS to ensure high Context Precision (≥ 0.75).
To run the evaluation on the 15 standard trade scenarios:
```bash
python src/evaluate.py
```
The results will be saved to `evaluation_results.json`.

---

## 📜 License
Developed for the Gen AI Hackathon 2026.
