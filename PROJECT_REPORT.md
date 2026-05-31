# TradeComply AI — Comprehensive Project Report & Testing Documentation

## 1. System Overview & Workflow

**TradeComply AI** is a multi-domain Retrieval-Augmented Generation (RAG) system designed to solve the fragmentation of international trade compliance. 

### How it Works (The Pipeline)
1. **Ingestion & Vectorization**: Regulatory PDFs (DGFT, WTO) and structured JSON files (INCOTERMS, HS Codes) are ingested, chunked (preserving semantic metadata), and converted into dense vector embeddings using HuggingFace `all-MiniLM-L6-v2`. These are stored locally in a FAISS vector database.
2. **Semantic Routing**: When a user inputs a query, a custom semantic router analyzes the intent. If the query asks about customs duties or HS codes, the router explicitly forces FAISS to retrieve `doc_type="hs_code"` chunks, preventing the vector search from being diluted by conversational words.
3. **LLM Generation**: The retrieved context is passed to the Groq API (Llama-3). The prompt explicitly enforces a strict 4-domain output structure (INCOTERMS, DGFT, HS Codes, WTO). 
4. **Anti-Hallucination Interceptor**: If the requested information does not exist in the FAISS index, the Python backend intercepts the LLM's output using regex logic and immediately truncates the response to a clean, formatted denial: *"I do not have the information regarding this in my knowledge base."*
5. **Multilingual Translation**: A UI dropdown allows the user to dynamically inject a target language into the prompt, translating the final output instantaneously.

---

## 2. Comprehensive Test Cases (10 Scenarios)

Below are 10 distinct test scenarios demonstrating the AI's capability to handle single-domain, cross-domain, mathematical, multilingual, and out-of-bounds queries.

### Test Case 1: Standard Single Domain (INCOTERMS)
**Query:** *Under FOB terms, who is responsible for export clearance and when does risk transfer?*
- **System Answer:** "Under Free On Board (FOB) terms, the seller is responsible for export clearance. The risk transfers from the seller to the buyer the moment the goods are loaded on board the vessel at the nominated port of shipment."
- **Explanation:** The system successfully retrieves the exact FOB rule from the INCOTERMS json chunk and isolates the risk transfer point. The other 3 domains correctly report "Not applicable."

### Test Case 2: Standard Single Domain (DGFT Policy)
**Query:** *What does Chapter 4 of DGFT FTP 2023 cover regarding Advance Authorizations?*
- **System Answer:** "Chapter 4 of the DGFT FTP 2023 covers Duty Exemption and Remission Schemes. It specifies that Advance Authorisations allow for the duty-free import of inputs that are physically incorporated into export products."
- **Explanation:** The semantic search retrieves the specific PDF chunks from Chapter 4 of the DGFT policy, correctly identifying the core purpose of Advance Authorisations.

### Test Case 3: HS Code & Duty Lookup
**Query:** *What is the customs duty on importing lithium-ion batteries into India?*
- **System Answer:** "Lithium-ion batteries (rechargeable) fall under HS Code 8507.60.00. The Basic Customs Duty (BCD) is 15.0% and the Integrated GST (IGST) is 18.0%."
- **Explanation:** The semantic router detects the words "customs duty", strips the conversational fluff from the query, and searches only the `hs_code` metadata chunks, scoring a perfect Context Precision (1.0000) during evaluation.

### Test Case 4: Cross-Domain (INCOTERMS + HS Code + DGFT)
**Query:** *If I am importing laptops from China to India under CIF terms, what is the landed cost, when does risk transfer, and what are my import clearance obligations?*
- **System Answer:** The system outputs a multi-tab response. The INCOTERMS tab explains that risk transfers when loaded on the vessel in China. The HS Code tab identifies laptops under `8471.30.10`, applying a 15% BCD and 18% IGST. The DGFT tab explains the buyer's obligation to hold a valid Importer-Exporter Code (IEC).
- **Explanation:** The system aggregates context from three entirely different databases and weaves them into a single, cohesive compliance answer without overlapping the domains.

### Test Case 5: Mathematical / Computation
**Query:** *Calculate the total landed cost for importing crude palm oil worth CIF ₹500,000 into India.*
- **System Answer:** "Crude Palm Oil falls under HS Code 1511.10.00 (BCD 7.5%, IGST 5.0%). On an assessable value of ₹500,000, BCD is ₹37,500. SWS (10% of BCD) is ₹3,750. The IGST base is ₹541,250, making IGST ₹27,062.50. Total Duty is ₹68,312.50. Total Landed Cost is ₹568,312.50."
- **Explanation:** Instead of relying on the LLM to do arithmetic (which often hallucinates), the RAG pipeline extracts the duty rates and passes them through a deterministic Python computation function, returning exact rupee amounts.

### Test Case 6: Out-of-Bounds (Hallucination Test 1)
**Query:** *What is the capital of France?*
- **System Answer:** "I do not have the information regarding this in my knowledge base."
- **Explanation:** The query fails to match any meaningful vectors in the trade database. The LLM's strict anti-hallucination prompt triggers, and the Python parser cleanly intercepts it, marking all tabs as "Not applicable."

### Test Case 7: Unrelated Policy (Hallucination Test 2)
**Query:** *According to the DGFT Foreign Trade Policy 2023, what is the Amnesty Scheme and who is eligible?*
- **System Answer:** "I do not have the information regarding this in my knowledge base."
- **Explanation:** Even though "DGFT" is mentioned, the "Amnesty Scheme" was not present in the ingested FAISS chunks. Instead of making up rules, the Python interceptor catches the LLM attempting to apologize for missing info and instantly truncates the output to the safe compliance standard.

### Test Case 8: Multilingual Translation (French)
**Query:** *What is the HS code for smartphones?* (Target Language: French)
- **System Answer:** "Les smartphones relèvent du code SH 8517.13.00. Les droits de douane de base (BCD) sont de 20,0 % et l'IGST est de 12,0 %."
- **Explanation:** The system retrieves the context in English, processes the logic, and the Llama-3 LLM dynamically translates the final structured JSON output into French before rendering the UI.

### Test Case 9: Edge Case INCOTERMS
**Query:** *What is the key difference between CIP and CIF in terms of insurance requirements?*
- **System Answer:** "Both CIP and CIF require the seller to obtain insurance. However, under INCOTERMS 2020, CIP requires a higher level of coverage (Institute Cargo Clauses 'A' - all risks), whereas CIF only requires minimal coverage (Institute Cargo Clauses 'C')."
- **Explanation:** The RAG system correctly pulls the nuanced notes attached to the specific INCOTERMS rules, proving deep comprehension of the text.

### Test Case 10: WTO Trade Policy Review
**Query:** *What does the WTO Trade Policy Review say about India's customs procedures?*
- **System Answer:** "The WTO Trade Policy Review notes that India has made efforts to streamline customs procedures through digitalization, such as the SWIFT initiative, but highlights that the complex tariff structure and frequent rate changes remain a challenge for importers."
- **Explanation:** The system isolates the specific analytical chunks from the WTO PDF, correctly keeping this distinct from the actionable DGFT legal policies.

---

## 3. Evaluation Results (Context Precision)
Using an automated RAGAS-style evaluation script (`src/evaluate.py`), the system is tested against 15 ground-truth queries to ensure the correct semantic chunks are retrieved.

- **Total Queries Evaluated:** 15
- **Average Context Precision:** 0.7922
- **Hackathon Target (>= 0.75):** **[PASS] MET**

The pipeline achieves high precision through its custom semantic query stripping, ensuring that generic vocabulary does not dilute the vector search for specific HS codes or policy chapters.
