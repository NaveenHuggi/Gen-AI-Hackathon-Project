"""
evaluate.py - RAGAS evaluation pipeline for the Trade Compliance RAG system.

Evaluates the system on 15 predefined trade scenario queries using
Context Precision from the RAGAS framework.
"""

import sys
import os
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_chain import TradeRAGPipeline
from config import GROQ_API_KEY, GROQ_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 15 Trade Scenario Evaluation Queries
# ═══════════════════════════════════════════════════════════

EVALUATION_QUERIES = [
    {
        "query": "Under CIF terms, at what point does risk transfer from seller to buyer, and what insurance coverage is the seller obligated to provide?",
        "expected_answer": "Under CIF, risk transfers when goods are placed on board the vessel at the port of shipment, not at the destination. The seller must provide minimum insurance under Institute Cargo Clauses (C).",
        "ground_truth_context_keywords": ["CIF", "risk transfer", "port of shipment", "on board", "Institute Cargo Clauses (C)", "insurance"],
    },
    {
        "query": "What is the key difference between CIP and CIF in terms of insurance requirements under INCOTERMS 2020?",
        "expected_answer": "CIP mandates Institute Cargo Clauses (A) — all-risk/maximum coverage. CIF only requires Institute Cargo Clauses (C) — minimum coverage. This distinction was a critical update in INCOTERMS 2020.",
        "ground_truth_context_keywords": ["CIP", "CIF", "Clause (A)", "Clause (C)", "insurance", "2020"],
    },
    {
        "query": "What was the INCOTERMS 2020 change from DAT and what does the new rule DPU mean?",
        "expected_answer": "DAT (Delivered at Terminal) was renamed to DPU (Delivered at Place Unloaded) in 2020. The change clarifies that delivery can occur at any designated place, not strictly a transport terminal. DPU is the only INCOTERM requiring the seller to unload goods.",
        "ground_truth_context_keywords": ["DAT", "DPU", "renamed", "terminal", "unloaded", "2020"],
    },
    {
        "query": "Under FOB terms, who is responsible for export clearance and when does risk transfer?",
        "expected_answer": "Under FOB, the seller is responsible for export clearance. Risk transfers to the buyer when goods are placed on board the vessel at the named port of shipment.",
        "ground_truth_context_keywords": ["FOB", "export clearance", "seller", "on board", "port of shipment", "risk"],
    },
    {
        "query": "What is the HS code for smartphones and what customs duties apply in India?",
        "expected_answer": "Smartphones fall under HS Code 8517.13.00 with a Basic Customs Duty of 20% and IGST of 12%.",
        "ground_truth_context_keywords": ["8517", "smartphone", "mobile", "BCD", "20.0%", "IGST", "12.0%"],
    },
    {
        "query": "What is the customs duty on importing lithium-ion batteries into India?",
        "expected_answer": "Lithium-ion rechargeable batteries fall under HS Code 8507.60.00 with BCD of 15% and IGST of 18%.",
        "ground_truth_context_keywords": ["8507", "lithium-ion", "battery", "15.0%", "18.0%"],
    },
    {
        "query": "Calculate the total landed cost for importing laptops worth CIF ₹100,000 into India.",
        "expected_answer": "Laptops fall under HS Code 8471.30.10 with BCD 15% and IGST 18%. BCD=₹15,000, SWS=₹1,500, IGST base=₹116,500, IGST=₹20,970. Total duty≈₹37,470. Landed cost≈₹137,470.",
        "ground_truth_context_keywords": ["8471", "laptop", "15.0%", "BCD", "IGST", "18.0%"],
    },
    {
        "query": "What are the conditions for conversion of a domestic unit into an Export Oriented Unit under DGFT FTP 2023?",
        "expected_answer": "Under Chapter 6 of DGFT FTP 2023, units with investment of Rs. 50 crores and above in plant and machinery, or exporting Rs. 50 crores and above annually, shall be placed before the Board of Approval (BOA) for a decision.",
        "ground_truth_context_keywords": ["Chapter 6", "EOU", "50 crore", "Approval"],
    },
    {
        "query": "What does Chapter 4 of DGFT FTP 2023 cover regarding Advance Authorizations?",
        "expected_answer": "Chapter 4 covers Duty Exemption Remission Schemes, including Advance Authorizations which allow duty-free import of inputs for export production.",
        "ground_truth_context_keywords": ["Chapter 4", "Exemption", "Advance Authorisation", "duty"],
    },
    {
        "query": "Under INCOTERMS 2020 EXW, who bears the risk and cost of export clearance?",
        "expected_answer": "Under EXW, the buyer bears the risk and cost of export clearance. The seller's only obligation is to make goods available at their premises. EXW is not recommended for international trade due to the complexity of export clearance for the buyer.",
        "ground_truth_context_keywords": ["EXW", "buyer", "export clearance", "risk", "cost", "seller's premises"],
    },
    {
        "query": "What is the difference between DAP and DPU in terms of unloading obligation?",
        "expected_answer": "Under DAP, goods are delivered ready for unloading but the buyer unloads. Under DPU, the seller must unload goods at the destination. DPU is the only INCOTERM where the seller has an unloading obligation.",
        "ground_truth_context_keywords": ["DAP", "DPU", "unloading", "seller", "buyer", "destination"],
    },
    {
        "query": "What customs duty applies to importing crude palm oil into India?",
        "expected_answer": "Crude palm oil falls under HS Code 1511.10.00 with a Basic Customs Duty of 7.5% and IGST of 5%.",
        "ground_truth_context_keywords": ["1511", "palm oil", "7.5%", "5.0%", "crude"],
    },
    {
        "query": "What is the import duty on gold bullion in India?",
        "expected_answer": "Gold in unwrought forms/bullion falls under HS Code 7108.12.00 with BCD of 6% and IGST of 3%.",
        "ground_truth_context_keywords": ["7108", "gold", "bullion", "6.0%", "3.0%"],
    },
    {
        "query": "Under CPT terms, explain the critical distinction between where risk transfers and where cost obligation ends.",
        "expected_answer": "Under CPT, risk transfers to the buyer when goods are handed to the first carrier at the place of shipment. However, the seller's cost obligation extends to the named destination — the seller pays freight to the destination. This means risk and cost transfer at DIFFERENT points.",
        "ground_truth_context_keywords": ["CPT", "risk", "first carrier", "cost", "destination", "different points"],
    },
    {
        "query": "What is the general provision regarding imports and exports under Chapter 2 of DGFT FTP 2023?",
        "expected_answer": "Chapter 2 outlines general provisions noting that trade is generally 'Free' unless specifically regulated by the government.",
        "ground_truth_context_keywords": ["Chapter 2", "general provisions", "Free", "regulated", "imports", "exports"],
    },
]


# ═══════════════════════════════════════════════════════════
# Evaluation Functions
# ═══════════════════════════════════════════════════════════

def evaluate_context_precision_manual(pipeline: TradeRAGPipeline) -> dict:
    """
    Evaluate context precision manually by checking if retrieved chunks
    contain the expected keywords from ground truth.
    
    This is a lightweight, deterministic evaluation that doesn't require
    an additional LLM call for judging relevance.
    """
    results = []
    total_precision = 0.0

    for i, scenario in enumerate(EVALUATION_QUERIES, 1):
        query = scenario["query"]
        keywords = scenario["ground_truth_context_keywords"]

        logger.info(f"Evaluating query {i}/{len(EVALUATION_QUERIES)}: {query[:80]}...")

        # Retrieve context (using updated pipeline with HS code routing)
        retrieved_docs = pipeline.retrieve_context(query, k=6)

        # Calculate precision at each rank
        relevance_flags = []
        for doc in retrieved_docs:
            content = doc.page_content.lower()
            # A chunk is relevant if it contains at least 2 ground truth keywords
            keyword_hits = sum(1 for kw in keywords if kw.lower() in content)
            is_relevant = keyword_hits >= 2
            relevance_flags.append(is_relevant)

        # Compute Context Precision (mean of Precision@k for relevant chunks)
        precision_at_k_values = []
        relevant_count = 0
        for k, is_rel in enumerate(relevance_flags, 1):
            if is_rel:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precision_at_k_values.append(precision_at_k)

        context_precision = (
            sum(precision_at_k_values) / len(precision_at_k_values)
            if precision_at_k_values
            else 0.0
        )

        results.append({
            "query_index": i,
            "query": query[:100],
            "context_precision": round(context_precision, 4),
            "relevant_chunks": sum(relevance_flags),
            "total_chunks": len(relevance_flags),
            "relevance_pattern": relevance_flags,
        })

        total_precision += context_precision
        logger.info(f"  → Context Precision: {context_precision:.4f} ({sum(relevance_flags)}/{len(relevance_flags)} relevant)")

    avg_precision = total_precision / len(EVALUATION_QUERIES)

    return {
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(EVALUATION_QUERIES),
        "average_context_precision": round(avg_precision, 4),
        "target_precision": 0.75,
        "target_met": avg_precision >= 0.75,
        "per_query_results": results,
    }


def run_evaluation():
    """Run the full evaluation pipeline and save results."""
    logger.info("=" * 60)
    logger.info("Starting RAGAS-style Context Precision Evaluation")
    logger.info("=" * 60)

    pipeline = TradeRAGPipeline()
    results = evaluate_context_precision_manual(pipeline)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluation_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total Queries Evaluated:      {results['num_queries']}")
    print(f"  Average Context Precision:    {results['average_context_precision']:.4f}")
    print(f"  Target (>= 0.75):             {'[PASS] MET' if results['target_met'] else '[FAIL] NOT MET'}")
    print("=" * 60)

    for r in results["per_query_results"]:
        status = "[PASS]" if r["context_precision"] >= 0.75 else "[FAIL]"
        print(f"  {status} Q{r['query_index']:2d}: CP={r['context_precision']:.4f} | {r['relevant_chunks']}/{r['total_chunks']} relevant | {r['query'][:60]}...")

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_evaluation()
