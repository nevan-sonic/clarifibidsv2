"""
Evidence Evaluator Module
=========================
Evaluates retrieved candidate chunks against the user's intent to determine if:
1. The evidence is statistically relevant (exceeds semantic similarity threshold).
2. The evidence is sufficient (contains substantive keywords answering the intent).
3. The RAG pipeline should PROCEED to answer generation, REFINE the search query,
   or flag the query as OUT_OF_DOMAIN / IRRELEVANT.
"""

from typing import List, Dict, Any
from app.core.config import settings


class EvidenceEvaluator:
    """
    Evaluates retrieval quality dynamically to guard against hallucination.
    Decides whether the retrieved documents provide enough factual backing
    before feeding them into the Large Language Model.
    """

    def __init__(
        self, 
        relevance_threshold: float = settings.RELEVANCE_THRESHOLD,
        sufficiency_threshold: float = settings.SUFFICIENCY_THRESHOLD
    ):
        self.relevance_threshold = relevance_threshold
        self.sufficiency_threshold = sufficiency_threshold
        
    def evaluate(
        self, 
        query: str, 
        intent: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        attempt_number: int = 1,
        is_out_of_domain: bool = False
    ) -> Dict[str, Any]:
        if is_out_of_domain:
            return {
                "is_relevant": False,
                "is_sufficient": False,
                "coverage_score": 0.0,
                "action": "OUT_OF_DOMAIN",
                "reason": "Query is outside the procedural domain of the GePNIC eProcurement portal."
            }

        if not retrieved_chunks:
            return {
                "is_relevant": False,
                "is_sufficient": False,
                "coverage_score": 0.0,
                "action": "IRRELEVANT_QUERY" if attempt_number >= settings.MAX_RETRIEVAL_ATTEMPTS else "REFINE",
                "reason": f"No candidate chunks satisfied the minimum relevance threshold ({self.relevance_threshold})."
            }
            
        top_score = retrieved_chunks[0].get("similarity_score", 0.0)
        is_relevant = top_score >= self.relevance_threshold
        
        # Step 3: Check Sufficiency (Query and Intent Overlap)
        query_words = set(w for w in query.lower().split() if len(w) > 3)
        intent_words = set(w for w in intent.lower().split() if len(w) > 3)
        check_words = query_words.union(intent_words)
        
        top_content = (
            str(retrieved_chunks[0].get("question") or "") + " " + 
            str(retrieved_chunks[0].get("answer") or "") + " " + 
            str(retrieved_chunks[0].get("content") or "")
        ).lower()
        
        overlap = sum(1 for w in check_words if w in top_content)
        intent_coverage = overlap / max(1, len(check_words))
        
        # High semantic similarity (>= 0.75) is sufficient on its own for direct queries
        if top_score >= 0.75:
            coverage_score = round(top_score, 4)
        else:
            coverage_score = round(0.7 * top_score + 0.3 * intent_coverage, 4)
            
        is_sufficient = is_relevant and (coverage_score >= self.sufficiency_threshold)
        
        if is_sufficient:
            action = "PROCEED"
            reason = f"Evidence meets relevance ({top_score:.2f} >= {self.relevance_threshold}) and sufficiency criteria."
        elif attempt_number < settings.MAX_RETRIEVAL_ATTEMPTS:
            action = "REFINE"
            reason = f"Evidence coverage ({coverage_score:.2f}) below threshold ({self.sufficiency_threshold}). Triggering corrective pass."
        else:
            action = "EXHAUSTED"
            reason = "Maximum retrieval attempts reached without reaching sufficiency threshold."
            
        return {
            "is_relevant": is_relevant,
            "is_sufficient": is_sufficient,
            "coverage_score": coverage_score,
            "action": action,
            "reason": reason
        }

evidence_evaluator = EvidenceEvaluator()
