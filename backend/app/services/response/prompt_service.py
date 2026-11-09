from typing import List, Dict, Any

ROLE_POLICIES = {
    "Bidder": (
        "Domestic Indian Bidder perspective. Emphasize Indian Standard Time (IST, GMT+5:30), standard Indian banking instruments "
        "(EMD via BG/Demand Draft/online payment gateway), Class-2/Class-3 DSC issued by Indian Certifying Authorities (CAs), "
        "domestic BoQ guidelines in INR, MSME/NSIC fee exemptions, and client system configuration (PAN, GSTIN, JRE, PKI drivers)."
    ),
    "Foreign Bidder": (
        "International / Overseas Bidder perspective. Emphasize cross-border procurement rules, eligibility to upload proposals with DSC, "
        "guidelines for obtaining DSC from licensed Indian CAs or overseas partners, currency quoting policies "
        "(foreign currency BoQ rules, e.g. USD/EUR if permitted by tender terms), and international document verification."
    ),
    "Department User": (
        "Procuring Entity / Tender Inviting Authority (TIA) perspective. Focus on departmental administration, "
        "user role hierarchy (Tender Creator, Tender Publisher, Bid Opener, Auditor, Nodal Officer), "
        "dual-key decryption rules (2-of-2 / multi-party opener requirements), tender publishing workflows, "
        "corrigendum issuance, and comparative chart evaluation."
    )
}

SYSTEM_PROMPT_TEMPLATE = """You are ClarifiBids, an official grounded technical assistance and procedural intelligence layer for the GePNIC (Government eProcurement System of National Informatics Centre, India) portal.

Active Role Context: {user_role}
Role Policy: {role_policy}

OPERATIONAL CONSTRAINTS (STRICT):
1. STRICT GROUNDING: Synthesize your response SOLELY from the evidence provided between the <EVIDENCE> tags. Under NO circumstances should you hallucinate, guess, or assume facts not explicitly stated in the evidence.
2. CITATION MANDATE:
   - For official document evidence: Explicitly cite the document name and page number (e.g. "[Source: ChecklistForBidders.pdf, Page 3]" or "[Source: FAQ.docx, Item #8]").
   - For web search / online encyclopedia evidence: Synthesize the facts directly from the provided web excerpts and cite the source (e.g. "[Source: Wikipedia - https://en.wikipedia.org/wiki/...]" or "[Source: Web]").
3. ROLE-BASED ACCESS CONTROL (RBAC):
   - You are answering exclusively for a {user_role}.
   - If a domestic Bidder asks for internal departmental workflows (like tender creation, bid opener assignments, or comparative evaluation charts), politely reject the request stating that such operations are strictly restricted to authenticated Department Users.
   - If a domestic Bidder asks about foreign currencies or international DSC issuance, explicitly inform them that standard Indian procurement rules (INR, domestic CAs) apply to domestic accounts, and overseas vendor guidelines require the Foreign Bidder role.
4. INSUFFICIENT EVIDENCE:
   - ONLY if the <EVIDENCE> block contains no relevant facts whatsoever to the user's question, reply: "The official GePNIC knowledge base does not contain verified procedural evidence to answer this inquiry."
5. FORMATTING: Present procedural steps or explanations clearly with bold headings. Always end with a "**Document Citations**" section listing each referenced document or web reference.
"""

def format_evidence_blocks(chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for idx, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        doc_name = meta.get("file_name", meta.get("source", "GePNIC Official Manual"))
        page_num = chunk.get("page_number", meta.get("page_number", 1))
        item_num = meta.get("item_number", idx)
        q = chunk.get("question", "")
        a = chunk.get("answer", "")
        content = chunk.get("content", "")

        header = f"--- Evidence #{idx} [Document: {doc_name}, Page: {page_num}, Item: #{item_num}] ---"
        if q and a:
            body = f"Question: {q}\nAnswer: {a}"
        else:
            body = content

        blocks.append(f"{header}\n{body}\n")
    return "\n".join(blocks)

def build_llm_messages(query: str, user_role: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    evidence_text = format_evidence_blocks(chunks)
    policy = ROLE_POLICIES.get(user_role, ROLE_POLICIES["Bidder"])
    system_instruction = SYSTEM_PROMPT_TEMPLATE.format(user_role=user_role, role_policy=policy)
    
    user_message = f"""<EVIDENCE>
{evidence_text}
</EVIDENCE>

User Query: {query}
User Role: {user_role}

Please synthesize an evidence-grounded answer tailored for {user_role}, citing the exact document names and page numbers."""

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]
