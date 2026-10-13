"""Shared routing question for the ERP chatbot.

The fine-tuned Laya checkpoint is trained on exactly this question and state shape, so every
script that trains, evaluates or serves the router must import them from here.
"""

LABELS = ("TEXT_TO_SQL", "RAG")

ROUTE_QUESTION = {
    "type": "choice",
    "instructions": (
        "Route an ERPNext chatbot question by the source of information needed to answer it. "
        "Focus on what the user wants, not on the business terms used."
    ),
    "criteria": {
        "TEXT_TO_SQL": (
            "Needs live ERP records: the user's own documents, specific records or IDs, "
            "amounts, totals, counts, balances, statuses, lists or reports."
        ),
        "RAG": (
            "Needs company documents: what a term means, how a process or feature works, "
            "how to do something in ERPNext, policies, rules and troubleshooting."
        ),
    },
}

QUESTIONS = {"route": ROUTE_QUESTION}


def make_state(query):
    return {"text": query}
