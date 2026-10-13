#!/usr/bin/env python3
"""
Hybrid ERP chatbot: the fine-tuned Laya router sends each question to Text-to-SQL or RAG,
and Gemini answers it.

  pip install google-genai
  set GEMINI_API_KEY=...          (PowerShell: $env:GEMINI_API_KEY="...")
  python chatbot.py

Commands: /sql <question> and /rag <question> force a branch; exit or quit leaves.
"""
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "erp_dataset" / "erp.db"
SCHEMA_PATH = ROOT / "erp_dataset" / "schema.sql"
DOCS_DIR = ROOT / "erp_dataset" / "docs"
INDEX_PATH = ROOT / "erp_dataset" / "rag_index.npz"

DB_TODAY = "2026-10-11"  # TODAY in generate_erp_dataset.py; statuses in erp.db are relative to it
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = "gemini-embedding-001"
MAX_ROWS = 50
TOP_K = 4


# ---------------------------------------------------------------- router

def load_router():
    import laya

    from router_question import QUESTIONS, make_state

    model = next((p for p in ("checkpoints/laya-erp-router", "checkpoints/laya-erp-router-head")
                  if (ROOT / p).is_dir()), "convaiinnovations/laya")
    print(f"Loading Laya router from {model} ...")
    agent = laya.load(str(ROOT / model) if model.startswith("checkpoints") else model)

    def route(question):
        answer = agent.predict(make_state(question), QUESTIONS)["answers"]["route"]
        return answer["choice"], answer.get("confidence") or 0.0

    return route


# ---------------------------------------------------------------- gemini

def make_client():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set. Get a key at https://aistudio.google.com/apikey and set it first.")
    from google import genai
    return genai.Client(api_key=key)


def generate(client, prompt, system, as_json=False):
    from google.genai import types
    config = types.GenerateContentConfig(
        system_instruction=system, temperature=0,
        response_mime_type="application/json" if as_json else "text/plain")
    text = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config).text or ""
    return json.loads(text) if as_json else text.strip()


def embed(client, texts, task_type):
    from google.genai import types
    vectors = []
    for i in range(0, len(texts), 100):
        resp = client.models.embed_content(model=EMBED_MODEL, contents=texts[i:i + 100],
                                           config=types.EmbedContentConfig(task_type=task_type))
        vectors.extend(e.values for e in resp.embeddings)
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- text-to-SQL

SQL_SYSTEM = f"""You translate questions about an ERP system into one SQLite query.
Rules:
- Return JSON: {{"sql": "<one SELECT or WITH query>"}}. No other statements, no comments.
- Use only the tables and columns in the schema. Today's date is {DB_TODAY}; dates are ISO text.
- Prefer readable columns (names rather than ids), round money to 2 decimals, and add LIMIT {MAX_ROWS} to lists.
- "My"/"our" means the company's records as a whole; there is no per-user data."""

ANSWER_SYSTEM = ("You are an ERP assistant. Answer the user's question in 1-3 sentences using only the query "
                 "result given. If the result is empty, say no matching records were found.")


def check_sql(sql):
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("only a single statement is allowed")
    if not re.match(r"(?is)^(select|with)\b", sql):
        raise ValueError("only SELECT queries are allowed")
    return sql


def run_sql(sql):
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description or []]
        return cols, cur.fetchmany(MAX_ROWS)
    finally:
        con.close()


def format_table(cols, rows, limit=10):
    if not cols:
        return "(no columns)"
    shown = [[("" if v is None else str(v)) for v in r] for r in rows[:limit]]
    widths = [min(30, max(len(c), *(len(r[i]) for r in shown))) if shown else len(c) for i, c in enumerate(cols)]
    line = lambda vals: "  ".join(v[:w].ljust(w) for v, w in zip(vals, widths))
    out = [line(cols), line(["-" * w for w in widths])] + [line(r) for r in shown]
    if len(rows) > limit:
        out.append(f"... {len(rows) - limit} more rows")
    return "\n".join(out) if rows else line(cols) + "\n(no rows)"


def answer_sql(client, schema, question):
    prompt = f"Schema:\n{schema}\n\nQuestion: {question}"
    sql = generate(client, prompt, SQL_SYSTEM, as_json=True)["sql"]
    try:
        sql = check_sql(sql)
        cols, rows = run_sql(sql)
    except sqlite3.Error as err:  # one repair attempt
        retry = f"{prompt}\n\nThis query failed:\n{sql}\nError: {err}\nReturn a corrected query."
        sql = check_sql(generate(client, retry, SQL_SYSTEM, as_json=True)["sql"])
        cols, rows = run_sql(sql)

    print(f"\nSQL:\n  {sql}\n")
    print(format_table(cols, rows))
    result = json.dumps({"columns": cols, "rows": rows[:MAX_ROWS]}, default=str)
    return generate(client, f"Question: {question}\nSQL: {sql}\nResult: {result}", ANSWER_SYSTEM)


# ---------------------------------------------------------------- RAG

RAG_SYSTEM = ("You are an ERP help assistant. Answer the question using only the provided company documents. "
              "Be concise and cite the document names in square brackets, e.g. [hr_leave_policy]. "
              "If the documents do not contain the answer, say you could not find it in the company documents.")


def chunk_docs():
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        title = lines[0].lstrip("# ") if lines and lines[0].startswith("#") else path.stem
        body = lines[1:] if lines and lines[0].startswith("#") else lines
        for i in range(0, max(1, len(body)), 3):
            chunks.append({"doc": path.stem, "text": f"{title}\n" + "\n".join(body[i:i + 3])})
    return chunks


def load_index(client):
    signature = json.dumps([EMBED_MODEL] + [[p.name, p.stat().st_mtime] for p in sorted(DOCS_DIR.glob("*.md"))])
    if INDEX_PATH.exists():
        data = np.load(INDEX_PATH, allow_pickle=False)
        if str(data["signature"]) == signature:
            return json.loads(str(data["chunks"])), data["vectors"]
    print("Building RAG index from erp_dataset/docs ...")
    chunks = chunk_docs()
    vectors = embed(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    np.savez(INDEX_PATH, signature=signature, chunks=json.dumps(chunks), vectors=vectors)
    return chunks, vectors


def answer_rag(client, index, question):
    chunks, vectors = index
    q = embed(client, [question], "RETRIEVAL_QUERY")[0]
    scores = vectors @ q
    top = [chunks[i] for i in np.argsort(-scores)[:TOP_K]]
    context = "\n\n".join(f"[{c['doc']}]\n{c['text']}" for c in top)
    sources = list(dict.fromkeys(c["doc"] for c in top))
    print(f"\nRetrieved from: {', '.join(sources)}")
    return generate(client, f"Documents:\n{context}\n\nQuestion: {question}", RAG_SYSTEM)


# ---------------------------------------------------------------- chat loop

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    client = make_client()
    route = load_router()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    index = load_index(client)

    print("\n" + "=" * 60)
    print("  ERP ASSISTANT  (Laya router -> Text-to-SQL | RAG, Gemini)")
    print("=" * 60)
    print("Ask about orders, invoices, stock, employees, or company policies.")
    print("/sql <q> or /rag <q> forces a branch. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        forced = re.match(r"(?i)^/(sql|rag)\s+(.+)", question)
        if forced:
            choice, question = ("TEXT_TO_SQL" if forced[1].lower() == "sql" else "RAG"), forced[2]
            print("Route: %s (forced)" % choice)
        else:
            choice, conf = route(question)
            print("Route: %s (confidence %.2f)" % (choice, conf))

        try:
            answer = answer_sql(client, schema, question) if choice == "TEXT_TO_SQL" else answer_rag(client, index, question)
            print(f"\nAssistant: {answer}\n" + "-" * 60)
        except Exception as err:
            print(f"\nCould not answer: {err}\n" + "-" * 60)


if __name__ == "__main__":
    main()
