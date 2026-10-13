#!/usr/bin/env python3
"""
Build a Laya fine-tuning dataset for the ERPNext chatbot router (TEXT_TO_SQL vs RAG).

Every ERPNext doctype / business term is asked about from both sides, so the model has to learn
the intent ("What is my Material Receipt amount?" -> SQL) instead of the topic
("What is a Material Receipt?" -> RAG).

Outputs (in --out directory), rows in laya-train's {state, questions, expected} format:
  train.jsonl   seen templates x seen doctypes/terms
  val.jsonl     random slice of the same pool (different fillings)
  test.jsonl    held-out templates OR held-out doctypes/terms -> measures generalisation
  stats.json

Usage:  python build_laya_dataset.py --out laya_data --seed 42
"""
import argparse
import json
import os
import random
from collections import Counter

from router_question import QUESTIONS, make_state

SQL, RAG = "TEXT_TO_SQL", "RAG"

# --------------------------------------------------------------------------- #
# 1. VOCABULARY
# --------------------------------------------------------------------------- #
# name: (plural, naming-series prefix, party kind, has amount, has items, statuses)
DOCTYPES = {
    "Material Receipt": ("Material Receipts", "MAT-STE", None, True, True, ["draft", "submitted", "cancelled"]),
    "Material Issue": ("Material Issues", "MAT-STE", None, True, True, ["draft", "submitted", "cancelled"]),
    "Material Transfer": ("Material Transfers", "MAT-STE", None, True, True, ["draft", "submitted", "in transit"]),
    "Stock Entry": ("Stock Entries", "MAT-STE", None, True, True, ["draft", "submitted", "cancelled"]),
    "Material Request": ("Material Requests", "MAT-MR", None, False, True, ["pending", "ordered", "partially ordered", "issued"]),
    "Purchase Order": ("Purchase Orders", "PUR-ORD", "supplier", True, True, ["to receive and bill", "to bill", "completed", "draft"]),
    "Purchase Receipt": ("Purchase Receipts", "MAT-PRE", "supplier", True, True, ["to bill", "completed", "return"]),
    "Purchase Invoice": ("Purchase Invoices", "ACC-PINV", "supplier", True, True, ["unpaid", "overdue", "paid"]),
    "Supplier Quotation": ("Supplier Quotations", "PUR-SQTN", "supplier", True, True, ["draft", "submitted", "expired"]),
    "Request for Quotation": ("Requests for Quotation", "PUR-RFQ", None, False, True, ["draft", "submitted"]),
    "Quotation": ("Quotations", "SAL-QTN", "customer", True, True, ["open", "ordered", "lost", "expired"]),
    "Sales Order": ("Sales Orders", "SAL-ORD", "customer", True, True, ["to deliver and bill", "to bill", "completed", "on hold"]),
    "Delivery Note": ("Delivery Notes", "MAT-DN", "customer", True, True, ["to bill", "completed", "return"]),
    "Sales Invoice": ("Sales Invoices", "ACC-SINV", "customer", True, True, ["unpaid", "overdue", "paid", "return"]),
    "Payment Entry": ("Payment Entries", "ACC-PAY", "customer", True, False, ["draft", "submitted", "cancelled"]),
    "Journal Entry": ("Journal Entries", "ACC-JV", None, True, False, ["draft", "submitted", "cancelled"]),
    "Payment Request": ("Payment Requests", "ACC-PRQ", "customer", True, False, ["initiated", "paid", "requested"]),
    "Credit Note": ("Credit Notes", "ACC-SINV", "customer", True, True, ["submitted", "draft"]),
    "Debit Note": ("Debit Notes", "ACC-PINV", "supplier", True, True, ["submitted", "draft"]),
    "Expense Claim": ("Expense Claims", "HR-EXP", "employee", True, False, ["draft", "approved", "rejected", "paid"]),
    "Leave Application": ("Leave Applications", "HR-LAP", "employee", False, False, ["open", "approved", "rejected"]),
    "Salary Slip": ("Salary Slips", "Sal Slip", "employee", True, False, ["draft", "submitted"]),
    "Employee Advance": ("Employee Advances", "HR-EAD", "employee", True, False, ["unpaid", "paid", "claimed"]),
    "Attendance": ("Attendance records", "HR-ATT", "employee", False, False, ["present", "absent", "on leave"]),
    "Work Order": ("Work Orders", "MFG-WO", None, False, True, ["not started", "in process", "completed", "stopped"]),
    "Job Card": ("Job Cards", "PO-JOB", None, False, False, ["open", "work in progress", "completed"]),
    "BOM": ("BOMs", "BOM", None, True, True, ["active", "default", "inactive"]),
    "Stock Reconciliation": ("Stock Reconciliations", "MAT-RECO", None, True, True, ["draft", "submitted"]),
    "Landed Cost Voucher": ("Landed Cost Vouchers", "MAT-LCV", None, True, False, ["draft", "submitted"]),
    "Asset": ("Assets", "ACC-ASS", None, True, False, ["submitted", "partially depreciated", "scrapped"]),
    "Project": ("Projects", "PROJ", "customer", True, False, ["open", "completed", "cancelled"]),
    "Lead": ("Leads", "CRM-LEAD", None, False, False, ["open", "replied", "converted", "lost"]),
    "Opportunity": ("Opportunities", "CRM-OPP", "customer", True, False, ["open", "quotation", "lost", "converted"]),
    "Pick List": ("Pick Lists", "STO-PICK", None, False, True, ["draft", "open", "completed"]),
}
TEST_DOCTYPES = {"Supplier Quotation", "Debit Note", "Job Card", "Landed Cost Voucher", "Employee Advance",
                 "Opportunity"}

# term: SQL phrasings that ask for that term's VALUE on real records
TERMS = {
    "valuation rate": ["What is the valuation rate of {item}?", "current valuation rate for {item}",
                       "valuation rate of {item} in {warehouse}"],
    "credit limit": ["What is the credit limit of {customer}?", "credit limit set for {customer}",
                     "how much credit limit is left for {customer}?"],
    "outstanding amount": ["What is the outstanding amount for {customer}?", "how much does {customer} owe us?",
                           "outstanding amount of {supplier} {period}"],
    "reorder level": ["What is the reorder level of {item}?", "reorder level set for {item} in {warehouse}"],
    "safety stock": ["What is the safety stock for {item}?", "safety stock quantity of {item}"],
    "projected quantity": ["What is the projected qty of {item} in {warehouse}?", "projected quantity for {item}"],
    "payment terms": ["What are the payment terms for {customer}?", "payment terms on {supplier}'s account"],
    "gross profit": ["What is our gross profit {period}?", "gross profit for {customer} {period}"],
    "TDS": ["How much TDS was deducted {period}?", "TDS deducted on payments to {supplier}"],
    "GST": ["What is the total GST collected {period}?", "GST payable {period}"],
    "accounts receivable": ["Show accounts receivable for {customer}", "what is our total accounts receivable today?"],
    "accounts payable": ["What is our total accounts payable?", "accounts payable to {supplier}"],
    "stock ageing": ["Show stock ageing for {warehouse}", "stock ageing of {item}"],
    "leave balance": ["What is my leave balance?", "leave balance of {employee}"],
    "cost center": ["What is the total expense booked to {cost_center} {period}?", "expenses under {cost_center} {period}"],
    "bank balance": ["What is the balance in {bank}?", "current bank balance of {bank}"],
    "depreciation": ["How much depreciation was booked {period}?", "depreciation on our assets {period}"],
    "budget variance": ["What is the budget variance for {cost_center} {period}?", "budget vs actual for {cost_center}"],
}
TEST_TERMS = {"TDS", "projected quantity", "budget variance", "stock ageing"}

CUSTOMERS = ["Apex Industries", "Delta Solutions", "Nova Retail", "Summit Traders", "Lotus Enterprises",
             "Horizon Logistics", "Zenith Manufacturing", "Pioneer Supplies", "Meridian Foods", "Orbit Electronics"]
SUPPLIERS = ["Granite Metals", "Cedar Packaging", "Falcon Components", "Ironwood Materials", "Juniper Wholesale",
             "Keystone Distribution", "Titan Steel", "Willow Chemicals"]
EMPLOYEES = ["Rahul Sharma", "Priya Patil", "Amit Mehta", "Neha Iyer", "Karan Reddy", "Sneha Nair", "Vikram Joshi"]
ITEMS = ["Ball Bearing 6204", "A4 Paper Ream", "Hydraulic Hose 1m", "Office Chair", "Safety Helmet",
         "Corrugated Box Large", "Servo Motor 750W", "Barcode Scanner", "MS Steel Sheet 2mm", "Copper Wire 4mm"]
WAREHOUSES = ["Stores - GLX", "Finished Goods - GLX", "Work In Progress - GLX", "Raw Material - GLX",
              "Pune Warehouse - GLX"]
COST_CENTERS = ["Main - GLX", "Production - GLX", "Sales - GLX", "Admin - GLX"]
BANKS = ["HDFC Bank - GLX", "ICICI Current Account - GLX", "Cash - GLX", "SBI Bank - GLX"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
PERIODS = ["today", "yesterday", "this week", "last week", "this month", "last month", "this quarter",
           "this year", "last year", "this financial year", "in the last 30 days", "in {month}",
           "in {month} {year}", "since {month}", "between {month} and {month2}"]

# --------------------------------------------------------------------------- #
# 2. TEMPLATES   (id, label, phrasings, needs)
#    needs: dt = doctype slot; amount / items / party = doctype capabilities;
#           term_rag = term slot; no dt/term = general template.
# --------------------------------------------------------------------------- #
TEMPLATES = [
 # ---- SQL on a doctype -------------------------------------------------- #
 ("s_my_amount", SQL, ["What is my {dt} amount?", "what's my {dt} amount {period}?", "my {dt} amount",
   "what is my total {dt} amount?"], {"dt", "amount"}),
 ("s_my_latest", SQL, ["What is my latest {dt}?", "what was my last {dt}?", "show my most recent {dt}",
   "my latest {dt}"], {"dt"}),
 ("s_total_period", SQL, ["What is the total {dt} amount {period}?", "total value of {dts} {period}",
   "sum of all {dts} {period}", "how much is the total of {dts} {period}?"], {"dt", "amount"}),
 ("s_show_last", SQL, ["Show my last {n} {dts}", "show the last {n} {dts}", "list the latest {n} {dts}",
   "give me the {n} most recent {dts}"], {"dt"}),
 ("s_list_period", SQL, ["List {dts} created {period}", "show all {dts} {period}", "which {dts} were made {period}?",
   "{dts} {period}"], {"dt"}),
 ("s_count_period", SQL, ["How many {dts} were created {period}?", "count of {dts} {period}",
   "number of {dts} {period}", "how many {dts} did we make {period}?"], {"dt"}),
 ("s_count_status", SQL, ["How many {dts} are {status}?", "count {status} {dts}", "number of {dts} in {status} status",
   "how many {status} {dts} do we have?"], {"dt"}),
 ("s_which_status", SQL, ["Which {dts} are still {status}?", "show {status} {dts}", "list all {dts} that are {status}",
   "find {dts} with status {status}"], {"dt"}),
 ("s_status_id", SQL, ["What is the status of {id}?", "status of {id}", "{id} status", "is {id} submitted yet?"],
  {"dt"}),
 ("s_amount_id", SQL, ["What is the amount of {id}?", "what's the grand total on {id}?", "{id} amount",
   "total amount for {dt} {id}"], {"dt", "amount"}),
 ("s_details_id", SQL, ["Show me the details of {id}", "open {dt} {id}", "details of {id}", "pull up {id}"], {"dt"}),
 ("s_field_id", SQL, ["What is the posting date of {id}?", "who created {id}?", "when was {id} submitted?",
   "what is the due date on {id}?"], {"dt"}),
 ("s_items_id", SQL, ["Which items are in {id}?", "list the items on {id}", "what quantity of {item} is in {id}?",
   "items and quantities in {dt} {id}"], {"dt", "items"}),
 ("s_item_period", SQL, ["Show {dts} for item {item} {period}", "{dts} containing {item} {period}",
   "how many {dts} include {item}?", "total quantity of {item} across {dts} {period}"], {"dt", "items"}),
 ("s_warehouse", SQL, ["{dts} for {warehouse} {period}", "show {dts} posted to {warehouse}",
   "how many {dts} went into {warehouse} {period}?", "list {dts} for warehouse {warehouse}"], {"dt", "items"}),
 ("s_party", SQL, ["Show {dts} for {party}", "list all {dts} of {party} {period}", "{party}'s {dts} {period}",
   "do we have any {dts} for {party}?"], {"dt", "party"}),
 ("s_party_total", SQL, ["What is the total {dt} amount for {party} {period}?", "how much {dt} value with {party} {period}?",
   "{dt} total for {party}", "sum of {dts} for {party} {period}"], {"dt", "party", "amount"}),
 ("s_pending_mine", SQL, ["Do I have any pending {dts}?", "how many {dts} are waiting for my approval?",
   "show {dts} pending my approval", "my pending {dts}"], {"dt"}),
 ("s_top", SQL, ["Top {n} {dts} by amount {period}", "show the {n} biggest {dts} {period}",
   "which {dt} had the highest value {period}?", "largest {dts} {period}"], {"dt", "amount"}),
 ("s_avg", SQL, ["What is the average {dt} amount {period}?", "average value of {dts} {period}",
   "mean {dt} amount {period}", "avg {dt} value {period}"], {"dt", "amount"}),
 ("s_compare", SQL, ["Compare {dt} totals for {month} and {month2}", "{dt} amount in {month} vs {month2}",
   "did {dts} go up or down from {month} to {month2}?", "month-wise {dt} totals for {year}"], {"dt", "amount"}),
 ("s_report", SQL, ["Give me a report of {dts} {period}", "export {dts} {period}", "{dt} report {period}",
   "generate a summary of {dts} {period}"], {"dt"}),
 ("s_by_user", SQL, ["How many {dts} did {employee} create {period}?", "{dts} made by {employee}",
   "show {employee}'s {dts} {period}", "list {dts} owned by {employee}"], {"dt"}),
 ("s_cancelled", SQL, ["Which {dts} were cancelled {period}?", "how many {dts} got cancelled {period}?",
   "list cancelled {dts}", "show the {dts} cancelled by {employee}"], {"dt"}),
 # ---- general SQL ------------------------------------------------------- #
 ("s_stock", SQL, ["How much stock of {item} is in {warehouse}?", "current stock of {item}",
   "available qty of {item} in {warehouse}", "do we have {item} in stock?"], set()),
 ("s_sales", SQL, ["What were total sales {period}?", "how much revenue did we make {period}?",
   "sales figure {period}", "our total billing {period}"], set()),
 ("s_purchase", SQL, ["What is the total purchase amount {period}?", "how much did we buy {period}?",
   "total purchases from {supplier} {period}", "purchase spend {period}"], set()),
 ("s_overdue", SQL, ["Which customers have overdue invoices?", "show customers with overdue payments",
   "list overdue invoices of {customer}", "who has not paid us yet?"], set()),
 ("s_low_stock", SQL, ["Which items are below reorder level?", "items running low in {warehouse}",
   "what should we reorder now?", "show items with zero stock"], set()),
 ("s_my_hr", SQL, ["What is my net pay for {month}?", "how many leaves have I taken this year?",
   "was I marked present on {month} 5?", "show my salary slip for {month}"], set()),
 ("s_top_party", SQL, ["Who are our top {n} customers {period}?", "top {n} suppliers by purchase value",
   "which customer bought the most {period}?", "best selling items {period}"], set()),
 ("s_received", SQL, ["How much payment did we receive {period}?", "payments received from {customer} {period}",
   "how much did we pay {supplier} {period}?", "cash collected {period}"], set()),
 # ---- RAG on a doctype -------------------------------------------------- #
 ("r_what_is", RAG, ["What is a {dt}?", "what is {dt}?", "what is a {dt} in ERPNext?", "what exactly is a {dt}?"],
  {"dt"}),
 ("r_meaning", RAG, ["{dt} meaning", "what does {dt} mean?", "define {dt}", "meaning of {dt} in ERP"], {"dt"}),
 ("r_explain", RAG, ["Explain the {dt} process", "explain {dt}", "how does a {dt} work?",
   "can you explain how {dt} works in ERPNext?"], {"dt"}),
 ("r_how_create", RAG, ["How do I create a {dt}?", "how to make a {dt}", "steps to create a {dt}",
   "where do I create a new {dt}?"], {"dt"}),
 ("r_how_cancel", RAG, ["How do I cancel a submitted {dt}?", "how to amend a {dt}?", "can I edit a {dt} after submitting it?",
   "what happens when I cancel a {dt}?"], {"dt"}),
 ("r_purpose", RAG, ["Why do we use {dt}?", "what is the purpose of a {dt}?", "when should I use a {dt}?",
   "what is {dt} used for?"], {"dt"}),
 ("r_diff", RAG, ["What is the difference between a {dt} and a {dt2}?", "{dt} vs {dt2}",
   "how is a {dt} different from a {dt2}?", "when do I use {dt} instead of {dt2}?"], {"dt"}),
 ("r_status_mean", RAG, ["What does {status} status mean on a {dt}?", "what does it mean when a {dt} is {status}?",
   "meaning of {status} status in {dt}", "why does my {dt} show {status}?"], {"dt"}),
 ("r_trouble", RAG, ["Why can't I submit my {dt}?", "I am getting an error while saving a {dt}, what should I check?",
   "{dt} not submitting, how to fix?", "why is the {dt} button greyed out?"], {"dt"}),
 ("r_approval", RAG, ["Who approves a {dt}?", "explain the {dt} approval workflow",
   "what is the approval process for {dts}?", "what is our policy for {dts}?"], {"dt"}),
 ("r_fields", RAG, ["Which fields are mandatory in a {dt}?", "what information is needed to fill a {dt}?",
   "what documents are required before making a {dt}?", "what should I enter in a {dt}?"], {"dt"}),
 ("r_accounting", RAG, ["What accounting entries does a {dt} create?", "does a {dt} affect stock or accounts?",
   "how does a {dt} impact the general ledger?", "which ledger is updated by a {dt}?"], {"dt"}),
 ("r_flow", RAG, ["What comes after a {dt} in the workflow?", "what is created from a {dt}?",
   "where does {dt} fit in the purchase/sales cycle?", "what is the next step after submitting a {dt}?"], {"dt"}),
 ("r_settings", RAG, ["How do I set a default warehouse for {dts}?", "how to change the naming series of {dt}?",
   "how to set up a print format for {dt}?", "how do I add a custom field to {dt}?"], {"dt"}),
 # ---- RAG on a term ------------------------------------------------------ #
 ("r_term_what", RAG, ["What is {term}?", "what does {term} mean?", "{term} meaning", "define {term}"], {"term_rag"}),
 ("r_term_how", RAG, ["How is {term} calculated?", "explain {term} in ERPNext", "how does {term} work?",
   "why is {term} important?"], {"term_rag"}),
 # ---- general RAG -------------------------------------------------------- #
 ("r_policy", RAG, ["What is our company's purchase policy?", "what is the leave policy?",
   "explain the refund policy", "what is the travel reimbursement limit?", "which documents describe our credit policy?",
   "what is the work from home policy?"], set()),
 ("r_concepts", RAG, ["What is three-way matching?", "what is perpetual inventory?", "what is FIFO valuation?",
   "how does month-end closing work?", "what is a fiscal year?", "what is the chart of accounts?"], set()),
 ("r_howto", RAG, ["How do I reset my ERPNext password?", "how do I add a new user?", "how to set up a new company?",
   "how do I raise a support ticket?", "how do I change the default currency?", "how to import items from Excel?"],
  set()),
 ("r_rules", RAG, ["Who approves purchase orders above 5 lakh?", "how many quotations are required for big purchases?",
   "what is the discount limit for salespeople?", "when is a customer put on credit hold?",
   "what is the notice period?", "how long do customers have to return goods?"], set()),
]
TERM_SQL_ID = "s_term_value"
TEST_TEMPLATES = {"s_avg", "s_field_id", "s_cancelled", "s_received", "s_my_latest",
                  "r_purpose", "r_accounting", "r_settings", "r_term_how", "r_rules"}

PREFIXES = ["please ", "hey, ", "can you tell me ", "I want to know ", "quick question: ", "hi, ", "just checking, "]


# --------------------------------------------------------------------------- #
# 3. GENERATION
# --------------------------------------------------------------------------- #
def typo(q, rng):
    words = q.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not idx:
        return q
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(len(w) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def augment(q, rng):
    if rng.random() < 0.2:
        q = rng.choice(PREFIXES) + q[0].lower() + q[1:]
    if rng.random() < 0.25:
        q = q.lower()
    if q.endswith("?") and rng.random() < 0.3:
        q = q[:-1]
    if rng.random() < 0.08:
        q = typo(q, rng)
    return q[0].upper() + q[1:] if rng.random() < 0.5 else q


def casing(name, rng):
    return name if rng.random() < 0.55 or name.isupper() else name.lower()


def sample_slots(rng, dt=None, term=None):
    year = rng.choice([2024, 2025, 2026])
    m1, m2 = rng.sample(range(12), 2)
    s = {"year": year, "month": MONTHS[m1], "month2": MONTHS[m2], "n": rng.choice([3, 5, 10, 20]),
         "customer": rng.choice(CUSTOMERS), "supplier": rng.choice(SUPPLIERS), "employee": rng.choice(EMPLOYEES),
         "item": rng.choice(ITEMS), "warehouse": rng.choice(WAREHOUSES), "cost_center": rng.choice(COST_CENTERS),
         "bank": rng.choice(BANKS)}
    s["period"] = rng.choice(PERIODS).format(**s)
    if dt:
        plural, prefix, party, _, _, statuses = DOCTYPES[dt]
        other = rng.choice([d for d in DOCTYPES if d != dt])
        s.update(dt=casing(dt, rng), dts=casing(plural, rng), dt2=casing(other, rng), status=rng.choice(statuses),
                 id="%s-%d-%05d" % (prefix, rng.choice([2025, 2026]), rng.randint(1, 4000)),
                 party=s.get(party, rng.choice(CUSTOMERS)) if party else "")
    if term:
        s["term"] = term
    return s


def units():
    """Every (template_id, label, phrasings, concept) combination the generator fills."""
    for tid, label, phrasings, needs in TEMPLATES:
        if "dt" in needs:
            for dt, (_, _, party, amount, items, _) in DOCTYPES.items():
                if ("amount" in needs and not amount) or ("items" in needs and not items) \
                        or ("party" in needs and not party):
                    continue
                yield tid, label, phrasings, dt
        elif "term_rag" in needs:
            for term in TERMS:
                yield tid, label, phrasings, term
        else:
            yield tid, label, phrasings, "general"
    for term, phrasings in TERMS.items():
        yield TERM_SQL_ID, SQL, phrasings, term


def split_of(tid, concept):
    if tid in TEST_TEMPLATES or concept in TEST_DOCTYPES or concept in TEST_TERMS:
        return "test"
    return "pool"


def generate(rng, per_unit, per_general):
    rows, seen = {"pool": [], "test": []}, set()
    for tid, label, phrasings, concept in units():
        target = per_general if concept == "general" else per_unit
        made, attempts = 0, 0
        while made < target and attempts < target * 30:
            attempts += 1
            s = sample_slots(rng, dt=concept if concept in DOCTYPES else None,
                             term=concept if concept in TERMS else None)
            q = augment(rng.choice(phrasings).format(**s), rng)
            key = " ".join(q.lower().replace("?", "").split())
            if key in seen:
                continue
            seen.add(key)
            rows[split_of(tid, concept)].append({"query": q, "label": label, "template": tid, "concept": concept})
            made += 1
    return rows


def balance(rows, rng, cap):
    by = {SQL: [r for r in rows if r["label"] == SQL], RAG: [r for r in rows if r["label"] == RAG]}
    n = min(len(by[SQL]), len(by[RAG]), cap // 2)
    out = rng.sample(by[SQL], n) + rng.sample(by[RAG], n)
    rng.shuffle(out)
    return out


def to_laya(r, rid):
    return {"id": rid, "state": make_state(r["query"]), "questions": QUESTIONS, "expected": {"route": r["label"]},
            "meta": {"template": r["template"], "concept": r["concept"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="laya_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-size", type=int, default=3000)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--test-size", type=int, default=600)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    rows = generate(rng, per_unit=6, per_general=50)
    pool = rows["pool"]
    rng.shuffle(pool)
    n_val = len(pool) * a.val_size // (a.val_size + a.train_size)
    splits = {"train": balance(pool[n_val:], rng, a.train_size),
              "val": balance(pool[:n_val], rng, a.val_size),
              "test": balance(rows["test"], rng, a.test_size)}

    stats = {}
    for name, split in splits.items():
        with open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for i, r in enumerate(split):
                f.write(json.dumps(to_laya(r, "%s-%05d" % (name, i)), ensure_ascii=False) + "\n")
        stats[name] = {"total": len(split), "labels": dict(Counter(r["label"] for r in split)),
                       "templates": len({r["template"] for r in split}),
                       "concepts": len({r["concept"] for r in split})}
    stats["held_out"] = {"templates": sorted(TEST_TEMPLATES), "doctypes": sorted(TEST_DOCTYPES),
                         "terms": sorted(TEST_TERMS)}
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
