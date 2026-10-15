#!/usr/bin/env python3
"""
Generate a synthetic ERP dataset for a hybrid RAG + text-to-SQL chatbot.

Outputs (in --out directory):
  erp.db                 SQLite database (sales, purchasing, inventory, finance, HR)
  schema.sql             DDL of the database
  docs/*.md              Policy / how-to documents for the RAG side
  stats.json             Row counts per table

Usage:  python -m erp_assistant.generate_erp_data --seed 42      # writes data/erp/ (rebuilds erp.db)
"""
import argparse
import json
import os
import random
import sqlite3
from datetime import date, timedelta

from .config import ERP_DIR

TODAY = date(2026, 10, 11)  # fixed "now" so overdue/open statuses are reproducible

# --------------------------------------------------------------------------- #
# 1. DATABASE
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept_id INTEGER REFERENCES departments(dept_id),
    job_title TEXT, hire_date TEXT, salary REAL, manager_id INTEGER REFERENCES employees(emp_id),
    status TEXT CHECK(status IN ('active','resigned')));
CREATE TABLE leave_requests (
    leave_id INTEGER PRIMARY KEY, emp_id INTEGER REFERENCES employees(emp_id),
    leave_type TEXT, start_date TEXT, end_date TEXT,
    status TEXT CHECK(status IN ('pending','approved','rejected')));
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT, country TEXT, segment TEXT,
    credit_limit REAL, payment_terms_days INTEGER, created_at TEXT);
CREATE TABLE vendors (
    vendor_id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT, country TEXT,
    payment_terms_days INTEGER, rating REAL);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, sku TEXT, name TEXT NOT NULL, category TEXT,
    unit_price REAL, unit_cost REAL, vendor_id INTEGER REFERENCES vendors(vendor_id), reorder_level INTEGER);
CREATE TABLE warehouses (warehouse_id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT);
CREATE TABLE inventory (
    product_id INTEGER REFERENCES products(product_id), warehouse_id INTEGER REFERENCES warehouses(warehouse_id),
    quantity INTEGER, PRIMARY KEY (product_id, warehouse_id));
CREATE TABLE sales_orders (
    so_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), order_date TEXT,
    status TEXT CHECK(status IN ('pending','confirmed','shipped','delivered','cancelled')),
    total_amount REAL, salesperson_id INTEGER REFERENCES employees(emp_id));
CREATE TABLE sales_order_items (
    item_id INTEGER PRIMARY KEY, so_id INTEGER REFERENCES sales_orders(so_id),
    product_id INTEGER REFERENCES products(product_id), quantity INTEGER, unit_price REAL);
CREATE TABLE purchase_orders (
    po_id INTEGER PRIMARY KEY, vendor_id INTEGER REFERENCES vendors(vendor_id), order_date TEXT,
    expected_date TEXT,
    status TEXT CHECK(status IN ('open','partially_received','received','cancelled')),
    total_amount REAL);
CREATE TABLE purchase_order_items (
    item_id INTEGER PRIMARY KEY, po_id INTEGER REFERENCES purchase_orders(po_id),
    product_id INTEGER REFERENCES products(product_id), quantity INTEGER, unit_cost REAL);
CREATE TABLE invoices (
    invoice_id INTEGER PRIMARY KEY, so_id INTEGER REFERENCES sales_orders(so_id),
    customer_id INTEGER REFERENCES customers(customer_id), invoice_date TEXT, due_date TEXT,
    amount REAL, status TEXT CHECK(status IN ('paid','unpaid','overdue')));
CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY, invoice_id INTEGER REFERENCES invoices(invoice_id),
    payment_date TEXT, amount REAL, method TEXT);
"""

FIRST = ["Aarav", "Vivaan", "Aditya", "Rohan", "Karan", "Neha", "Priya", "Ananya", "Isha", "Kavya",
         "Rahul", "Sneha", "Amit", "Pooja", "Vikram", "Meera", "Arjun", "Divya", "Nikhil", "Riya",
         "Sanjay", "Anjali", "Manish", "Shruti", "Varun", "Tanvi", "Harsh", "Swati", "Yash", "Nisha"]
LAST = ["Sharma", "Patil", "Mehta", "Iyer", "Reddy", "Nair", "Gupta", "Joshi", "Kulkarni", "Desai",
        "Singh", "Verma", "Kapoor", "Shah", "Bose", "Rao", "Chopra", "Menon", "Pillai", "Jain"]
CITIES = ["Mumbai", "Pune", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Ahmedabad", "Kolkata", "Jaipur", "Nashik"]
PREFIX = ["Apex", "Blue Ridge", "Cedar", "Delta", "Evergreen", "Falcon", "Granite", "Horizon", "Ironwood",
          "Juniper", "Keystone", "Lotus", "Meridian", "Nova", "Orbit", "Pioneer", "Quantum", "Redwood",
          "Summit", "Titan", "Unity", "Vertex", "Willow", "Zenith"]
CUST_SUFFIX = ["Industries", "Traders", "Enterprises", "Logistics", "Manufacturing", "Retail", "Solutions", "Supplies"]
VEND_SUFFIX = ["Components", "Materials", "Wholesale", "Distribution", "Packaging", "Metals"]
DEPTS = {"Sales": 14, "Finance": 8, "HR": 5, "Procurement": 8, "Warehouse": 12, "IT": 6, "Production": 10}
TITLES = {"Sales": ["Sales Executive", "Account Manager"], "Finance": ["Accountant", "Finance Analyst"],
          "HR": ["HR Executive", "Recruiter"], "Procurement": ["Buyer", "Purchasing Officer"],
          "Warehouse": ["Warehouse Associate", "Inventory Controller"], "IT": ["Support Engineer", "ERP Analyst"],
          "Production": ["Machine Operator", "Production Supervisor"]}
SALARY = {"Sales": (35000, 90000), "Finance": (45000, 110000), "HR": (38000, 85000),
          "Procurement": (40000, 95000), "Warehouse": (25000, 60000), "IT": (50000, 130000),
          "Production": (28000, 70000)}
CATALOG = {
    "Electronics": (["Barcode Scanner", "Label Printer", "Wireless Router", "UPS 1kVA", "LED Panel Light",
                     "Industrial Tablet", "Thermal Receipt Printer", "Power Strip"], (800, 25000)),
    "Furniture": (["Office Chair", "Standing Desk", "Filing Cabinet", "Conference Table", "Storage Rack",
                   "Workbench", "Visitor Sofa", "Whiteboard"], (2500, 45000)),
    "Stationery": (["A4 Paper Ream", "Ballpoint Pen Box", "Stapler Heavy Duty", "Marker Set", "Notebook Pack",
                    "File Folder Pack", "Sticky Notes Pack", "Packing Tape Roll"], (40, 900)),
    "Machinery Parts": (["Ball Bearing 6204", "V-Belt B52", "Hydraulic Hose 1m", "Gear Pump", "Pneumatic Cylinder",
                         "Proximity Sensor", "Servo Motor 750W", "Coupling Flange"], (150, 30000)),
    "Packaging": (["Corrugated Box Large", "Bubble Wrap Roll", "Stretch Film Roll", "Pallet Wooden",
                   "Strapping Band Roll", "Carton Sealing Tape", "Foam Sheet", "Poly Bag Pack"], (30, 1500)),
    "Safety Equipment": (["Safety Helmet", "Safety Gloves Pair", "Safety Goggles", "Fire Extinguisher 5kg",
                          "First Aid Kit", "High Visibility Vest", "Ear Plugs Box", "Safety Shoes Pair"], (60, 4500)),
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def rdate(rng, start, end):
    return start + timedelta(days=rng.randint(0, (end - start).days))


def build_db(path, rng):
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    cur = con.cursor()

    # departments & employees
    dept_ids = {}
    for i, name in enumerate(DEPTS, 1):
        dept_ids[name] = i
        cur.execute("INSERT INTO departments VALUES (?,?)", (i, name))
    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], sum(DEPTS.values()))
    emp_id, heads, sales_emps = 0, {}, []
    for dname, size in DEPTS.items():
        for k in range(size):
            emp_id += 1
            lo, hi = SALARY[dname]
            title = f"{dname} Head" if k == 0 else rng.choice(TITLES[dname])
            if k == 0:
                heads[dname] = emp_id
                sal = hi * rng.uniform(1.1, 1.4)
            else:
                sal = rng.uniform(lo, hi)
            status = "resigned" if (k > 0 and rng.random() < 0.05) else "active"
            cur.execute("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?)",
                        (emp_id, names[emp_id - 1], dept_ids[dname], title,
                         rdate(rng, date(2016, 1, 1), date(2026, 6, 30)).isoformat(), round(sal, -2),
                         None if k == 0 else heads[dname], status))
            if dname == "Sales" and status == "active":
                sales_emps.append(emp_id)
    n_emp = emp_id

    # leave requests
    for i in range(1, 161):
        s = rdate(rng, date(2026, 1, 1), date(2026, 12, 20))
        status = "pending" if s > TODAY - timedelta(days=10) and rng.random() < 0.7 else \
            rng.choices(["approved", "rejected", "pending"], [75, 15, 10])[0]
        cur.execute("INSERT INTO leave_requests VALUES (?,?,?,?,?,?)",
                    (i, rng.randint(1, n_emp), rng.choice(["annual", "sick", "casual", "unpaid"]),
                     s.isoformat(), (s + timedelta(days=rng.randint(0, 6))).isoformat(), status))

    # vendors, customers, warehouses
    vnames = rng.sample([f"{p} {s}" for p in PREFIX for s in VEND_SUFFIX], 25)
    for i, n in enumerate(vnames, 1):
        cur.execute("INSERT INTO vendors VALUES (?,?,?,?,?,?)",
                    (i, n, rng.choice(CITIES), "India", rng.choice([15, 30, 45, 60]), round(rng.uniform(2.5, 5.0), 1)))
    cnames = rng.sample([f"{p} {s}" for p in PREFIX for s in CUST_SUFFIX], 80)
    cust_terms = {}
    for i, n in enumerate(cnames, 1):
        terms = rng.choice([15, 30, 30, 45, 60])
        cust_terms[i] = terms
        cur.execute("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)",
                    (i, n, rng.choice(CITIES), "India", rng.choice(["Retail", "Wholesale", "Corporate", "Government"]),
                     rng.choice([100000, 250000, 500000, 1000000]), terms,
                     rdate(rng, date(2023, 6, 1), date(2026, 9, 30)).isoformat()))
    wh = [("Main Warehouse", "Pune"), ("North Hub", "Delhi"), ("South Hub", "Chennai")]
    for i, (n, c) in enumerate(wh, 1):
        cur.execute("INSERT INTO warehouses VALUES (?,?,?)", (i, n, c))

    # products & inventory
    products = []  # (id, price, cost)
    pid = 0
    for cat, (items, (lo, hi)) in CATALOG.items():
        for name in items:
            pid += 1
            price = round(rng.uniform(lo, hi), 0)
            cost = round(price * rng.uniform(0.55, 0.8), 0)
            cur.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)",
                        (pid, f"SKU-{pid:04d}", name, cat, price, cost, rng.randint(1, 25), rng.choice([50, 100, 150, 200])))
            products.append((pid, price, cost))
            for w in range(1, 4):
                cur.execute("INSERT INTO inventory VALUES (?,?,?)", (pid, w, rng.randint(0, 260)))

    # sales orders, items, invoices, payments
    item_id = inv_id = pay_id = 0
    for so in range(1, 2001):
        od = rdate(rng, date(2024, 1, 1), date(2026, 10, 5))
        age = (TODAY - od).days
        status = rng.choices(["delivered", "cancelled", "shipped"], [88, 9, 3])[0] if age > 60 else \
            rng.choices(["pending", "confirmed", "shipped", "delivered", "cancelled"], [25, 20, 20, 27, 8])[0]
        cust = rng.randint(1, 80)
        total = 0.0
        for p in rng.sample(products, rng.randint(1, 5)):
            item_id += 1
            q = rng.randint(1, 50)
            up = round(p[1] * rng.uniform(0.9, 1.05), 2)
            total += q * up
            cur.execute("INSERT INTO sales_order_items VALUES (?,?,?,?,?)", (item_id, so, p[0], q, up))
        total = round(total, 2)
        cur.execute("INSERT INTO sales_orders VALUES (?,?,?,?,?,?)",
                    (so, cust, od.isoformat(), status, total, rng.choice(sales_emps)))
        if status in ("delivered", "shipped"):
            inv_id += 1
            idate = od + timedelta(days=rng.randint(0, 5))
            due = idate + timedelta(days=cust_terms[cust])
            paid = rng.random() < (0.9 if age > 90 else 0.5)
            st = "paid" if paid else ("overdue" if due < TODAY else "unpaid")
            cur.execute("INSERT INTO invoices VALUES (?,?,?,?,?,?,?)",
                        (inv_id, so, cust, idate.isoformat(), due.isoformat(), total, st))
            if paid:
                pay_id += 1
                pdate = min(TODAY, idate + timedelta(days=rng.randint(5, cust_terms[cust] + 20)))
                cur.execute("INSERT INTO payments VALUES (?,?,?,?,?)",
                            (pay_id, inv_id, pdate.isoformat(), total,
                             rng.choice(["bank_transfer", "cheque", "upi", "card"])))

    # purchase orders
    item_id = 0
    for po in range(1, 601):
        od = rdate(rng, date(2024, 1, 1), date(2026, 10, 5))
        age = (TODAY - od).days
        status = rng.choices(["received", "cancelled", "partially_received"], [90, 6, 4])[0] if age > 60 else \
            rng.choices(["open", "partially_received", "received", "cancelled"], [40, 20, 30, 10])[0]
        total = 0.0
        for p in rng.sample(products, rng.randint(1, 4)):
            item_id += 1
            q = rng.randint(10, 300)
            total += q * p[2]
            cur.execute("INSERT INTO purchase_order_items VALUES (?,?,?,?,?)", (item_id, po, p[0], q, p[2]))
        cur.execute("INSERT INTO purchase_orders VALUES (?,?,?,?,?,?)",
                    (po, rng.randint(1, 25), od.isoformat(), (od + timedelta(days=rng.randint(7, 30))).isoformat(),
                     status, round(total, 2)))
    con.commit()
    return con


# --------------------------------------------------------------------------- #
# 2. RAG DOCUMENTS
# --------------------------------------------------------------------------- #
DOCS = {
"hr_leave_policy": """# Leave Policy
Full-time employees receive 24 days of paid annual leave per calendar year, accrued at 2 days per month.
Up to 10 unused days can be carried forward to the next year and expire on 31 March.
Sick leave is 10 days per year. A medical certificate is required for sick leave longer than 2 consecutive days.
Casual leave is 6 days per year and cannot be combined with annual leave for more than 3 days.
Unpaid leave needs written approval from HR in addition to the line manager.
Leave requests are approved by the line manager within 2 working days. Unapproved requests are escalated to the department head.
""",
"hr_apply_leave_steps": """# How to apply for leave in the ERP
1. Open HR > Leave > New Request.
2. Choose the leave type (annual, sick, casual, unpaid) and select start and end dates.
3. Add a reason and attach a medical certificate for sick leave longer than 2 days.
4. Click Submit. Your line manager receives a notification and the status shows as Pending.
You can cancel a pending request from HR > Leave > My Requests. Approved leave can only be cancelled by HR.
""",
"hr_notice_period": """# Resignation and Notice Period
Confirmed staff must serve a 30-day notice period. Managers and department heads serve 60 days.
Employees on probation serve 7 days. Notice can be bought out with approval from HR and the Finance Manager.
Submit the resignation under HR > Separation > New Resignation. The exit clearance checklist covers laptop return,
ERP access revocation, final expense settlement and knowledge handover. Final settlement is paid within 30 days of the last working day.
""",
"hr_remote_work": """# Remote and Hybrid Work Policy
Eligible employees may work remotely up to 2 days per week with prior approval from their line manager.
Core hours are 10:00 to 16:00 and employees must be reachable during this time.
Warehouse and Production roles are on-site only. Company laptops must use the VPN when accessing the ERP remotely.
""",
"finance_expense_reimbursement": """# Expense Reimbursement Policy
Expenses must be submitted within 30 days of being incurred via Finance > Expenses > New Claim.
Receipts are mandatory for any expense above 25. Claims are approved first by the line manager and then by Finance.
Approved claims are paid in the next payroll cycle. Client meals are capped at 1,500 per person.
Personal expenses, fines and alcohol are not reimbursable.
""",
"finance_travel_policy": """# Travel Policy
Raise a travel request at least 7 days before departure through Finance > Travel > New Request.
Flights are economy class. Train travel is preferred for journeys under 500 km.
Hotel limits per night: Tier 1 cities 6,000; Tier 2 cities 4,000. Daily allowance is 1,200.
Travel advances up to 50% of the estimated cost can be requested and must be settled within 15 days of return.
""",
"finance_payment_terms_credit": """# Customer Payment Terms and Credit
Standard payment terms are 30 days net from the invoice date. New customers start on 15-day terms until three orders are completed.
Orders that would exceed a customer's credit limit are placed on hold until payment is received or the limit is raised.
Credit limit increases require approval from the Finance Manager and a review of the last 12 months of payment history.
Government customers may be offered 60-day terms with Finance Director approval.
""",
"finance_late_payment_followup": """# Late Payment Follow-up Policy
Day 1 after due date: automated reminder email is sent from Finance > Receivables > Reminders.
Day 7: second reminder and a phone call from the account manager.
Day 15: escalation to the Sales Head and Finance Manager.
Day 30: the customer is placed on credit hold and no new orders are confirmed until the overdue amount is cleared.
Interest of 1.5% per month may be charged on invoices overdue by more than 30 days.
""",
"finance_credit_note_cancel_invoice": """# Credit Notes and Invoice Cancellation
An invoice can be cancelled only if it has not been paid, and needs Finance Manager approval.
To issue a credit note go to Finance > Receivables > Credit Note > New. You need the original invoice number,
a reason code (damaged goods, pricing error, return, goodwill) and supporting documents.
Credit notes above 10,000 need Finance Director approval. Credit notes are applied against the next invoice or refunded within 7 working days.
""",
"finance_month_end_close": """# Month-End Close Process
Cut-off is 18:00 on the last working day of the month. Steps:
1. Complete all goods receipts and post pending vendor invoices.
2. Reconcile bank accounts and customer receipts.
3. Book accruals for services received but not invoiced.
4. Run the depreciation job under Finance > Fixed Assets.
5. Review the trial balance and clear suspense accounts.
6. Lock the accounting period by working day 3. Reopening a locked period needs Finance Director approval.
""",
"sales_create_sales_order": """# How to create a sales order
1. Go to Sales > Orders > New Sales Order.
2. Select the customer. The system shows the credit limit and any overdue balance.
3. Add products and quantities. Stock availability is shown per warehouse.
4. Apply any discount (see the discount approval policy) and choose the delivery date.
5. Click Submit for Confirmation.
Order statuses move from pending to confirmed, shipped and delivered. Orders can be cancelled only before shipment.
""",
"sales_discount_policy": """# Discount Approval Policy
Salespersons may give discounts up to 5% on list price. Discounts between 5% and 10% need the Sales Manager's approval.
Discounts above 10% need the Finance Director. Every discount above 5% needs a documented reason in the order notes.
Orders with unapproved discounts stay in pending status and cannot be confirmed.
""",
"sales_return_policy": """# Return and Refund Policy
Goods can be returned within 14 days of delivery. Damaged or defective goods must be reported within 48 hours with photos.
Non-defective returns attract a 10% restocking fee. Returned goods must be in original packaging.
Approved returns are processed through Sales > Returns > New RMA, inspected on receipt, and refunded by credit note within 7 working days.
""",
"procurement_po_approval": """# Purchase Order Approval Limits
Purchase orders up to 5,000 are approved by the Purchasing Officer. From 5,001 to 25,000 the Procurement Manager approves.
Above 25,000 the Finance Director must approve. Purchases above 10,000 require three written quotations.
Split orders to avoid approval limits are not allowed and are flagged during audit.
""",
"procurement_vendor_onboarding": """# Vendor Onboarding
New vendors must supply a company registration certificate, tax registration number, cancelled cheque or bank letter and two business references.
Fill in the Vendor Request form under Procurement > Vendors > New. Compliance review takes 5 working days.
New vendors start with a rating of 3.0 and are reviewed after their first three deliveries.
""",
"procurement_late_delivery": """# Late Delivery and Vendor Penalties
If a delivery is more than 3 days late the buyer notifies the vendor in writing. A penalty of 0.5% of the PO value per day applies, capped at 5%.
After 14 days of delay the PO can be cancelled and re-issued to another vendor. Repeated late deliveries reduce the vendor rating
and vendors rated below 2.5 are placed on review.
""",
"inventory_reorder_safety_stock": """# Reorder Process and Safety Stock
Each product has a reorder level. When total stock across warehouses falls below it, the system flags the item on the Inventory > Low Stock dashboard.
Safety stock is set to two weeks of average demand. The buyer then raises a purchase requisition under Procurement > Requisitions > New,
which is converted into a purchase order after approval. Typical vendor lead time is 7 to 21 days.
""",
"inventory_goods_receipt": """# Goods Receipt (GRN)
When goods arrive, the warehouse team verifies them against the purchase order for quantity and condition.
Quality inspection must be completed within 24 hours. Post the GRN under Inventory > Receipts > New GRN.
Vendor invoices are paid only after three-way match between PO, GRN and invoice. Short or damaged deliveries are recorded as discrepancies.
""",
"inventory_master_data": """# Product Master Data Changes
Edit products under Inventory > Products > Edit. Price changes need Sales Manager approval and take effect from the next day.
SKU codes cannot be changed once a product has transactions. To retire a product mark it Inactive instead of deleting it.
""",
"it_access_password_support": """# Access, Passwords and Support
Passwords expire every 90 days. Use the Forgot Password link on the login page or contact the IT helpdesk.
To request access to a module, submit an access request on the IT portal with your manager's approval; it takes 2 working days.
Raise a support ticket under Help > Raise Ticket. Priority 1 (system down) gets a response within 1 hour, Priority 2 within 4 hours.
""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ERP_DIR))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "docs"), exist_ok=True)

    con = build_db(os.path.join(a.out, "erp.db"), rng)
    with open(os.path.join(a.out, "schema.sql"), "w") as f:
        f.write(SCHEMA.strip() + "\n")
    for name, text in DOCS.items():
        with open(os.path.join(a.out, "docs", name + ".md"), "w") as f:
            f.write(text.strip() + "\n")
    stats = {"tables": {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for (t,) in
                        con.execute("SELECT name FROM sqlite_master WHERE type='table'")}}
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
