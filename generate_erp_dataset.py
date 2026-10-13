#!/usr/bin/env python3
"""
Generate a synthetic ERP dataset for a hybrid RAG + text-to-SQL chatbot.

Outputs (in --out directory):
  erp.db                 SQLite database (sales, purchasing, inventory, finance, HR)
  schema.sql             DDL of the database
  docs/*.md              Policy / how-to documents for the RAG side
  router_train.jsonl     Labeled queries: sql | rag | both | neither
  router_val.jsonl
  router_test.jsonl
  stats.json             Counts per split/label

Splits are done by TEMPLATE (all phrasings of a template go to one split), so the
test set measures generalisation to unseen question patterns, not memorisation.

Usage:  python generate_erp_dataset.py --out erp_dataset --seed 42
"""
import argparse
import json
import os
import random
import sqlite3
from collections import Counter
from datetime import date, timedelta

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

# --------------------------------------------------------------------------- #
# 3. ROUTER TEMPLATES
# --------------------------------------------------------------------------- #
# (id, phrasings, sql)
SQL_T = [
 ("sales_month", ["What were total sales in {month} {year}?", "total sales for {month} {year}",
   "how much revenue did we make in {month} {year}", "show me the {month} {year} sales total"],
  "SELECT ROUND(SUM(total_amount),2) FROM sales_orders WHERE strftime('%Y-%m', order_date) = '{ym}' AND status != 'cancelled';"),
 ("top_customers", ["Who are our top {n} customers by revenue?", "top {n} customers by sales",
   "list the {n} biggest customers by order value", "which {n} customers bought the most?"],
  "SELECT c.name, ROUND(SUM(so.total_amount),2) AS revenue FROM sales_orders so JOIN customers c ON c.customer_id = so.customer_id WHERE so.status != 'cancelled' GROUP BY c.customer_id ORDER BY revenue DESC LIMIT {n};"),
 ("stock_product", ["How many {product} do we have in stock?", "current stock of {product}",
   "what's the available quantity for {product}?", "inventory level of {product}"],
  "SELECT COALESCE(SUM(i.quantity),0) FROM inventory i JOIN products p ON p.product_id = i.product_id WHERE p.name = '{product}';"),
 ("below_reorder", ["Which products are below their reorder level?", "list items that need restocking",
   "show products with stock under the reorder point", "what items are running low on stock?"],
  "SELECT p.name, SUM(i.quantity) AS qty, p.reorder_level FROM products p JOIN inventory i ON i.product_id = p.product_id GROUP BY p.product_id HAVING qty < p.reorder_level;"),
 ("overdue_count", ["How many invoices are overdue?", "number of overdue invoices",
   "count of invoices past their due date", "how many unpaid invoices are past due?"],
  "SELECT COUNT(*) FROM invoices WHERE status = 'overdue';"),
 ("outstanding_customer", ["What is the outstanding balance for {customer}?", "how much does {customer} owe us?",
   "unpaid invoice total for {customer}", "total receivable from {customer}"],
  "SELECT ROUND(SUM(i.amount),2) FROM invoices i JOIN customers c ON c.customer_id = i.customer_id WHERE c.name = '{customer}' AND i.status IN ('unpaid','overdue');"),
 ("dept_headcount", ["How many employees are in the {dept} department?", "headcount of {dept}",
   "number of active staff in {dept}", "how many people work in {dept}?"],
  "SELECT COUNT(*) FROM employees e JOIN departments d ON d.dept_id = e.dept_id WHERE d.name = '{dept}' AND e.status = 'active';"),
 ("open_po_vendor", ["How many open purchase orders do we have with {vendor}?", "open POs for {vendor}",
   "pending purchase orders from {vendor}", "count POs not yet fully received from {vendor}"],
  "SELECT COUNT(*) FROM purchase_orders po JOIN vendors v ON v.vendor_id = po.vendor_id WHERE v.name = '{vendor}' AND po.status IN ('open','partially_received');"),
 ("avg_order_year", ["What was the average order value in {year}?", "average sales order amount for {year}",
   "mean order value {year}", "avg order size in {year}"],
  "SELECT ROUND(AVG(total_amount),2) FROM sales_orders WHERE strftime('%Y', order_date) = '{year}' AND status != 'cancelled';"),
 ("pending_orders", ["How many sales orders are pending?", "number of orders awaiting confirmation",
   "count pending sales orders", "how many orders are still pending?"],
  "SELECT COUNT(*) FROM sales_orders WHERE status = 'pending';"),
 ("revenue_category", ["Show revenue by product category for {year}", "category-wise sales in {year}",
   "how did each product category perform in {year}?", "revenue breakdown by category {year}"],
  "SELECT p.category, ROUND(SUM(soi.quantity * soi.unit_price),2) AS revenue FROM sales_order_items soi JOIN sales_orders so ON so.so_id = soi.so_id JOIN products p ON p.product_id = soi.product_id WHERE so.status != 'cancelled' AND strftime('%Y', so.order_date) = '{year}' GROUP BY p.category ORDER BY revenue DESC;"),
 ("manager_of", ["Who is {employee}'s manager?", "who does {employee} report to?",
   "line manager of {employee}", "show the reporting manager for {employee}"],
  "SELECT m.name FROM employees e JOIN employees m ON e.manager_id = m.emp_id WHERE e.name = '{employee}';"),
 ("pending_leaves", ["How many leave requests are pending approval?", "count pending leave requests",
   "number of leaves awaiting approval", "how many leave applications are waiting?"],
  "SELECT COUNT(*) FROM leave_requests WHERE status = 'pending';"),
 ("payments_month", ["How much did we collect in payments in {month} {year}?", "total payments received in {month} {year}",
   "collections for {month} {year}", "cash received during {month} {year}"],
  "SELECT ROUND(SUM(amount),2) FROM payments WHERE strftime('%Y-%m', payment_date) = '{ym}';"),
 ("product_price", ["What is the unit price of {product}?", "price of {product}",
   "how much does {product} cost?", "selling price for {product}"],
  "SELECT unit_price FROM products WHERE name = '{product}';"),
 ("customers_city", ["List customers in {city}", "which customers are based in {city}?",
   "show me all clients from {city}", "customers located in {city}"],
  "SELECT name FROM customers WHERE city = '{city}';"),
 ("latest_order", ["What was the latest order from {customer}?", "most recent order placed by {customer}",
   "show {customer}'s last order", "when did {customer} last order?"],
  "SELECT so.so_id, so.order_date, so.total_amount FROM sales_orders so JOIN customers c ON c.customer_id = so.customer_id WHERE c.name = '{customer}' ORDER BY so.order_date DESC LIMIT 1;"),
 ("vendor_spend", ["How much did we spend with {vendor} in {year}?", "total purchases from {vendor} in {year}",
   "purchase spend for {vendor} {year}", "{vendor} total PO value in {year}"],
  "SELECT ROUND(SUM(po.total_amount),2) FROM purchase_orders po JOIN vendors v ON v.vendor_id = po.vendor_id WHERE v.name = '{vendor}' AND po.status != 'cancelled' AND strftime('%Y', po.order_date) = '{year}';"),
 ("top_paid", ["Who are the {n} highest paid employees?", "top {n} salaries",
   "list the {n} employees with the highest salary", "show top {n} earners"],
  "SELECT name, salary FROM employees WHERE status = 'active' ORDER BY salary DESC LIMIT {n};"),
 ("new_customers", ["How many new customers did we add in {year}?", "customers onboarded in {year}",
   "new clients acquired in {year}", "number of customers created in {year}"],
  "SELECT COUNT(*) FROM customers WHERE strftime('%Y', created_at) = '{year}';"),
 ("warehouse_stock", ["What is the total stock in {warehouse}?", "total units stored at {warehouse}",
   "how many items are in {warehouse}?", "inventory count for {warehouse}"],
  "SELECT SUM(i.quantity) FROM inventory i JOIN warehouses w ON w.warehouse_id = i.warehouse_id WHERE w.name = '{warehouse}';"),
 ("top_salesperson", ["Who is the top salesperson in {year}?", "best performing sales rep for {year}",
   "which salesperson sold the most in {year}?", "top sales rep by revenue {year}"],
  "SELECT e.name, ROUND(SUM(so.total_amount),2) AS rev FROM sales_orders so JOIN employees e ON e.emp_id = so.salesperson_id WHERE strftime('%Y', so.order_date) = '{year}' AND so.status != 'cancelled' GROUP BY e.emp_id ORDER BY rev DESC LIMIT 1;"),
 ("unpaid_total", ["What is the total amount of unpaid invoices?", "total receivables outstanding",
   "how much money is pending from customers?", "sum of all open invoices"],
  "SELECT ROUND(SUM(amount),2) FROM invoices WHERE status IN ('unpaid','overdue');"),
 ("inventory_value", ["What is the total value of our inventory?", "inventory valuation at cost",
   "how much is our stock worth?", "total stock value"],
  "SELECT ROUND(SUM(i.quantity * p.unit_cost),2) FROM inventory i JOIN products p ON p.product_id = i.product_id;"),
 ("late_pos", ["Which purchase orders are past their expected delivery date?", "list late purchase orders",
   "show overdue POs", "which POs should have arrived already?"],
  "SELECT po_id, vendor_id, expected_date FROM purchase_orders WHERE status IN ('open','partially_received') AND expected_date < '2026-10-11';"),
 ("customer_order_count", ["How many orders has {customer} placed in {year}?", "number of orders by {customer} in {year}",
   "{customer} order count {year}", "count {customer}'s orders for {year}"],
  "SELECT COUNT(*) FROM sales_orders so JOIN customers c ON c.customer_id = so.customer_id WHERE c.name = '{customer}' AND strftime('%Y', so.order_date) = '{year}';"),
 ("vendor_rating", ["What is the rating of {vendor}?", "show {vendor}'s vendor rating",
   "how is {vendor} rated?", "vendor score for {vendor}"],
  "SELECT rating FROM vendors WHERE name = '{vendor}';"),
 ("best_seller", ["What is our best selling product in {year}?", "top selling item in {year}",
   "which product sold the most units in {year}?", "best seller {year}"],
  "SELECT p.name, SUM(soi.quantity) AS units FROM sales_order_items soi JOIN sales_orders so ON so.so_id = soi.so_id JOIN products p ON p.product_id = soi.product_id WHERE so.status != 'cancelled' AND strftime('%Y', so.order_date) = '{year}' GROUP BY p.product_id ORDER BY units DESC LIMIT 1;"),
]

# (id, phrasings, doc)
RAG_T = [
 ("annual_leave", ["How many days of annual leave am I entitled to?", "what is the leave policy?",
   "how does leave carry forward work?", "what's the sick leave policy and do I need a medical certificate?"], "hr_leave_policy"),
 ("apply_leave", ["How do I apply for leave in the ERP?", "steps to submit a leave request",
   "where do I request time off?", "how to raise a leave application?"], "hr_apply_leave_steps"),
 ("po_approval", ["What approval is needed for a large purchase order?", "who approves purchase orders above 25,000?",
   "what are the PO approval limits?", "how many quotations are required for big purchases?"], "procurement_po_approval"),
 ("create_so", ["How do I create a sales order?", "steps to enter a new sales order in the system",
   "walk me through creating a customer order", "where do I create a sales order in the ERP?"], "sales_create_sales_order"),
 ("return_policy", ["What is the return policy for damaged goods?", "how long do customers have to return items?",
   "is there a restocking fee for returns?", "how do I process a customer return?"], "sales_return_policy"),
 ("password_reset", ["How do I reset my ERP password?", "I forgot my password, what should I do?",
   "how often do passwords expire?", "how do I get access to a new module?"], "it_access_password_support"),
 ("payment_terms", ["What are our standard payment terms for customers?", "what payment terms do new customers get?",
   "how do I get a credit limit increased?", "what happens when an order exceeds the credit limit?"], "finance_payment_terms_credit"),
 ("vendor_onboarding", ["How do I onboard a new vendor?", "what documents are needed to register a supplier?",
   "what is the vendor approval process?", "how long does vendor compliance review take?"], "procurement_vendor_onboarding"),
 ("expense_policy", ["What is the expense reimbursement policy?", "how do I submit an expense claim?",
   "do I need receipts for expense claims?", "how long do I have to file expenses?"], "finance_expense_reimbursement"),
 ("credit_note", ["How do I issue a credit note?", "what documents are needed for a credit note?",
   "who approves credit notes?", "steps to create a credit note for a customer"], "finance_credit_note_cancel_invoice"),
 ("cancel_invoice", ["Can I cancel an invoice after it is issued?", "how do I cancel an invoice?",
   "what approval is needed to cancel an invoice?", "is it possible to void a paid invoice?"], "finance_credit_note_cancel_invoice"),
 ("month_end", ["How does the month-end close process work?", "what are the steps for closing the books?",
   "what is the cut-off time for month end?", "when is the accounting period locked?"], "finance_month_end_close"),
 ("reorder_process", ["What is the reorder process when stock is low?", "how do I raise a purchase requisition?",
   "how is safety stock calculated?", "what happens when an item falls below reorder level?"], "inventory_reorder_safety_stock"),
 ("remote_work", ["What is the remote work policy?", "how many days can I work from home?",
   "what are the core working hours?", "who is eligible for hybrid work?"], "hr_remote_work"),
 ("grn", ["How do I post a goods receipt?", "what is the GRN process?",
   "what is three-way matching?", "how are short deliveries recorded?"], "inventory_goods_receipt"),
 ("discount_policy", ["What is the discount approval policy?", "how much discount can a salesperson give?",
   "who approves discounts above 10%?", "do discounts need a documented reason?"], "sales_discount_policy"),
 ("notice_period", ["What is the notice period for resignation?", "how do I submit my resignation in the system?",
   "can the notice period be bought out?", "what is the exit clearance process?"], "hr_notice_period"),
 ("master_data", ["How do I update product master data?", "can I change a SKU code?",
   "who approves product price changes?", "how do I retire a product?"], "inventory_master_data"),
 ("late_delivery", ["What happens if a vendor delivers late?", "what is the penalty for late vendor delivery?",
   "when can we cancel a delayed purchase order?", "how are vendor ratings affected by delays?"], "procurement_late_delivery"),
 ("travel_policy", ["What is the travel policy?", "how do I raise a travel request?",
   "what is the hotel limit per night?", "can I get a travel advance?"], "finance_travel_policy"),
 ("support_ticket", ["How do I raise a support ticket for the ERP?", "who do I contact when the system is down?",
   "what is the response time for priority 1 issues?", "where do I report an ERP bug?"], "it_access_password_support"),
 ("late_followup", ["What is our late payment follow-up policy?", "when is a customer put on credit hold?",
   "do we charge interest on overdue invoices?", "what are the escalation steps for unpaid invoices?"], "finance_late_payment_followup"),
]

# (id, phrasings, doc)
BOTH_T = [
 ("overdue_policy", ["Which customers have overdue invoices and what is our policy for following up on late payments?",
   "List customers with overdue invoices and explain the escalation steps.",
   "who is late on payment, and how should I follow up according to policy?"], "finance_late_payment_followup"),
 ("stock_reorder", ["How many {product} are in stock, and what is the reorder process if it's low?",
   "Check the stock of {product} and tell me how to raise a requisition.",
   "current quantity of {product} and our safety stock policy"], "inventory_reorder_safety_stock"),
 ("leave_pending", ["How many leave requests are pending and what is the approval process?",
   "count the pending leaves and explain how leave approval works",
   "number of pending leave applications, plus the leave carry-forward rules"], "hr_leave_policy"),
 ("customer_terms", ["How much does {customer} owe us and what are the standard payment terms?",
   "show the outstanding balance for {customer} and explain our credit policy",
   "what is {customer}'s unpaid total, and when do we put customers on credit hold?"], "finance_payment_terms_credit"),
 ("vendor_late", ["Show open purchase orders for {vendor} and explain what happens when a vendor delivers late.",
   "how many open POs does {vendor} have, and what is the late delivery penalty?",
   "open orders with {vendor} plus the policy on delayed deliveries"], "procurement_late_delivery"),
 ("dept_notice", ["How many people are in {dept} and what is the notice period for resignation?",
   "headcount of {dept} and the exit process for employees",
   "number of active employees in {dept}, also how long is the notice period?"], "hr_notice_period"),
 ("pending_discount", ["How many sales orders are pending and what is the discount approval policy?",
   "count pending orders and tell me who approves discounts above 10%",
   "pending sales orders now, and the rules for discount approval"], "sales_discount_policy"),
 ("vendor_spend_onboard", ["How much did we spend with {vendor} in {year}, and how do I onboard a similar vendor?",
   "total purchases from {vendor} in {year} and the vendor onboarding steps",
   "{vendor} spend for {year}, plus what documents a new supplier needs"], "procurement_vendor_onboarding"),
 ("below_reorder_req", ["Which products are below reorder level and how do I raise a purchase requisition?",
   "list low stock items and explain how to create a requisition",
   "what needs restocking and what is the procedure to order it?"], "inventory_reorder_safety_stock"),
 ("overdue_credit_hold", ["How many invoices are overdue and when is a customer put on credit hold?",
   "count overdue invoices and explain the credit hold rule",
   "number of invoices past due plus the interest policy on late payments"], "finance_late_payment_followup"),
 ("revenue_close", ["What was our revenue in {month} {year} and how is the month-end close performed?",
   "total sales for {month} {year}, and what are the closing steps?",
   "{month} {year} sales total and when the period gets locked"], "finance_month_end_close"),
 ("top_cust_credit", ["Who are our top {n} customers and what is the process to raise a credit limit?",
   "list the top {n} customers by revenue and explain how credit limit increases are approved",
   "{n} biggest customers, plus the credit limit approval rules"], "finance_payment_terms_credit"),
 ("po_value_approval", ["How many open purchase orders does {vendor} have and who approves large POs?",
   "open POs for {vendor}, plus the approval limits for purchase orders",
   "pending orders with {vendor} and the quotation requirement for big purchases"], "procurement_po_approval"),
 ("return_orders", ["How many orders did {customer} place in {year} and what is the return policy?",
   "order count for {customer} in {year} and how long they have to return goods",
   "{customer} orders in {year}, also explain the restocking fee"], "sales_return_policy"),
]

# (id, phrasings) -- non-ERP / chit-chat
NEITHER_T = [
 ("greeting", ["hi", "hello", "hey there", "good morning", "good evening", "hello, how are you?"]),
 ("thanks", ["thanks", "thank you so much", "ok thanks", "great, thanks!", "perfect, thank you"]),
 ("identity", ["who are you?", "what can you do?", "are you a bot?", "what is your name?", "what are you able to help with?"]),
 ("bye", ["bye", "goodbye", "see you later", "talk to you tomorrow", "that's all for now"]),
 ("general_knowledge", ["what is the capital of France?", "who wrote Hamlet?", "how far is the moon from earth?",
                        "who painted the Mona Lisa?", "what is the tallest mountain in the world?"]),
 ("joke", ["tell me a joke", "say something funny", "make me laugh", "do you know any good riddles?"]),
 ("coding", ["write a python function to reverse a string", "how do I center a div in css?",
             "explain recursion with an example", "what is the difference between a list and a tuple in python?"]),
 ("weather", ["what's the weather like today?", "will it rain tomorrow?", "is it going to be hot this weekend?"]),
 ("math_units", ["what is 15% of 2400?", "convert 5 km to miles", "what is the square root of 144?", "how many days are in a leap year?"]),
 ("translation", ["translate 'good morning' to Spanish", "how do you say thank you in German?", "what is 'welcome' in French?"]),
 ("recommend", ["recommend a good movie", "suggest a restaurant for dinner", "what book should I read next?", "any good podcasts?"]),
 ("ack", ["ok", "cool", "got it", "alright", "sure", "nice"]),
 ("cooking", ["how do I make pasta?", "what is a good recipe for dal?", "how long should I boil eggs?"]),
 ("health_life", ["how can I sleep better?", "give me tips to stay productive", "how do I learn a new language quickly?"]),
]

PREFIXES = ["please ", "hey, ", "can you tell me ", "I want to know ", "quick question: ", "hi, ",
            "could you show me ", "just checking, "]


def augment(q, rng):
    if rng.random() < 0.25:
        q = rng.choice(PREFIXES) + q[0].lower() + q[1:]
    if rng.random() < 0.2:
        q = q.lower()
    if q.endswith("?") and rng.random() < 0.3:
        q = q[:-1]
    return q


def load_slot_sources(con):
    c = con.cursor()
    q = lambda sql: [r[0] for r in c.execute(sql).fetchall()]
    return {
        "customer": q("SELECT name FROM customers"),
        "vendor": q("SELECT name FROM vendors"),
        "product": q("SELECT name FROM products"),
        "dept": q("SELECT name FROM departments"),
        "employee": q("SELECT name FROM employees WHERE manager_id IS NOT NULL"),
        "city": q("SELECT DISTINCT city FROM customers"),
        "warehouse": q("SELECT name FROM warehouses"),
    }


def sample_slots(rng, src):
    year = rng.choice([2024, 2025, 2026])
    month_idx = rng.randint(1, 9 if year == 2026 else 12)
    s = {k: rng.choice(v) for k, v in src.items()}
    s.update(year=year, month=MONTHS[month_idx - 1], ym=f"{year}-{month_idx:02d}", n=rng.choice([3, 5, 10]))
    return s


def split_ids(ids, rng):
    ids = sorted(ids)
    rng.shuffle(ids)
    n = len(ids)
    nt, nv = max(1, round(n * 0.15)), max(1, round(n * 0.15))
    return {"test": ids[:nt], "val": ids[nt:nt + nv], "train": ids[nt + nv:]}


def gen_rows(tpl, label, rng, src, target, sql=None, doc=None):
    rows, seen, attempts = [], set(), 0
    while len(rows) < target and attempts < target * 25:
        attempts += 1
        s = sample_slots(rng, src)
        q = augment(rng.choice(tpl[1]).format(**s), rng)
        key = q.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        row = {"query": q, "label": label, "template": tpl[0]}
        if sql:
            row["sql"] = sql.format(**s)
        if doc:
            row["doc"] = doc + ".md"
        rows.append(row)
    return rows


def build_router_dataset(con, rng, outdir):
    src = load_slot_sources(con)
    groups = {
        "sql": ({t[0]: t for t in SQL_T}, 90),
        "rag": ({t[0]: t for t in RAG_T}, 60),
        "both": ({t[0]: t for t in BOTH_T}, 60),
        "neither": ({t[0]: t for t in NEITHER_T}, 40),
    }
    splits = {"train": [], "val": [], "test": []}
    seen_global = set()
    for label, (tmap, target) in groups.items():
        assign = split_ids(list(tmap), rng)
        for split, ids in assign.items():
            for tid in ids:
                t = tmap[tid]
                sql = t[2] if label == "sql" else None
                doc = t[2] if label in ("rag", "both") else None
                for r in gen_rows(t, label, rng, src, target, sql, doc):
                    k = r["query"].lower().strip()
                    if k in seen_global:
                        continue
                    seen_global.add(k)
                    splits[split].append(r)

    # validate gold SQL executes; drop failures
    bad = empty = 0
    cur = con.cursor()
    for split, rows in splits.items():
        keep = []
        for r in rows:
            if "sql" in r:
                try:
                    res = cur.execute(r["sql"]).fetchall()
                    if not res or all(v is None for v in res[0]):
                        empty += 1
                        r["result_empty"] = True
                except sqlite3.Error:
                    bad += 1
                    continue
            keep.append(r)
        rng.shuffle(keep)
        splits[split] = keep

    stats = {}
    for split, rows in splits.items():
        for i, r in enumerate(rows):
            r["id"] = f"{split}-{i:05d}"
        with open(os.path.join(outdir, f"router_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = dict(Counter(r["label"] for r in rows), total=len(rows))
    stats["sql_failed_and_dropped"] = bad
    stats["sql_with_empty_result"] = empty
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="erp_dataset")
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
    stats = build_router_dataset(con, rng, a.out)
    stats["tables"] = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for (t,) in
                       con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
