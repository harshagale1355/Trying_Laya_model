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
