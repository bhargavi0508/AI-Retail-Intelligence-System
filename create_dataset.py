import csv
import random
from datetime import date, timedelta

# -----------------------------
# SETTINGS
# -----------------------------

NUM_ROWS = 500

random.seed(42)

# -----------------------------
# PRODUCT + MODEL DATA
# -----------------------------

products = {
    "Laptop": {
        "category": "Computers",
        "models": {
            "ASUS VivoBook": 55000,
            "HP Pavilion": 65000,
            "MacBook Air": 95000,
            "Dell Inspiron": 60000,
            "Lenovo IdeaPad": 58000
        }
    },

    "Mobile": {
        "category": "Smartphones",
        "models": {
            "iPhone 16": 75000,
            "Samsung Galaxy S25": 70000,
            "OnePlus 13": 60000,
            "Google Pixel 9": 65000,
            "Xiaomi 15": 45000
        }
    },

    "Tablet": {
        "category": "Tablets",
        "models": {
            "iPad Air": 60000,
            "Galaxy Tab": 45000,
            "OnePlus Pad": 40000,
            "Lenovo Tab": 30000
        }
    },

    "Headphones": {
        "category": "Audio",
        "models": {
            "Sony WH-1000XM5": 30000,
            "AirPods Pro": 25000,
            "JBL Tune": 8000,
            "Bose QC": 28000
        }
    },

    "Smartwatch": {
        "category": "Wearables",
        "models": {
            "Apple Watch": 45000,
            "Galaxy Watch": 30000,
            "Garmin": 35000,
            "OnePlus Watch": 18000
        }
    },

    "Keyboard": {
        "category": "Accessories",
        "models": {
            "Logitech K380": 3500,
            "HP Wireless Keyboard": 2500,
            "Dell Keyboard": 2200,
            "ASUS Keyboard": 3000
        }
    },

    "Monitor": {
        "category": "Displays",
        "models": {
            "Dell 24 Monitor": 15000,
            "LG UltraGear": 25000,
            "Samsung Odyssey": 30000,
            "ASUS TUF": 28000
        }
    },

    "Mouse": {
        "category": "Accessories",
        "models": {
            "Logitech M331": 1500,
            "HP Wireless Mouse": 1200,
            "Dell WM126": 1100,
            "ASUS TUF Mouse": 1800
        }
    }
}

# -----------------------------
# BUSINESS DATA
# -----------------------------

region_cities = {
    "AP": [
        "Guntur",
        "Vijayawada",
        "Visakhapatnam",
        "Rajahmundry"
    ],

    "TS": [
        "Hyderabad",
        "Warangal",
        "Nizamabad",
        "Karimnagar"
    ],

    "KA": [
        "Bangalore",
        "Mysore",
        "Mangalore",
        "Hubli"
    ],

    "TN": [
        "Chennai",
        "Coimbatore",
        "Madurai",
        "Salem"
    ]
}

customer_types = [
    "Individual",
    "Business",
    "Student"
]

sales_channels = [
    "Online",
    "Retail Store",
    "Dealer"
]

payment_methods = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash",
    "Net Banking"
]

salespersons = [
    "Anil",
    "Priya",
    "Rahul",
    "Sneha",
    "Kiran",
    "Divya",
    "Arjun",
    "Meena"
]

order_statuses = [
    "Completed",
    "Completed",
    "Completed",
    "Completed",
    "Cancelled"
]

# -----------------------------
# CSV COLUMNS
# -----------------------------

columns = [
    "Order_ID",
    "Date",
    "Product",
    "Product_Model",
    "Category",
    "Region",
    "City",
    "Units_Sold",
    "Unit_Price",
    "Discount_Percent",
    "Revenue",
    "Current_Stock",
    "Customer_Type",
    "Sales_Channel",
    "Payment_Method",
    "Customer_Age",
    "Salesperson",
    "Order_Status"
]

# -----------------------------
# CREATE DATA
# -----------------------------

rows = []

start_date = date(2025, 1, 1)

for i in range(1, NUM_ROWS + 1):

    product = random.choice(list(products.keys()))

    product_info = products[product]

    model = random.choice(list(product_info["models"].keys()))

    unit_price = product_info["models"][model]

    region = random.choice(list(region_cities.keys()))

    city = random.choice(region_cities[region])

    units_sold = random.randint(1, 15)

    discount = random.choice([
        0, 5, 5, 10, 10, 15, 20
    ])

    revenue = (
        units_sold
        * unit_price
        * (1 - discount / 100)
    )

    current_stock = random.randint(5, 100)

    customer_type = random.choice(customer_types)

    sales_channel = random.choice(sales_channels)

    payment_method = random.choice(payment_methods)

    customer_age = random.randint(18, 60)

    salesperson = random.choice(salespersons)

    order_status = random.choice(order_statuses)

    order_date = start_date + timedelta(
        days=random.randint(0, 364)
    )

    rows.append([
        f"ORD{i:04d}",
        order_date.strftime("%Y-%m-%d"),
        product,
        model,
        product_info["category"],
        region,
        city,
        units_sold,
        unit_price,
        discount,
        round(revenue, 2),
        current_stock,
        customer_type,
        sales_channel,
        payment_method,
        customer_age,
        salesperson,
        order_status
    ])

# -----------------------------
# SAVE CSV
# -----------------------------

output_file = "data/electronics_sales_data.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(columns)

    writer.writerows(rows)

# -----------------------------
# SUCCESS MESSAGE
# -----------------------------

print()
print("=" * 50)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 50)

print(f"File    : {output_file}")
print(f"Rows    : {NUM_ROWS}")
print(f"Columns : {len(columns)}")

print()
print("New important fields:")
print("- Product_Model")
print("- Current_Stock")

print()
print("Product Models:")
for product, info in products.items():
    print(f"\n{product}:")
    for model in info["models"]:
        print(f"  - {model}")

print()
print("Dataset is a simulated electronics retail dataset.")
print("It is designed for demonstrating real-world")
print("sales, demand and inventory analysis.")