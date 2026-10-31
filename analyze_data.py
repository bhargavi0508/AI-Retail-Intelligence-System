import pandas as pd

# Load the sales dataset
file_path = "data/sales_data.csv"
df = pd.read_csv(file_path)

print("\n===== SALES DATA ANALYSIS =====\n")

# 1. Show first 5 records
print("First 5 Records:")
print(df.head())

# 2. Dataset information
print("\nDataset Information:")
print(df.info())

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Total revenue
total_revenue = df["Revenue"].sum()
print(f"\nTotal Revenue: ₹{total_revenue:,.2f}")

# 5. Total units sold
total_units = df["Units_Sold"].sum()
print(f"Total Units Sold: {total_units}")

# 6. Revenue by product
product_revenue = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)

print("\nRevenue by Product:")
print(product_revenue)

# 7. Revenue by region
region_revenue = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)

print("\nRevenue by Region:")
print(region_revenue)

# 8. Best-selling product
best_product = product_revenue.idxmax()
print(f"\nBest Performing Product: {best_product}")

# 9. Best-performing region
best_region = region_revenue.idxmax()
print(f"Best Performing Region: {best_region}")