import pandas as pd
import plotly.express as px

# Load the sales dataset
file_path = "data/sales_data.csv"
df = pd.read_csv(file_path)

print("\n===== SALES DATA VISUALIZATION =====\n")

# 1. Revenue by Product
product_revenue = df.groupby("Product", as_index=False)["Revenue"].sum()

fig1 = px.bar(
    product_revenue,
    x="Product",
    y="Revenue",
    title="Revenue by Product",
    text_auto=True
)

fig1.show()


# 2. Revenue by Region
region_revenue = df.groupby("Region", as_index=False)["Revenue"].sum()

fig2 = px.pie(
    region_revenue,
    names="Region",
    values="Revenue",
    title="Revenue by Region"
)

fig2.show()


# 3. Units Sold by Product
product_units = df.groupby("Product", as_index=False)["Units_Sold"].sum()

fig3 = px.bar(
    product_units,
    x="Product",
    y="Units_Sold",
    title="Units Sold by Product",
    text_auto=True
)

fig3.show()

print("Charts generated successfully!")