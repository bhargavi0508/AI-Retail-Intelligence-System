
# AI DATA ANALYST AGENT
# Electronics Sales, Demand & Inventory Intelligence System
# ============================================================

import numpy as np
import pandas as pd

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATASET = "data/electronics_sales_data.csv"
OLLAMA_MODEL = "llama3.2"

df = None
current_dataset_path = DEFAULT_DATASET


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(file_path=DEFAULT_DATASET):
    global df, current_dataset_path

    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.strip()

        numeric_columns = [
            "Units_Sold",
            "Unit_Price",
            "Discount_Percent",
            "Revenue",
            "Current_Stock",
            "Customer_Age"
        ]

        for column in numeric_columns:
            if column in data.columns:
                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )

        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(
                data["Date"],
                errors="coerce"
            )

        df = data
        current_dataset_path = file_path

        print(f"\nDataset loaded: {file_path}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        return df

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        return None


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset():

    required_columns = [
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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nMissing columns:")

        for column in missing_columns:
            print(" -", column)

        return False

    return True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    return f"₹{value:,.2f}"


def number(value):
    return f"{value:,.0f}"


def first_item(series):

    items = list(series.items())

    if items:
        return items[0]

    return ("N/A", 0)


def safe_mean(series):

    value = series.mean()

    if pd.isna(value):
        return 0

    return float(value)


# ============================================================
# VERIFIED DATA
# ============================================================

def get_verified_data():

    if df is None:
        return {}

    data = df.copy()

    # --------------------------------------------------------
    # Basic totals
    # --------------------------------------------------------

    total_revenue = data["Revenue"].sum()
    total_units = data["Units_Sold"].sum()

    # --------------------------------------------------------
    # Revenue analysis
    # --------------------------------------------------------

    revenue_by_product = (
        data.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_by_model = (
        data.groupby("Product_Model")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_by_category = (
        data.groupby("Category")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_by_region = (
        data.groupby("Region")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_by_city = (
        data.groupby("City")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_by_channel = (
        data.groupby("Sales_Channel")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_by_customer_type = (
        data.groupby("Customer_Type")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_by_payment = (
        data.groupby("Payment_Method")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_by_salesperson = (
        data.groupby("Salesperson")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    # --------------------------------------------------------
    # Demand analysis
    # --------------------------------------------------------

    demand_by_product = (
        data.groupby("Product")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    demand_by_model = (
        data.groupby("Product_Model")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    demand_by_category = (
        data.groupby("Category")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    demand_by_region = (
        data.groupby("Region")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    demand_by_city = (
        data.groupby("City")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    # --------------------------------------------------------
    # Inventory
    #
    # Current_Stock occurs on transaction rows, so we use
    # average stock per model and explicitly call it
    # "Average Stock".
    # --------------------------------------------------------

    stock_by_model = (
        data.groupby("Product_Model")["Current_Stock"]
        .mean()
        .sort_values(ascending=True)
    )

    stock_by_product = (
        data.groupby("Product")["Current_Stock"]
        .mean()
        .sort_values(ascending=True)
    )

    # --------------------------------------------------------
    # Customer analysis
    # --------------------------------------------------------

    average_customer_age = safe_mean(
        data["Customer_Age"]
    )

    # --------------------------------------------------------
    # Discount
    # --------------------------------------------------------

    average_discount = safe_mean(
        data["Discount_Percent"]
    )

    # --------------------------------------------------------
    # Order status
    # --------------------------------------------------------

    order_status = (
        data["Order_Status"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Monthly revenue
    # --------------------------------------------------------

    date_data = data.dropna(subset=["Date"])

    monthly_revenue = (
        date_data.groupby(
            date_data["Date"].dt.to_period("M")
        )["Revenue"]
        .sum()
        .sort_index()
    )

    # --------------------------------------------------------
    # Monthly demand
    # --------------------------------------------------------

    monthly_demand = (
        date_data.groupby(
            date_data["Date"].dt.to_period("M")
        )["Units_Sold"]
        .sum()
        .sort_index()
    )

    return {

        "Total Revenue": total_revenue,
        "Total Units": total_units,

        "Revenue by Product": revenue_by_product,
        "Revenue by Model": revenue_by_model,
        "Revenue by Category": revenue_by_category,
        "Revenue by Region": revenue_by_region,
        "Revenue by City": revenue_by_city,
        "Revenue by Sales Channel": revenue_by_channel,
        "Revenue by Customer Type": revenue_by_customer_type,
        "Revenue by Payment Method": revenue_by_payment,
        "Revenue by Salesperson": revenue_by_salesperson,

        "Demand by Product": demand_by_product,
        "Demand by Model": demand_by_model,
        "Demand by Category": demand_by_category,
        "Demand by Region": demand_by_region,
        "Demand by City": demand_by_city,

        "Stock by Model": stock_by_model,
        "Stock by Product": stock_by_product,

        "Average Discount": average_discount,
        "Average Customer Age": average_customer_age,

        "Order Status": order_status,

        "Monthly Revenue": monthly_revenue,
        "Monthly Demand": monthly_demand
    }


# ============================================================
# INVENTORY RISK ENGINE
#
# Risk is based on the relationship between demand and stock.
#
# Demand score:
#   Higher demand = higher score
#
# Stock score:
#   Lower stock = higher score
#
# Combined:
#   60% demand + 40% low-stock pressure
#
# This avoids using an arbitrary fixed stock threshold.
# ============================================================

def get_inventory_risk_data():

    verified = get_verified_data()

    demand = verified["Demand by Model"]
    stock = verified["Stock by Model"]

    inventory = pd.concat(
        [
            demand.rename("Demand"),
            stock.rename("Average_Stock")
        ],
        axis=1
    ).dropna()

    if inventory.empty:
        return inventory

    # --------------------------------------------------------
    # Demand percentile
    # Higher demand -> higher score
    # --------------------------------------------------------

    inventory["Demand_Score"] = (
        inventory["Demand"].rank(pct=True)
    )

    # --------------------------------------------------------
    # Stock pressure
    # Lower stock -> higher score
    # --------------------------------------------------------

    inventory["Low_Stock_Score"] = (
        1 - inventory["Average_Stock"].rank(pct=True)
    )

    # --------------------------------------------------------
    # Combined risk score
    # --------------------------------------------------------

    inventory["Risk_Score"] = (
        inventory["Demand_Score"] * 0.60
        +
        inventory["Low_Stock_Score"] * 0.40
    )

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    def classify_risk(score):

        if score >= 0.70:
            return "HIGH"

        elif score >= 0.45:
            return "MEDIUM"

        return "LOW"

    inventory["Risk_Level"] = (
        inventory["Risk_Score"]
        .apply(classify_risk)
    )

    # --------------------------------------------------------
    # Reorder priority
    # --------------------------------------------------------

    def reorder_action(row):

        if row["Risk_Level"] == "HIGH":
            return "REORDER SOON"

        elif row["Risk_Level"] == "MEDIUM":
            return "MONITOR"

        return "NO IMMEDIATE ACTION"

    inventory["Recommendation"] = (
        inventory.apply(
            reorder_action,
            axis=1
        )
    )

    inventory = inventory.sort_values(
        by="Risk_Score",
        ascending=False
    )

    return inventory


# ============================================================
# INVENTORY RISK AGENT
# ============================================================

def inventory_risk_agent(question=""):

    inventory = get_inventory_risk_data()

    if inventory.empty:
        return "Inventory data is not available."

    high_risk = inventory[
        inventory["Risk_Level"] == "HIGH"
    ]

    medium_risk = inventory[
        inventory["Risk_Level"] == "MEDIUM"
    ]

    low_risk = inventory[
        inventory["Risk_Level"] == "LOW"
    ]

    result = []

    result.append(
        "Inventory Risk Analysis"
    )
    result.append(
        "======================="
    )

    result.append(
        "\nRisk is calculated using demand pressure "
        "and relative stock pressure."
    )

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    result.append(
        "\n🔴 HIGH RISK"
    )
    result.append(
        "-----------"
    )

    if high_risk.empty:

        result.append(
            "No high-risk models detected."
        )

    else:

        for model, row in high_risk.iterrows():

            result.append(
                f"{model}\n"
                f"  Demand: {number(row['Demand'])} units\n"
                f"  Average Stock: "
                f"{row['Average_Stock']:.2f} units\n"
                f"  Risk Score: "
                f"{row['Risk_Score']:.2f}\n"
                f"  Action: {row['Recommendation']}"
            )

    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    result.append(
        "\n🟠 MEDIUM RISK"
    )
    result.append(
        "-------------"
    )

    if medium_risk.empty:

        result.append(
            "No medium-risk models detected."
        )

    else:

        for model, row in medium_risk.head(10).iterrows():

            result.append(
                f"{model}\n"
                f"  Demand: {number(row['Demand'])} units\n"
                f"  Average Stock: "
                f"{row['Average_Stock']:.2f} units\n"
                f"  Risk Score: "
                f"{row['Risk_Score']:.2f}\n"
                f"  Action: {row['Recommendation']}"
            )

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    result.append(
        "\n🟢 LOW RISK"
    )
    result.append(
        "----------"
    )

    if low_risk.empty:

        result.append(
            "No low-risk models detected."
        )

    else:

        for model, row in low_risk.head(10).iterrows():

            result.append(
                f"{model} | "
                f"Demand: {number(row['Demand'])} | "
                f"Average Stock: "
                f"{row['Average_Stock']:.2f}"
            )

    return "\n".join(result)


# ============================================================
# REORDER AGENT
# ============================================================

def reorder_agent():

    inventory = get_inventory_risk_data()

    if inventory.empty:
        return "Inventory data is not available."

    reorder_models = inventory[
        inventory["Risk_Level"] == "HIGH"
    ]

    result = []

    result.append(
        "Reorder Recommendations"
    )

    result.append(
        "======================="
    )

    if reorder_models.empty:

        result.append(
            "\nNo models currently require urgent reorder."
        )

        return "\n".join(result)

    result.append(
        "\nThe following models have the highest "
        "inventory risk:\n"
    )

    for model, row in reorder_models.iterrows():

        result.append(
            f"🔴 {model}\n"
            f"   Demand: {number(row['Demand'])} units\n"
            f"   Average Stock: "
            f"{row['Average_Stock']:.2f} units\n"
            f"   Risk Score: "
            f"{row['Risk_Score']:.2f}\n"
            f"   Recommendation: REORDER SOON\n"
        )

    result.append(
        "Note: The system identifies reorder priority "
        "from relative demand and stock pressure. "
        "It does not claim an exact purchase quantity."
    )

    return "\n".join(result)


# ============================================================
# HIGH DEMAND + LOW STOCK
# ============================================================

def high_demand_low_stock_agent():

    inventory = get_inventory_risk_data()

    if inventory.empty:
        return "Inventory data is not available."

    # Sort by risk score
    selected = inventory[
        inventory["Risk_Level"].isin(
            ["HIGH", "MEDIUM"]
        )
    ].sort_values(
        "Risk_Score",
        ascending=False
    )

    result = []

    result.append(
        "High-Demand + Low-Stock Models"
    )

    result.append(
        "=============================="
    )

    if selected.empty:

        result.append(
            "\nNo concerning models detected."
        )

        return "\n".join(result)

    for model, row in selected.iterrows():

        result.append(
            f"\n{model}\n"
            f"  Demand: {number(row['Demand'])} units\n"
            f"  Average Stock: "
            f"{row['Average_Stock']:.2f} units\n"
            f"  Risk: {row['Risk_Level']}\n"
            f"  Risk Score: "
            f"{row['Risk_Score']:.2f}\n"
            f"  Action: {row['Recommendation']}"
        )

    return "\n".join(result)


# ============================================================
# DATA ANALYST AGENT
# ============================================================

def data_analyst_agent(question):

    verified = get_verified_data()

    if not verified:
        return "Dataset is not loaded."

    q = question.lower().strip()

    # --------------------------------------------------------
    # TOTAL REVENUE
    # --------------------------------------------------------

    if (
        "total revenue" in q
        or "overall revenue" in q
        or "total sales" in q
        or "overall sales" in q
    ):

        return (
            f"Total Revenue: "
            f"{money(verified['Total Revenue'])}"
        )

    # --------------------------------------------------------
    # TOTAL UNITS
    # --------------------------------------------------------

    if (
        "total units" in q
        or "units sold" in q
        or "total quantity" in q
    ):

        return (
            f"Total Units Sold: "
            f"{number(verified['Total Units'])}"
        )

    # --------------------------------------------------------
    # HIGHEST REVENUE PRODUCT
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "revenue" in q
        and "product" in q
    ):

        item, value = first_item(
            verified["Revenue by Product"]
        )

        return (
            f"Highest Revenue Product: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # HIGHEST REVENUE MODEL
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "revenue" in q
        and (
            "model" in q
            or "product model" in q
        )
    ):

        item, value = first_item(
            verified["Revenue by Model"]
        )

        return (
            f"Highest Revenue Model: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # HIGHEST DEMAND MODEL
    # --------------------------------------------------------

    if (
        "highest demand" in q
        or "highest demand model" in q
        or "most demanded" in q
        or "best selling model" in q
        or "top selling model" in q
        or (
            "highest" in q
            and "demand" in q
            and "model" in q
        )
    ):

        item, value = first_item(
            verified["Demand by Model"]
        )

        return (
            f"Highest Demand Model: {item}\n"
            f"Units Sold: {number(value)}"
        )

    # --------------------------------------------------------
    # HIGHEST DEMAND PRODUCT
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "demand" in q
        and "product" in q
    ):

        item, value = first_item(
            verified["Demand by Product"]
        )

        return (
            f"Highest Demand Product: {item}\n"
            f"Units Sold: {number(value)}"
        )

    # --------------------------------------------------------
    # LOWEST STOCK MODEL
    # --------------------------------------------------------

    if (
        (
            "lowest stock" in q
            or "least stock" in q
            or "minimum stock" in q
        )
        and "model" in q
    ):

        item, value = first_item(
            verified["Stock by Model"]
        )

        return (
            f"Lowest Stock Model: {item}\n"
            f"Average Current Stock: "
            f"{value:,.2f} units"
        )

    # --------------------------------------------------------
    # HIGHEST REGION
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "region" in q
        and "revenue" in q
    ):

        item, value = first_item(
            verified["Revenue by Region"]
        )

        return (
            f"Highest Revenue Region: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # HIGHEST CITY
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "city" in q
        and "revenue" in q
    ):

        item, value = first_item(
            verified["Revenue by City"]
        )

        return (
            f"Highest Revenue City: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # HIGHEST CHANNEL
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and (
            "channel" in q
            or "sales channel" in q
        )
        and "revenue" in q
    ):

        item, value = first_item(
            verified["Revenue by Sales Channel"]
        )

        return (
            f"Highest Revenue Sales Channel: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # HIGHEST CUSTOMER TYPE
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "customer" in q
        and "revenue" in q
    ):

        item, value = first_item(
            verified["Revenue by Customer Type"]
        )

        return (
            f"Highest Revenue Customer Type: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # PAYMENT METHOD
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "payment" in q
        and "revenue" in q
    ):

        item, value = first_item(
            verified["Revenue by Payment Method"]
        )

        return (
            f"Highest Revenue Payment Method: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # SALESPERSON
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "salesperson" in q
        and "revenue" in q
    ):

        item, value = first_item(
            verified["Revenue by Salesperson"]
        )

        return (
            f"Top Salesperson: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if (
        ("highest" in q or "best" in q or "top" in q)
        and "category" in q
        and "revenue" in q
    ):

        item, value = first_item(
            verified["Revenue by Category"]
        )

        return (
            f"Highest Revenue Category: {item}\n"
            f"Revenue: {money(value)}"
        )

    # --------------------------------------------------------
    # AVERAGE DISCOUNT
    # --------------------------------------------------------

    if (
        "average discount" in q
        or "mean discount" in q
        or "discount percentage" in q
    ):

        return (
            f"Average Discount: "
            f"{verified['Average Discount']:.2f}%"
        )

    # --------------------------------------------------------
    # CUSTOMER AGE
    # --------------------------------------------------------

    if (
        "average age" in q
        or "customer age" in q
    ):

        return (
            f"Average Customer Age: "
            f"{verified['Average Customer Age']:.1f} years"
        )

    # --------------------------------------------------------
    # ORDER STATUS
    # --------------------------------------------------------

    if (
        "order status" in q
        or "orders by status" in q
    ):

        result = []

        for status, count in (
            verified["Order Status"].items()
        ):

            result.append(
                f"{status}: {number(count)} orders"
            )

        return (
            "Order Status:\n"
            + "\n".join(result)
        )

    # --------------------------------------------------------
    # REVENUE BY PRODUCT
    # --------------------------------------------------------

    if "revenue by product" in q:

        result = []

        for product, value in (
            verified["Revenue by Product"].items()
        ):

            result.append(
                f"{product}: {money(value)}"
            )

        return (
            "Revenue by Product:\n"
            + "\n".join(result)
        )

    # --------------------------------------------------------
    # REVENUE BY REGION
    # --------------------------------------------------------

    if "revenue by region" in q:

        result = []

        for region, value in (
            verified["Revenue by Region"].items()
        ):

            result.append(
                f"{region}: {money(value)}"
            )

        return (
            "Revenue by Region:\n"
            + "\n".join(result)
        )

    # --------------------------------------------------------
    # REVENUE BY CITY
    # --------------------------------------------------------

    if "revenue by city" in q:

        result = []

        for city, value in (
            verified["Revenue by City"].items()
        ):

            result.append(
                f"{city}: {money(value)}"
            )

        return (
            "Revenue by City:\n"
            + "\n".join(result)
        )

    # --------------------------------------------------------
    # MONTHLY REVENUE
    # --------------------------------------------------------

    if (
        "monthly revenue" in q
        or "revenue by month" in q
    ):

        result = []

        for month, value in (
            verified["Monthly Revenue"].items()
        ):

            result.append(
                f"{month}: {money(value)}"
            )

        return (
            "Monthly Revenue:\n"
            + "\n".join(result)
        )

    # --------------------------------------------------------
    # TOP PRODUCTS
    # --------------------------------------------------------

    if (
        "top products" in q
        or "best products" in q
    ):

        result = []

        for product, value in list(
            verified["Revenue by Product"].items()
        )[:5]:

            result.append(
                f"{product}: {money(value)}"
            )

        return (
            "Top Products by Revenue:\n"
            + "\n".join(result)
        )

    return None


# ============================================================
# INVENTORY AGENT
# ============================================================

def inventory_agent(question):

    q = question.lower()

    # --------------------------------------------------------
    # INVENTORY RISK
    # --------------------------------------------------------

    if (
        "inventory risk" in q
        or "inventory risks" in q
        or "risk analysis" in q
        or "show inventory risk" in q
    ):

        return inventory_risk_agent(question)

    # --------------------------------------------------------
    # REORDER
    # --------------------------------------------------------

    if (
        "reorder" in q
        or "need reorder" in q
        or "needs reorder" in q
        or "which products need" in q
    ):

        return reorder_agent()

    # --------------------------------------------------------
    # HIGH DEMAND + LOW STOCK
    # --------------------------------------------------------

    if (
        (
            "high demand" in q
            or "high-demand" in q
        )
        and (
            "low stock" in q
            or "low-stock" in q
        )
    ):

        return high_demand_low_stock_agent()

    # --------------------------------------------------------
    # LOW STOCK
    # --------------------------------------------------------

    if (
        "low stock" in q
        or "low inventory" in q
        or "stock alert" in q
    ):

        verified = get_verified_data()

        stock = verified["Stock by Model"]

        result = []

        for model, value in stock.head(10).items():

            result.append(
                f"{model}: "
                f"{value:,.2f} average stock units"
            )

        return (
            "Lowest Average Stock Models:\n"
            + "\n".join(result)
        )

    return None


# ============================================================
# FORECASTING AGENT
# ============================================================

def prediction_agent():

    if df is None:
        return "Dataset is not loaded."

    data = df.copy()

    required_columns = ["Date", "Product_Model", "Units_Sold"]
    missing = [column for column in required_columns if column not in data.columns]

    if missing:
        return (
            "Forecasting cannot be performed.\n"
            f"Missing columns: {', '.join(missing)}"
        )

    data = data.dropna(subset=required_columns)

    if data.empty:
        return "Not enough historical data for forecasting."

    monthly_demand = (
        data.groupby([data["Date"].dt.to_period("M"), "Product_Model"])["Units_Sold"]
        .sum().reset_index()
    )
    monthly_demand.columns = ["Month", "Product_Model", "Demand"]

    forecasts = []

    for model_name in monthly_demand["Product_Model"].unique():
        model_data = monthly_demand[monthly_demand["Product_Model"] == model_name].sort_values("Month")
        if len(model_data) < 2:
            continue

        x = np.arange(len(model_data))
        y = model_data["Demand"].values.astype(float)

        try:
            coefficients = np.polyfit(x, y, 1)
            trend_model = np.poly1d(coefficients)
            forecast = max(0, float(trend_model(len(model_data))))
            average_monthly_demand = float(model_data["Demand"].sum()) / len(model_data)
            last_month_demand = float(model_data["Demand"].iloc[-1])

            forecasts.append({
                "Product_Model": model_name,
                "Last_Month_Demand": last_month_demand,
                "Average_Monthly_Demand": average_monthly_demand,
                "Forecast_Demand": forecast
            })
        except Exception:
            continue

    if not forecasts:
        return "Not enough historical monthly data for model-wise forecasting."

    forecast_df = pd.DataFrame(forecasts).sort_values("Forecast_Demand", ascending=False)

    result = [
        "Next Month Demand Forecast",
        "===========================",
        ""
    ]

    for _, row in forecast_df.iterrows():
        result.append(
            f"{row['Product_Model']}: "
            f"Forecast = {row['Forecast_Demand']:.0f} units | "
            f"Last Month = {row['Last_Month_Demand']:.0f} units | "
            f"Avg Monthly = {row['Average_Monthly_Demand']:.0f} units"
        )

    top = forecast_df.iloc[0]
    result.extend([
        "",
        "Highest Forecasted Demand",
        "-------------------------",
        f"Model: {top['Product_Model']}",
        f"Forecast Demand: {top['Forecast_Demand']:.0f} units",
        "",
        "Note: Forecasts are based on historical monthly demand using a simple linear trend model. They are estimates, not guaranteed future values."
    ])

    return "\n".join(result)

def inventory_planning_agent():
    """
    Inventory Planning:
    Combines forecasted demand with average stock to estimate
    stock coverage and suggest inventory actions.
    """

    df = load_dataset()

    required_columns = [
        "Product_Model",
        "Units_Sold",
        "Current_Stock",
        "Date"
    ]

    for column in required_columns:
        if column not in df.columns:
            return f"Inventory planning cannot run because '{column}' column is missing."

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove invalid dates
    df = df.dropna(subset=["Date"])

    if df.empty:
        return "Inventory planning cannot run because no valid dates were found."

    planning_rows = []

    # Analyze each product model
    for model, group in df.groupby("Product_Model"):

        group = group.sort_values("Date").copy()

        # -----------------------------------------
        # 1. Monthly demand
        # -----------------------------------------
        monthly_demand = (
            group.set_index("Date")
            .resample("ME")["Units_Sold"]
            .sum()
        )

        monthly_demand = monthly_demand[monthly_demand > 0]

        if monthly_demand.empty:
            continue

        # -----------------------------------------
        # 2. Average monthly demand
        # -----------------------------------------
        average_monthly_demand = monthly_demand.mean()

        # -----------------------------------------
        # 3. Forecast next month demand
        # -----------------------------------------
        if len(monthly_demand) >= 2:

            x = np.arange(len(monthly_demand))
            y = monthly_demand.values

            slope, intercept = np.polyfit(x, y, 1)

            next_month_index = len(monthly_demand)

            forecast = slope * next_month_index + intercept

        else:
            forecast = average_monthly_demand

        forecast = max(0, round(float(forecast)))

        # -----------------------------------------
        # 4. Average stock
        # -----------------------------------------
        average_stock = group["Current_Stock"].mean()

        # -----------------------------------------
        # 5. Estimated stock coverage
        # -----------------------------------------
        if average_monthly_demand > 0:
            stock_coverage = average_stock / average_monthly_demand
        else:
            stock_coverage = 0

        # -----------------------------------------
        # 6. Planning action
        # -----------------------------------------
        if forecast > average_stock:
            action = "REORDER - HIGH PRIORITY"

        elif stock_coverage < 1:
            action = "REORDER - URGENT"

        elif stock_coverage < 2:
            action = "MONITOR STOCK"

        else:
            action = "HEALTHY STOCK"

        planning_rows.append({
            "Product_Model": model,
            "Forecast_Demand": forecast,
            "Average_Monthly_Demand": round(average_monthly_demand, 2),
            "Average_Stock": round(average_stock, 2),
            "Estimated_Months_of_Stock": round(stock_coverage, 2),
            "Action": action
        })

    if not planning_rows:
        return "No inventory planning data available."

    planning = pd.DataFrame(planning_rows)

    # -----------------------------------------
    # Sort by priority
    # -----------------------------------------
    priority_order = {
        "REORDER - HIGH PRIORITY": 1,
        "REORDER - URGENT": 2,
        "MONITOR STOCK": 3,
        "HEALTHY STOCK": 4
    }

    planning["Priority"] = planning["Action"].map(priority_order)

    planning = planning.sort_values(
        ["Priority", "Estimated_Months_of_Stock"]
    )

    # -----------------------------------------
    # Display results
    # -----------------------------------------
    output = []

    output.append("")
    output.append("Inventory Planning")
    output.append("==================")
    output.append("")

    for _, row in planning.iterrows():

        output.append(
            f"{row['Product_Model']}: "
            f"Forecast = {row['Forecast_Demand']} units | "
            f"Avg Monthly Demand = {row['Average_Monthly_Demand']} units | "
            f"Avg Stock = {row['Average_Stock']} units | "
            f"Stock Coverage = {row['Estimated_Months_of_Stock']} months | "
            f"Action = {row['Action']}"
        )

    # -----------------------------------------
    # Important summary
    # -----------------------------------------
    reorder_models = planning[
        planning["Action"].isin([
            "REORDER - HIGH PRIORITY",
            "REORDER - URGENT"
        ])
    ]

    output.append("")
    output.append("Inventory Planning Summary")
    output.append("--------------------------")

    if not reorder_models.empty:

        output.append(
            f"Models requiring reorder attention: "
            f"{len(reorder_models)}"
        )

        output.append("")

        for _, row in reorder_models.iterrows():

            output.append(
                f"- {row['Product_Model']} "
                f"({row['Action']})"
            )

    else:

        output.append(
            "No models currently require reorder attention."
        )

    output.append("")
    output.append(
        "Note: Stock coverage is estimated using average stock "
        "and historical average monthly demand. These are planning "
        "estimates, not exact warehouse quantities or guaranteed "
        "purchase quantities."
    )

    return "\n".join(output)

# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def insight_agent():

    verified = get_verified_data()

    product, product_value = first_item(
        verified["Revenue by Product"]
    )

    model, model_demand = first_item(
        verified["Demand by Model"]
    )

    region, region_value = first_item(
        verified["Revenue by Region"]
    )

    city, city_value = first_item(
        verified["Revenue by City"]
    )

    channel, channel_value = first_item(
        verified["Revenue by Sales Channel"]
    )

    customer, customer_value = first_item(
        verified["Revenue by Customer Type"]
    )

    payment, payment_value = first_item(
        verified["Revenue by Payment Method"]
    )

    low_stock_model, low_stock_value = first_item(
        verified["Stock by Model"]
    )

    return (
        "Business Insights\n"
        "=================\n\n"

        f"1. Top Revenue Product\n"
        f"   {product} generated "
        f"{money(product_value)}.\n\n"

        f"2. Highest Demand Model\n"
        f"   {model} sold "
        f"{number(model_demand)} units.\n\n"

        f"3. Top Region\n"
        f"   {region} generated "
        f"{money(region_value)}.\n\n"

        f"4. Top City\n"
        f"   {city} generated "
        f"{money(city_value)}.\n\n"

        f"5. Best Sales Channel\n"
        f"   {channel} generated "
        f"{money(channel_value)}.\n\n"

        f"6. Highest Revenue Customer Type\n"
        f"   {customer} generated "
        f"{money(customer_value)}.\n\n"

        f"7. Leading Payment Method\n"
        f"   {payment} generated "
        f"{money(payment_value)}.\n\n"

        f"8. Lowest Stock Model\n"
        f"   {low_stock_model} has average stock "
        f"of {low_stock_value:,.2f} units."
    )


# ============================================================
# RECOMMENDATION AGENT
# ============================================================

def recommendation_agent():

    verified = get_verified_data()

    product, product_value = first_item(
        verified["Revenue by Product"]
    )

    model, demand_value = first_item(
        verified["Demand by Model"]
    )

    region, region_value = first_item(
        verified["Revenue by Region"]
    )

    city, city_value = first_item(
        verified["Revenue by City"]
    )

    channel, channel_value = first_item(
        verified["Revenue by Sales Channel"]
    )

    inventory = get_inventory_risk_data()

    high_risk = inventory[
        inventory["Risk_Level"] == "HIGH"
    ]

    result = []

    result.append(
        "Business Recommendations"
    )

    result.append(
        "========================="
    )

    result.append(
        f"\n1. Focus more on {product}"
        f"\n   It generated the highest product "
        f"revenue of {money(product_value)}."
    )

    result.append(
        f"\n2. Monitor {model}"
        f"\n   It has the highest recorded demand "
        f"with {number(demand_value)} units sold."
    )

    result.append(
        f"\n3. Strengthen operations in {region}"
        f"\n   It generated the highest regional "
        f"revenue of {money(region_value)}."
    )

    result.append(
        f"\n4. Analyze {city} performance"
        f"\n   It is the highest-revenue city with "
        f"{money(city_value)}."
    )

    result.append(
        f"\n5. Optimize the {channel} channel"
        f"\n   It contributes the highest channel "
        f"revenue of {money(channel_value)}."
    )

    # --------------------------------------------------------
    # Inventory recommendations
    # --------------------------------------------------------

    if not high_risk.empty:

        result.append(
            "\n6. Inventory Reorder Priority"
        )

        for model_name, row in high_risk.head(5).iterrows():

            result.append(
                f"\n   {model_name}"
                f"\n   Demand: "
                f"{number(row['Demand'])} units"
                f"\n   Average Stock: "
                f"{row['Average_Stock']:.2f} units"
                f"\n   Recommendation: "
                f"REORDER SOON"
            )

    else:

        result.append(
            "\n6. Inventory"
            "\n   No high-risk models detected."
        )

    result.append(
        "\n7. Decision Support"
        "\n   Combine revenue, demand, stock and "
        "forecasting information before making "
        "inventory decisions."
    )

    return "\n".join(result)


# ============================================================
# VISUALIZATION AGENT
# ============================================================

def visualization_agent():

    verified = get_verified_data()

    print("\nTop 5 Products by Revenue")
    print("-------------------------")

    for product, revenue in list(
        verified["Revenue by Product"].items()
    )[:5]:

        print(
            f"{product:<20} "
            f"{money(revenue)}"
        )

    print("\nTop 5 Models by Demand")
    print("----------------------")

    for model, units in list(
        verified["Demand by Model"].items()
    )[:5]:

        print(
            f"{model:<30} "
            f"{number(units)} units"
        )

    print("\nRevenue by Region")
    print("-----------------")

    for region, revenue in (
        verified["Revenue by Region"].items()
    ):

        print(
            f"{region:<10} "
            f"{money(revenue)}"
        )

    return "Visualization data displayed above."


# ============================================================
# OLLAMA AI EXPLANATION
# ============================================================

def llama_explanation(question):

    if not OLLAMA_AVAILABLE:
        return None

    verified = get_verified_data()

    product, product_value = first_item(
        verified["Revenue by Product"]
    )

    model, model_demand = first_item(
        verified["Demand by Model"]
    )

    region, region_value = first_item(
        verified["Revenue by Region"]
    )

    city, city_value = first_item(
        verified["Revenue by City"]
    )

    channel, channel_value = first_item(
        verified["Revenue by Sales Channel"]
    )

    inventory = get_inventory_risk_data()

    facts = f"""
VERIFIED BUSINESS FACTS FROM PANDAS:

Total Revenue:
{money(verified["Total Revenue"])}

Total Units Sold:
{number(verified["Total Units"])}

Highest Revenue Product:
{product} - {money(product_value)}

Highest Demand Model:
{model} - {number(model_demand)} units

Highest Revenue Region:
{region} - {money(region_value)}

Highest Revenue City:
{city} - {money(city_value)}

Highest Revenue Sales Channel:
{channel} - {money(channel_value)}

Average Discount:
{verified["Average Discount"]:.2f}%

Average Customer Age:
{verified["Average Customer Age"]:.1f} years

Highest Revenue Customer Type:
{first_item(verified["Revenue by Customer Type"])[0]}

Highest Revenue Payment Method:
{first_item(verified["Revenue by Payment Method"])[0]}
"""

    # Add inventory facts
    if not inventory.empty:

        high_risk = inventory[
            inventory["Risk_Level"] == "HIGH"
        ]

        facts += "\n\nINVENTORY RISK FACTS:\n"

        for model_name, row in high_risk.head(10).iterrows():

            facts += (
                f"{model_name}: "
                f"Demand {number(row['Demand'])} units, "
                f"Average Stock "
                f"{row['Average_Stock']:.2f}, "
                f"Risk {row['Risk_Level']}\n"
            )

    prompt = f"""
You are an AI business analyst.

Answer the manager's question using ONLY
the verified facts below.

Do NOT invent numbers.
Do NOT change any numbers.
Do NOT claim the dataset is real market data.
If the facts do not answer the question,
say that the available data is insufficient.

Manager Question:
{question}

{facts}

Give a short, clear business-oriented answer.
"""

    try:

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception:
        return None


# ============================================================
# MANAGER AGENT
# ============================================================

def manager_agent(question):

    print(
        "\n[Manager Agent → Processing Request]"
    )

    q = question.lower().strip()

    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    if (
        "business insight" in q
        or "business insights" in q
        or "give insights" in q
        or "show insights" in q
    ):

        print(
            "\n[Manager Agent → Insight Agent]"
        )

        return insight_agent()

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    if (
        "recommendation" in q
        or "recommendations" in q
        or "what should we do" in q
        or "what do you recommend" in q
    ):

        print(
            "\n[Manager Agent → Recommendation Agent]"
        )

        return recommendation_agent()

    # --------------------------------------------------------
    # INVENTORY RISK
    # --------------------------------------------------------

    if (
        "inventory risk" in q
        or "inventory risks" in q
        or "risk analysis" in q
        or "show inventory risk" in q
    ):

        print(
            "\n[Manager Agent → Inventory Risk Agent]"
        )

        return inventory_risk_agent(question)

    # --------------------------------------------------------
    # REORDER
    # --------------------------------------------------------

    if (
        "reorder" in q
        or "need reorder" in q
        or "needs reorder" in q
    ):

        print(
            "\n[Manager Agent → Reorder Agent]"
        )

        return reorder_agent()

    # --------------------------------------------------------
    # HIGH DEMAND + LOW STOCK
    # --------------------------------------------------------

    if (
        (
            "high demand" in q
            or "high-demand" in q
        )
        and (
            "low stock" in q
            or "low-stock" in q
        )
    ):

        print(
            "\n[Manager Agent → Inventory Agent]"
        )

        return high_demand_low_stock_agent()

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    if (
        "forecast" in q
        or "prediction" in q
        or "predict" in q
        or "next month" in q
    ):

        print(
            "\n[Manager Agent → Prediction Agent]"
        )

        return prediction_agent()


        # --------------------------------------------------------
    # INVENTORY PLANNING
    # --------------------------------------------------------

    if (
        "inventory planning" in q
        or "inventory plan" in q
        or "stock planning" in q
        or "stock coverage" in q
        or "reorder planning" in q
    ):

        print(
            "\n[Manager Agent → Inventory Planning Agent]"
        )

        return inventory_planning_agent()

    
    # --------------------------------------------------------
    # INVENTORY / STOCK
    # --------------------------------------------------------

    if (
        "low stock" in q
        or "low inventory" in q
        or "stock alert" in q
        or "inventory" in q
    ):

        print(
            "\n[Manager Agent → Inventory Agent]"
        )

        result = inventory_agent(question)

        if result:
            return result

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    if (
        "visualize" in q
        or "visualization" in q
        or "chart" in q
        or "top 5" in q
    ):

        print(
            "\n[Manager Agent → Visualization Agent]"
        )

        return visualization_agent()

    # --------------------------------------------------------
    # DATA ANALYST
    # --------------------------------------------------------

    print(
        "\n[Manager Agent → Data Analyst Agent]"
    )

    result = data_analyst_agent(question)

    if result:
        return result

    # --------------------------------------------------------
    # LOCAL LLAMA FALLBACK
    # --------------------------------------------------------

    print(
        "\n[Manager Agent → Local Llama 3.2]"
    )

    ai_result = llama_explanation(question)

    if ai_result:
        return ai_result

    return (
        "I could not identify that question.\n\n"
        "Try questions such as:\n"
        "- What is the total revenue?\n"
        "- Which product has the highest revenue?\n"
        "- Which model has the highest demand?\n"
        "- Which model has the lowest stock?\n"
        "- Show inventory risk\n"
        "- Which products need reorder?\n"
        "- Show high demand low stock models\n"
        "- Which region has the highest revenue?\n"
        "- Which city has the highest revenue?\n"
        "- Which sales channel has the highest revenue?\n"
        "- Give me business insights\n"
        "- Give me recommendations\n"
        "- Forecast next month revenue"
    )


# ============================================================
# TERMINAL APPLICATION
# ============================================================

def main():

    print("\n" + "=" * 65)

    print(
        "       AI DATA ANALYST AGENT"
    )

    print(
        " Electronics Sales, Demand & Inventory Intelligence"
    )

    print("=" * 65)

    loaded = load_dataset(
        DEFAULT_DATASET
    )

    if loaded is None:
        return

    if not validate_dataset():

        print(
            "\nDataset schema does not match "
            "the expected structure."
        )

        return

    print("\nAvailable Analysis:")
    print("-------------------")

    print("1. Products")
    print("2. Product Models")
    print("3. Regions")
    print("4. Cities")
    print("5. Categories")
    print("6. Customer Types")
    print("7. Sales Channels")
    print("8. Payment Methods")
    print("9. Salespersons")
    print("10. Monthly Revenue")
    print("11. Discounts")
    print("12. Order Status")
    print("13. Inventory")
    print("14. Inventory Risk")
    print("15. Reorder Recommendations")
    print("16. Forecasting")
    print("17. Business Insights")
    print("18. Recommendations")

    print("\nExample Questions:")
    print("------------------")

    print(
        "Which product has the highest revenue?"
    )

    print(
        "Which model has the highest demand?"
    )

    print(
        "Which model has the lowest stock?"
    )

    print(
        "Show inventory risk"
    )

    print(
        "Which products need reorder?"
    )

    print(
        "Show high demand low stock models"
    )

    print(
        "Which region has the highest revenue?"
    )

    print(
        "Which city has the highest revenue?"
    )

    print(
        "Which sales channel has the highest revenue?"
    )

    print(
        "Which customer type has the highest revenue?"
    )

    print(
        "Which payment method has the highest revenue?"
    )

    print(
        "What is the average discount?"
    )

    print(
        "Give me business insights"
    )

    print(
        "Give me recommendations"
    )

    print(
        "Forecast next month demand by model"
    )

    print("\nType 'exit' to stop.")

    while True:

        try:

            question = input(
                "\nAsk your business question: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nProgram stopped."
            )

            break

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nAI Data Analyst Agent stopped."
            )

            break

        if not question:
            continue

        try:

            answer = manager_agent(
                question
            )

            print(
                "\n" + "-" * 65
            )

            print(answer)

            print(
                "-" * 65
            )

        except Exception as e:

            print(
                f"\nError while processing request: {e}"
            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()