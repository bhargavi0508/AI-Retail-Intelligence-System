from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

from agent import (
    manager_agent,
    load_dataset,
    inventory_dashboard_data
)

# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="AI Retail Intelligence Dashboard",
    description="AI-Powered Electronics Sales, Demand & Inventory Intelligence System",
    version="1.0"
)


# ============================================
# GLOBAL DATASET
# ============================================

df = None


# ============================================
# QUESTION MODEL
# ============================================

class QuestionRequest(BaseModel):

    question: str


# ============================================
# HOME / DASHBOARD
# ============================================

@app.get("/", response_class=HTMLResponse)
def dashboard():

    html_file = Path("templates/index.html")

    if not html_file.exists():

        return """
        <h2>Dashboard file not found.</h2>
        <p>Please make sure templates/index.html exists.</p>
        """

    return html_file.read_text(
        encoding="utf-8"
    )


# ============================================
# AI QUESTION API
# ============================================

@app.post("/ask")
def ask_ai(
    request: QuestionRequest
):

    try:

        answer = manager_agent(
            request.question
        )

        return {
            "answer": answer
        }

    except Exception as e:

        return {
            "answer":
                f"Error while processing question: {str(e)}"
        }


# ============================================
# KPI API
# ============================================

@app.get("/kpis")
def get_kpis():

    global df

    # Load dataset if required

    if df is None or df.empty:

        df = load_dataset()

    # Dataset validation

    if df is None or df.empty:

        return {
            "error":
                "Dataset is not loaded."
        }

    # ----------------------------------------
    # Total Revenue
    # ----------------------------------------

    total_revenue = (
        df["Revenue"].sum()
    )

    # ----------------------------------------
    # Total Units Sold
    # ----------------------------------------

    total_units = (
        df["Units_Sold"].sum()
    )

    # ----------------------------------------
    # Top Product
    # ----------------------------------------

    top_product = (

        df.groupby("Product")["Revenue"]
        .sum()
        .idxmax()

    )

    # ----------------------------------------
    # Top Demand Model
    # ----------------------------------------

    top_model = (

        df.groupby("Product_Model")["Units_Sold"]
        .sum()
        .idxmax()

    )

    # ----------------------------------------
    # Top Region
    # ----------------------------------------

    top_region = (

        df.groupby("Region")["Revenue"]
        .sum()
        .idxmax()

    )

    return {

        "total_revenue":
            float(total_revenue),

        "total_units":
            int(total_units),

        "top_product":
            str(top_product),

        "top_model":
            str(top_model),

        "top_region":
            str(top_region)

    }


# ============================================
# BUSINESS ANALYTICS API
# ============================================

@app.get("/analytics")
def get_analytics():

    global df

    # Load dataset if required

    if df is None or df.empty:

        df = load_dataset()

    # Dataset validation

    if df is None or df.empty:

        return {
            "error":
                "Dataset is not loaded."
        }

    # ========================================
    # PRODUCT REVENUE
    # ========================================

    product_revenue = (

        df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(5)

    )

    # ========================================
    # MODEL DEMAND
    # ========================================

    model_demand = (

        df.groupby(
            "Product_Model"
        )["Units_Sold"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(5)

    )

    # ========================================
    # REGION REVENUE
    # ========================================

    region_revenue = (

        df.groupby("Region")["Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

    )

    # ========================================
    # CITY REVENUE
    # ========================================

    city_revenue = (

        df.groupby("City")["Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(5)

    )

    # ========================================
    # SALES CHANNEL REVENUE
    # ========================================

    channel_revenue = (

        df.groupby(
            "Sales_Channel"
        )["Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

    )

    # ========================================
    # CUSTOMER TYPE REVENUE
    # ========================================

    customer_revenue = (

        df.groupby(
            "Customer_Type"
        )["Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

    )

    # ========================================
    # PAYMENT METHOD REVENUE
    # ========================================

    payment_revenue = (

        df.groupby(
            "Payment_Method"
        )["Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

    )

    # ========================================
    # RETURN ANALYTICS
    # ========================================

    return {

        "product_revenue": [

            {
                "name": str(name),
                "value": float(value)
            }

            for name, value
            in product_revenue.items()

        ],

        "model_demand": [

            {
                "name": str(name),
                "value": int(value)
            }

            for name, value
            in model_demand.items()

        ],

        "region_revenue": [

            {
                "name": str(name),
                "value": float(value)
            }

            for name, value
            in region_revenue.items()

        ],

        "city_revenue": [

            {
                "name": str(name),
                "value": float(value)
            }

            for name, value
            in city_revenue.items()

        ],

        "channel_revenue": [

            {
                "name": str(name),
                "value": float(value)
            }

            for name, value
            in channel_revenue.items()

        ],

        "customer_revenue": [

            {
                "name": str(name),
                "value": float(value)
            }

            for name, value
            in customer_revenue.items()

        ],

        "payment_revenue": [

            {
                "name": str(name),
                "value": float(value)
            }

            for name, value
            in payment_revenue.items()

        ]

    }


# ============================================
# INVENTORY INTELLIGENCE API
# ============================================

@app.get("/inventory")
def get_inventory():

    try:

        inventory_data = inventory_dashboard_data()

        return inventory_data

    except Exception as e:

        return {
            "error":
                f"Inventory analysis error: {str(e)}"
        }

# ============================================
# RUN SERVER DIRECTLY
# ============================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )