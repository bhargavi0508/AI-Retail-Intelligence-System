from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import os
import shutil
import uvicorn

from agent import manager_agent, load_dataset, get_verified_data


app = FastAPI(title="AI Retail Intelligence")


UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_dataset_path = "data/electronics_sales_data.csv"


# ============================================================
# DASHBOARD HTML
# ============================================================

def dashboard_page(message="", answer=""):

    dashboard_html = ""

    try:
        load_dataset(current_dataset_path)
        verified = get_verified_data()

        total_revenue = verified["Total Revenue"]
        total_units = verified["Total Units"]

        top_product = list(verified["Revenue by Product"])[0]
        top_region = list(verified["Revenue by Region"])[0]
        top_city = list(verified["Revenue by City"])[0]
        top_channel = list(verified["Revenue by Sales Channel"])[0]

        dashboard_html = f"""
        <div class="cards">

            <div class="card">
                <div class="icon">💰</div>
                <h3>Total Revenue</h3>
                <p>₹{total_revenue:,.2f}</p>
            </div>

            <div class="card">
                <div class="icon">📦</div>
                <h3>Total Units Sold</h3>
                <p>{total_units:,}</p>
            </div>

            <div class="card">
                <div class="icon">🏆</div>
                <h3>Top Product</h3>
                <p>{top_product[0]}</p>
            </div>

            <div class="card">
                <div class="icon">📍</div>
                <h3>Top Region</h3>
                <p>{top_region[0]}</p>
            </div>

            <div class="card">
                <div class="icon">🏙️</div>
                <h3>Top City</h3>
                <p>{top_city[0]}</p>
            </div>

            <div class="card">
                <div class="icon">🛒</div>
                <h3>Top Sales Channel</h3>
                <p>{top_channel[0]}</p>
            </div>

        </div>
        """

    except Exception as e:
        dashboard_html = f"""
        <div class="error">
            Dashboard data unavailable: {str(e)}
        </div>
        """


    answer_html = ""

    if answer:
        answer_html = f"""
        <div class="answer-box">
            <div class="answer-title">🤖 AI Assistant</div>
            <div class="answer-content">
                {answer}
            </div>
        </div>
        """


    return f"""
<!DOCTYPE html>

<html>

<head>

<title>AI Retail Intelligence</title>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f7fb;
    color: #1f2937;
}}

.navbar {{
    background: #111827;
    color: white;
    padding: 18px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.logo {{
    font-size: 23px;
    font-weight: bold;
}}

.logo span {{
    color: #60a5fa;
}}

.nav-text {{
    color: #cbd5e1;
    font-size: 14px;
}}

.container {{
    width: 92%;
    max-width: 1250px;
    margin: 35px auto;
}}

.hero {{
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: white;
    padding: 38px;
    border-radius: 18px;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.18);
}}

.hero h1 {{
    margin: 0 0 10px 0;
    font-size: 34px;
}}

.hero p {{
    margin: 0;
    color: #e0e7ff;
    font-size: 16px;
}}

.section-title {{
    margin-top: 30px;
    margin-bottom: 15px;
    font-size: 22px;
}}

.cards {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}}

.card {{
    background: white;
    padding: 22px;
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
}}

.card:hover {{
    transform: translateY(-3px);
    transition: 0.2s;
}}

.icon {{
    font-size: 28px;
}}

.card h3 {{
    margin: 12px 0 7px;
    font-size: 14px;
    color: #6b7280;
}}

.card p {{
    margin: 0;
    font-size: 21px;
    font-weight: bold;
    color: #111827;
}}

.panel {{
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}}

input[type="file"] {{
    margin: 12px 0;
}}

input[type="text"] {{
    width: 100%;
    padding: 14px;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    font-size: 15px;
    margin-bottom: 12px;
}}

button {{
    border: none;
    padding: 12px 22px;
    border-radius: 9px;
    cursor: pointer;
    font-size: 14px;
    font-weight: bold;
}}

.upload-btn {{
    background: #2563eb;
    color: white;
}}

.ask-btn {{
    background: #111827;
    color: white;
}}

button:hover {{
    opacity: 0.9;
}}

.answer-box {{
    background: #eef6ff;
    border-left: 5px solid #2563eb;
    padding: 20px;
    margin-top: 20px;
    border-radius: 10px;
}}

.answer-title {{
    font-weight: bold;
    margin-bottom: 10px;
    font-size: 17px;
}}

.answer-content {{
    white-space: pre-line;
    line-height: 1.6;
}}

.error {{
    background: #fee2e2;
    color: #991b1b;
    padding: 15px;
    border-radius: 10px;
}}

.footer {{
    text-align: center;
    padding: 30px;
    color: #6b7280;
    font-size: 13px;
}}

@media (max-width: 900px) {{

    .cards {{
        grid-template-columns: repeat(2, 1fr);
    }}

}}

@media (max-width: 600px) {{

    .navbar {{
        padding: 15px 20px;
    }}

    .container {{
        width: 94%;
    }}

    .cards {{
        grid-template-columns: 1fr;
    }}

    .hero h1 {{
        font-size: 26px;
    }}

}}

</style>

</head>


<body>


<div class="navbar">

    <div class="logo">
        🤖 AI <span>Retail Intelligence</span>
    </div>

    <div class="nav-text">
        AI-Powered Business Analytics
    </div>

</div>


<div class="container">


<div class="hero">

    <h1>AI Retail Intelligence Dashboard</h1>

    <p>
        Analyze sales, products, customers, regions and business
        performance using your AI Data Analyst Agent.
    </p>

</div>


<h2 class="section-title">
    📊 Business Overview
</h2>

{dashboard_html}


<h2 class="section-title">
    📁 Upload Dataset
</h2>

<div class="panel">

    <p>
        Upload your CSV or Excel sales dataset.
    </p>

    <form action="/upload" method="post" enctype="multipart/form-data">

        <input
            type="file"
            name="file"
            accept=".csv,.xlsx,.xls"
            required
        >

        <br>

        <button class="upload-btn" type="submit">
            Upload Dataset
        </button>

    </form>

</div>


<h2 class="section-title">
    🤖 Ask Your Data
</h2>

<div class="panel">

    <p>
        Ask questions about your business data in natural language.
    </p>

    <form action="/ask" method="post">

        <input
            type="text"
            name="question"
            placeholder="Example: Which product has the highest revenue?"
            required
        >

        <button class="ask-btn" type="submit">
            Ask AI
        </button>

    </form>

    {answer_html}

</div>


<h2 class="section-title">
    🚀 AI Capabilities
</h2>

<div class="cards">

    <div class="card">
        <div class="icon">📈</div>
        <h3>Revenue Forecasting</h3>
        <p>Predict future revenue</p>
    </div>

    <div class="card">
        <div class="icon">📦</div>
        <h3>Inventory Intelligence</h3>
        <p>Identify stock risks</p>
    </div>

    <div class="card">
        <div class="icon">💡</div>
        <h3>Business Insights</h3>
        <p>Discover important patterns</p>
    </div>

    <div class="card">
        <div class="icon">🎯</div>
        <h3>Recommendations</h3>
        <p>Get data-driven actions</p>
    </div>

    <div class="card">
        <div class="icon">📊</div>
        <h3>Data Visualization</h3>
        <p>Explore charts and trends</p>
    </div>

    <div class="card">
        <div class="icon">🧠</div>
        <h3>AI Data Analyst</h3>
        <p>Ask questions naturally</p>
    </div>

</div>


</div>


<div class="footer">

    AI Retail Intelligence • Built with Python, FastAPI, Pandas & Ollama

</div>


</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return dashboard_page()


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.post("/upload", response_class=HTMLResponse)
async def upload_dataset(file: UploadFile = File(...)):

    global current_dataset_path

    filename = file.filename

    if not filename.lower().endswith((".csv", ".xlsx", ".xls")):

        return dashboard_page(
            message="Invalid file type."
        )

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    try:

        # Validate uploaded dataset

        if filename.lower().endswith(".csv"):

            df = pd.read_csv(file_path)

        else:

            df = pd.read_excel(file_path)

        if df.empty:

            return dashboard_page(
                message="Uploaded dataset is empty."
            )

        load_dataset(file_path)

        current_dataset_path = file_path

        return dashboard_page(
            message=f"Dataset uploaded successfully: {filename}"
        )

    except Exception as e:

        return dashboard_page(
            message=f"Error loading dataset: {str(e)}"
        )


# ============================================================
# ASK AI
# ============================================================

@app.post("/ask", response_class=HTMLResponse)
async def ask_ai(question: str = Form(...)):

    global current_dataset_path

    try:

        load_dataset(current_dataset_path)

        answer = manager_agent(question)

        return dashboard_page(
            answer=answer
        )

    except Exception as e:

        return dashboard_page(
            answer=f"Error: {str(e)}"
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )