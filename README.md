# 📈 Sales Analytics Dashboard

A production-ready sales analytics dashboard built with **Python + Streamlit + Plotly**.
Dark-themed, interactive, and deployable to Streamlit Cloud in minutes.

---

## Screenshots / Features

| Feature | Details |
|---|---|
| 📤 Upload CSV | Drop your own data or use built-in sample data (3,000 rows) |
| 🔢 5 KPI cards | Revenue · Orders · AOV · Customers · Units |
| 📈 Revenue trend | Weekly area chart with hover |
| 🥧 Category donut | Revenue split with centre total |
| 🏆 Top products | Horizontal bar chart, colour-scaled |
| 📅 MoM growth | Green/red bar chart |
| 🗺️ US region map | Bubble map sized by revenue |
| 🔥 Heatmap | Day-of-week × Month revenue |
| ⬇️ Export | Download filtered data as CSV |

---

## File structure

```
sales_dashboard/
├── app.py                   ← Streamlit entry point
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml          ← Dark theme + server config
└── utils/
    ├── __init__.py
    ├── data_loader.py       ← CSV loader + sample data generator
    ├── kpi.py               ← KPI calculations with delta %
    └── charts.py            ← 7 Plotly chart builders
```

---

## CSV format (for your own data)

| Column      | Type   | Example     |
|-------------|--------|-------------|
| Date        | date   | 2024-03-15  |
| Revenue     | float  | 249.99      |
| Region      | string | North       |
| Category    | string | Electronics |
| Product     | string | Laptop Pro  |
| Units       | int    | 3           |
| CustomerID  | string | C1234       |

---

## Run locally

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
# → Open http://localhost:8501
```

---

## Deploy to Streamlit Cloud (free)

1. Push this folder to a **GitHub repo** (can be private)
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app
3. Set **Main file path** → `app.py`
4. Click **Deploy** 🚀

No secrets or environment variables required.

---

## Extend it

| Idea | Where |
|---|---|
| Add forecasting | New `utils/forecast.py` with `statsforecast` |
| Connect to a DB | Replace `generate_sample_data()` with SQLAlchemy |
| Add auth | `streamlit-authenticator` package |
| Email alerts | New `utils/alerts.py` with `smtplib` |
| More charts | Add functions to `utils/charts.py` |
