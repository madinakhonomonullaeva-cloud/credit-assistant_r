"""
AI Credit Decision Assistant
Forecast-based repayment burden MVP for dissertation purposes.

What this app does:
1. Loads the prepared Uzbekistan macro-financial dataset from data_fw.xlsx.
2. Forecasts inflation, policy/refinancing rate, average salary and macro pressure over the loan term.
3. Combines forecasted macro context with user-entered financial data.
4. Simulates whether the loan may become burdensome month by month.

Important:
- This is NOT financial advice.
- This is NOT a credit score.
- This is NOT a bank decision system.
- It does not predict legal default.
- It estimates repayment burden / repayment pressure for responsible borrowing support.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from datetime import date
import math

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Credit Decision Assistant",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
.main > .block-container {
    max-width: 1120px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 55%, #60a5fa 100%);
    border-radius: 24px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -0.04em;
}
.hero p {
    margin: 0;
    opacity: 0.92;
    font-size: 0.98rem;
}
.soft-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 4px 18px rgba(15,23,42,0.05);
}
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 16px 18px;
    color: #1e3a8a;
}
.disclaimer {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 0.86rem;
    color: #475569;
}
.low-pill, .medium-pill, .high-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 0.9rem;
}
.low-pill { background: #ecfdf5; color: #047857; border: 1px solid #6ee7b7; }
.medium-pill { background: #fffbeb; color: #b45309; border: 1px solid #fcd34d; }
.high-pill { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = APP_DIR / "data_fw.xlsx"
MAIN_SHEET = "Final_Monthly_ML_Dataset"

PRESSURE_TO_NUM = {"Low": 0, "Medium": 1, "High": 2}
NUM_TO_PRESSURE = {0: "Low", 1: "Medium", 2: "High"}

FEATURE_COLUMNS = [
    "policy_rate_pct",
    "inflation_yoy_pct",
    "cpi_mom_pct",
    "nominal_wage_monthly_approx",
    "real_wage_growth_pct",
    "real_policy_rate_pct",
    "debt_burden_indicator_pct",
]

REQUIRED_COLUMNS = [
    "date",
    "policy_rate_pct",
    "inflation_yoy_pct",
    "cpi_mom_pct",
    "nominal_wage_monthly_approx",
    "real_wage_growth_pct",
    "real_policy_rate_pct",
    "debt_burden_indicator_pct",
    "repayment_pressure_level",
]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class UserInput:
    loan_amount: float
    annual_interest_rate_pct: float
    loan_term_months: int
    monthly_income: float
    essential_expenses: float
    existing_loan_payments: float
    current_balance: float
    salary_day: int
    payment_due_day: int
    salary_growth_mode: str
    expected_annual_salary_growth_pct: float
    expenses_grow_with_inflation: bool


# ============================================================
# CORE HELPERS
# ============================================================


def fmt_uzs(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:,.0f}".replace(",", " ") + " UZS"


def fmt_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}%"


def annuity_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    """Standard annuity payment formula."""
    if principal <= 0 or months <= 0:
        return 0.0
    monthly_rate = annual_rate_pct / 100 / 12
    if monthly_rate == 0:
        return principal / months
    return principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)


def pressure_badge(level: str) -> str:
    css = {"Low": "low-pill", "Medium": "medium-pill", "High": "high-pill"}.get(level, "medium-pill")
    return f'<span class="{css}">{level}</span>'


def user_burden_badge(level: str) -> str:
    return pressure_badge(level)


# ============================================================
# LOAD MACRO DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_macro_dataset(uploaded_file=None) -> pd.DataFrame:
    """Load prepared macro dataset from Excel file or uploaded file."""
    if uploaded_file is not None:
        source = uploaded_file
    elif DEFAULT_DATA_FILE.exists():
        source = DEFAULT_DATA_FILE
    else:
        raise FileNotFoundError(
            "data_fw.xlsx was not found. Upload it in the sidebar or place it next to app.py."
        )

    xls = pd.ExcelFile(source)
    sheet = MAIN_SHEET if MAIN_SHEET in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(source, sheet_name=sheet)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"The macro dataset is missing required columns: {missing}")

    df = df.dropna(how="all").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["repayment_pressure_level"] = df["repayment_pressure_level"].astype(str).str.strip()
    df = df[df["repayment_pressure_level"].isin(["Low", "Medium", "High"])].copy()
    df["pressure_numeric"] = df["repayment_pressure_level"].map(PRESSURE_TO_NUM)

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    return df


# ============================================================
# MACRO PRESSURE MODEL — EXPLAINABLE DECISION TREE RULES
# ============================================================


def classify_macro_pressure(row: pd.Series | dict) -> str:
    """
    Explainable macro pressure classification based on the Decision Tree rules
    produced in the Colab analysis.

    Rules:
    - inflation_yoy_pct > 11.95 => High
    - inflation_yoy_pct <= 11.95 and real_wage_growth_pct > 10.28 => Low
    - inflation_yoy_pct <= 11.95 and real_wage_growth_pct <= 10.28 and real_policy_rate_pct <= -0.35 => High
    - inflation_yoy_pct <= 6.62 => Low
    - otherwise => Medium
    """
    inflation = float(row.get("inflation_yoy_pct", 0))
    real_wage_growth = float(row.get("real_wage_growth_pct", 0))
    real_policy_rate = float(row.get("real_policy_rate_pct", 0))

    if inflation > 11.95:
        return "High"
    if real_wage_growth > 10.28:
        return "Low"
    if real_policy_rate <= -0.35:
        return "High"
    if inflation <= 6.62:
        return "Low"
    return "Medium"


# ============================================================
# FORECASTING HELPERS
# ============================================================


def damped_linear_forecast(series: pd.Series, horizon: int, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    """
    Simple robust forecasting without external packages.
    Uses last 36 observations, estimates a damped linear trend, and clips values.
    This is model-based forecasting, not an official economic forecast.
    """
    clean = pd.Series(series).astype(float).dropna().reset_index(drop=True)
    if clean.empty:
        return np.zeros(horizon)

    window = clean.tail(min(36, len(clean))).reset_index(drop=True)
    last_value = float(window.iloc[-1])

    if len(window) >= 6:
        x = np.arange(len(window))
        try:
            slope, intercept = np.polyfit(x, window.values, 1)
        except Exception:
            slope = 0.0
    else:
        slope = 0.0

    # Dampen trend to avoid unrealistic long-term extrapolation.
    slope = slope * 0.35
    forecast = np.array([last_value + slope * (i + 1) for i in range(horizon)], dtype=float)

    if lower is not None:
        forecast = np.maximum(forecast, lower)
    if upper is not None:
        forecast = np.minimum(forecast, upper)

    return forecast


def forecast_average_salary(series: pd.Series, horizon: int) -> np.ndarray:
    """Forecast nominal average salary using median recent monthly growth with clipping."""
    clean = pd.Series(series).astype(float).dropna().reset_index(drop=True)
    if clean.empty:
        return np.ones(horizon)

    recent_growth = clean.pct_change().tail(24).replace([np.inf, -np.inf], np.nan).dropna()
    monthly_growth = float(recent_growth.median()) if not recent_growth.empty else 0.005
    monthly_growth = float(np.clip(monthly_growth, 0.001, 0.018))

    values = []
    current = float(clean.iloc[-1])
    for _ in range(horizon):
        current = current * (1 + monthly_growth)
        values.append(current)
    return np.array(values)


def build_macro_forecast(df: pd.DataFrame, start_date: pd.Timestamp, horizon_months: int) -> pd.DataFrame:
    """Create macro forecast for the selected loan horizon."""
    last_date = df["date"].max()
    # Forecast starts from the month after historical data ends.
    forecast_start = last_date + pd.DateOffset(months=1)

    # If user start date is later than forecast_start, we still forecast from forecast_start and then filter.
    end_date = start_date + pd.DateOffset(months=horizon_months - 1)
    full_end = max(end_date, forecast_start)
    future_dates = pd.date_range(start=forecast_start, end=full_end, freq="MS")
    full_horizon = len(future_dates)

    future = pd.DataFrame({"date": future_dates})

    future["policy_rate_pct"] = damped_linear_forecast(df["policy_rate_pct"], full_horizon, lower=0, upper=40)
    future["inflation_yoy_pct"] = damped_linear_forecast(df["inflation_yoy_pct"], full_horizon, lower=0, upper=50)
    future["cpi_mom_pct"] = damped_linear_forecast(df["cpi_mom_pct"], full_horizon, lower=-5, upper=10)
    future["nominal_wage_monthly_approx"] = forecast_average_salary(df["nominal_wage_monthly_approx"], full_horizon)
    future["debt_burden_indicator_pct"] = damped_linear_forecast(
        df["debt_burden_indicator_pct"], full_horizon, lower=0, upper=100
    )

    # Combine historical and forecasted data to calculate YoY salary growth.
    historical = df[["date", "policy_rate_pct", "inflation_yoy_pct", "cpi_mom_pct", "nominal_wage_monthly_approx", "debt_burden_indicator_pct"]].copy()
    combined = pd.concat([historical, future], ignore_index=True).sort_values("date")
    combined["nominal_wage_yoy_growth_pct"] = combined["nominal_wage_monthly_approx"].pct_change(12) * 100
    combined["real_wage_growth_pct"] = combined["nominal_wage_yoy_growth_pct"] - combined["inflation_yoy_pct"]
    combined["real_policy_rate_pct"] = combined["policy_rate_pct"] - combined["inflation_yoy_pct"]

    last_real_wage = float(df["real_wage_growth_pct"].dropna().iloc[-1])
    combined["real_wage_growth_pct"] = combined["real_wage_growth_pct"].fillna(last_real_wage)

    future_full = combined[combined["date"].isin(future_dates)].copy()
    future_full["year"] = future_full["date"].dt.year
    future_full["month"] = future_full["date"].dt.month
    future_full["predicted_macro_pressure"] = future_full.apply(classify_macro_pressure, axis=1)
    future_full["macro_pressure_numeric"] = future_full["predicted_macro_pressure"].map(PRESSURE_TO_NUM)

    # Select only the user's loan term.
    loan_forecast = future_full[(future_full["date"] >= start_date) & (future_full["date"] <= end_date)].copy()

    # If the user chose a start date before the forecast period, use first available future month.
    if loan_forecast.empty:
        loan_forecast = future_full.head(horizon_months).copy()
    else:
        loan_forecast = loan_forecast.head(horizon_months).copy()

    loan_forecast["loan_month"] = np.arange(1, len(loan_forecast) + 1)
    return loan_forecast.reset_index(drop=True)


# ============================================================
# USER BURDEN SIMULATION
# ============================================================


def simulate_user_loan_burden(user: UserInput, macro_forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate user repayment burden for every month of the loan."""
    monthly_payment = annuity_payment(user.loan_amount, user.annual_interest_rate_pct, user.loan_term_months)

    rows = []
    projected_income = float(user.monthly_income)
    projected_expenses = float(user.essential_expenses)

    fixed_monthly_salary_growth = (1 + user.expected_annual_salary_growth_pct / 100) ** (1 / 12) - 1
    previous_avg_salary = float(macro_forecast["nominal_wage_monthly_approx"].iloc[0])

    for i, row in macro_forecast.iterrows():
        loan_month = int(row["loan_month"])

        if loan_month == 1:
            projected_income = float(user.monthly_income)
            projected_expenses = float(user.essential_expenses)
        else:
            if user.salary_growth_mode == "In line with average salary forecast":
                current_avg_salary = float(row["nominal_wage_monthly_approx"])
                avg_growth = (current_avg_salary / previous_avg_salary) - 1 if previous_avg_salary > 0 else 0
                avg_growth = float(np.clip(avg_growth, -0.02, 0.03))
                projected_income *= (1 + avg_growth)
                previous_avg_salary = current_avg_salary
            elif user.salary_growth_mode == "Fixed annual growth assumption":
                projected_income *= (1 + fixed_monthly_salary_growth)
            else:
                projected_income = projected_income

            if user.expenses_grow_with_inflation:
                monthly_cpi = float(row["cpi_mom_pct"]) / 100
                monthly_cpi = float(np.clip(monthly_cpi, -0.03, 0.05))
                projected_expenses *= (1 + monthly_cpi)

        payment_to_income_ratio = monthly_payment / projected_income if projected_income > 0 else np.nan
        total_debt_burden_ratio = (monthly_payment + user.existing_loan_payments) / projected_income if projected_income > 0 else np.nan
        free_cash_after_payment = projected_income + user.current_balance - projected_expenses - user.existing_loan_payments - monthly_payment
        monthly_cash_flow_after_payment = projected_income - projected_expenses - user.existing_loan_payments - monthly_payment
        income_vs_average_salary = projected_income / float(row["nominal_wage_monthly_approx"])
        payment_before_salary = user.payment_due_day < user.salary_day
        macro_pressure = row["predicted_macro_pressure"]

        score = 0
        reasons = []

        if payment_to_income_ratio >= 0.40:
            score += 2
            reasons.append("Monthly payment is 40% or more of projected income")
        elif payment_to_income_ratio >= 0.25:
            score += 1
            reasons.append("Monthly payment is 25–40% of projected income")

        if total_debt_burden_ratio >= 0.50:
            score += 2
            reasons.append("Total debt burden is 50% or more of projected income")
        elif total_debt_burden_ratio >= 0.35:
            score += 1
            reasons.append("Total debt burden is 35–50% of projected income")

        if monthly_cash_flow_after_payment < 0:
            score += 2
            reasons.append("Monthly cash flow after expenses and loan payment is negative")
        elif monthly_cash_flow_after_payment < projected_income * 0.10:
            score += 1
            reasons.append("Monthly cash buffer is below 10% of projected income")

        if income_vs_average_salary < 0.80:
            score += 1
            reasons.append("Projected income is below 80% of forecasted average salary")

        if payment_before_salary:
            score += 1
            reasons.append("Payment date comes before salary date")

        if macro_pressure == "High":
            score += 2
            reasons.append("Forecasted macro repayment pressure is High")
        elif macro_pressure == "Medium":
            score += 1
            reasons.append("Forecasted macro repayment pressure is Medium")

        if float(row["inflation_yoy_pct"]) >= 12:
            score += 1
            reasons.append("Forecasted inflation is high")

        if score <= 2:
            user_burden = "Low"
        elif score <= 4:
            user_burden = "Medium"
        else:
            user_burden = "High"

        rows.append(
            {
                "date": row["date"],
                "loan_month": loan_month,
                "monthly_payment": monthly_payment,
                "projected_user_income": projected_income,
                "projected_essential_expenses": projected_expenses,
                "existing_loan_payments": user.existing_loan_payments,
                "payment_to_income_pct": payment_to_income_ratio * 100,
                "total_debt_burden_pct": total_debt_burden_ratio * 100,
                "monthly_cash_flow_after_payment": monthly_cash_flow_after_payment,
                "free_cash_after_payment_including_current_balance": free_cash_after_payment,
                "income_vs_average_salary": income_vs_average_salary,
                "payment_before_salary": payment_before_salary,
                "forecast_inflation_yoy_pct": row["inflation_yoy_pct"],
                "forecast_policy_rate_pct": row["policy_rate_pct"],
                "forecast_average_salary": row["nominal_wage_monthly_approx"],
                "forecast_real_wage_growth_pct": row["real_wage_growth_pct"],
                "forecast_macro_pressure": macro_pressure,
                "user_pressure_score": score,
                "predicted_user_repayment_burden": user_burden,
                "reasons": "; ".join(reasons),
            }
        )

    sim = pd.DataFrame(rows)
    sim["user_burden_numeric"] = sim["predicted_user_repayment_burden"].map(PRESSURE_TO_NUM)

    distribution = (
        sim["predicted_user_repayment_burden"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    distribution.columns = ["burden_level", "months_count"]
    distribution["share_pct"] = distribution["months_count"] / len(sim) * 100

    high_months = int(distribution.loc[distribution["burden_level"] == "High", "months_count"].iloc[0])
    medium_months = int(distribution.loc[distribution["burden_level"] == "Medium", "months_count"].iloc[0])

    first_high = sim[sim["predicted_user_repayment_burden"] == "High"].head(1)
    first_high_month = int(first_high["loan_month"].iloc[0]) if not first_high.empty else None
    first_high_date = str(first_high["date"].dt.date.iloc[0]) if not first_high.empty else None

    if high_months / len(sim) >= 0.30:
        result = "High long-term repayment burden"
    elif high_months > 0 or medium_months / len(sim) >= 0.50:
        result = "Medium long-term repayment burden"
    else:
        result = "Low long-term repayment burden"

    summary = pd.DataFrame(
        {
            "metric": [
                "loan_amount",
                "loan_term_months",
                "annual_interest_rate_pct",
                "monthly_payment",
                "total_repayable",
                "total_interest_cost",
                "initial_user_income",
                "final_projected_user_income",
                "initial_essential_expenses",
                "final_projected_essential_expenses",
                "low_burden_months",
                "medium_burden_months",
                "high_burden_months",
                "first_high_burden_month",
                "first_high_burden_date",
                "avg_payment_to_income_pct",
                "max_payment_to_income_pct",
                "avg_total_debt_burden_pct",
                "max_total_debt_burden_pct",
                "min_monthly_cash_flow_after_payment",
                "long_term_result",
            ],
            "value": [
                user.loan_amount,
                user.loan_term_months,
                user.annual_interest_rate_pct,
                monthly_payment,
                monthly_payment * user.loan_term_months,
                monthly_payment * user.loan_term_months - user.loan_amount,
                user.monthly_income,
                sim["projected_user_income"].iloc[-1],
                user.essential_expenses,
                sim["projected_essential_expenses"].iloc[-1],
                int(distribution.loc[distribution["burden_level"] == "Low", "months_count"].iloc[0]),
                medium_months,
                high_months,
                first_high_month,
                first_high_date,
                sim["payment_to_income_pct"].mean(),
                sim["payment_to_income_pct"].max(),
                sim["total_debt_burden_pct"].mean(),
                sim["total_debt_burden_pct"].max(),
                sim["monthly_cash_flow_after_payment"].min(),
                result,
            ],
        }
    )

    return sim, distribution, summary


def build_recommendation(summary: pd.DataFrame) -> str:
    result = summary.loc[summary["metric"] == "long_term_result", "value"].iloc[0]
    high_months = int(summary.loc[summary["metric"] == "high_burden_months", "value"].iloc[0])
    min_cash = float(summary.loc[summary["metric"] == "min_monthly_cash_flow_after_payment", "value"].iloc[0])

    if "High" in result:
        return (
            f"The loan may become highly burdensome during the selected term. "
            f"High-pressure months: {high_months}. Minimum projected monthly cash flow after payment: {fmt_uzs(min_cash)}. "
            "Consider reducing the loan amount, lowering the monthly payment, changing the payment date to after salary, "
            "or postponing the loan until income becomes more stable."
        )
    if "Medium" in result:
        return (
            f"The loan appears possible but may create pressure in some months. "
            f"Minimum projected monthly cash flow after payment: {fmt_uzs(min_cash)}. "
            "Keep a cash buffer, avoid taking additional debt, and check the payment date against salary timing."
        )
    return (
        "The loan appears relatively manageable under the forecasted assumptions. "
        "Still monitor inflation, income stability, expenses, and payment timing."
    )


# ============================================================
# UI
# ============================================================

st.markdown(
    """
<div class="hero">
  <h1>💳 AI Credit Decision Assistant</h1>
  <p>Forecast-based repayment burden analysis for responsible borrowing in Uzbekistan.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="disclaimer">
<b>Research disclaimer.</b> This MVP is created for dissertation demonstration. It is not financial advice, not a credit score, not a bank decision, and does not approve or reject credit. It estimates repayment burden using user-entered data and forecasted macro-financial context.
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar data upload
st.sidebar.header("Macro dataset")
uploaded_macro = st.sidebar.file_uploader(
    "Optional: upload data_fw.xlsx",
    type=["xlsx"],
    help="If data_fw.xlsx is already in the GitHub repository next to app.py, you do not need to upload it.",
)

try:
    macro_df = load_macro_dataset(uploaded_macro)
except Exception as e:
    st.error(f"Could not load macro dataset: {e}")
    st.stop()

latest = macro_df.sort_values("date").iloc[-1]

with st.sidebar:
    st.success("Macro dataset loaded")
    st.write(f"Period: {macro_df['date'].min().date()} — {macro_df['date'].max().date()}")
    st.write(f"Rows: {len(macro_df)}")
    st.markdown("Latest macro context:")
    st.write(f"Inflation: {fmt_pct(latest['inflation_yoy_pct'])}")
    st.write(f"Policy rate: {fmt_pct(latest['policy_rate_pct'])}")
    st.write(f"Average salary: {fmt_uzs(latest['nominal_wage_monthly_approx'])}")
    st.write(f"Pressure: {latest['repayment_pressure_level']}")


# Input form
st.subheader("1. Enter loan and personal finance data")

with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amount = st.number_input("Loan amount (UZS)", min_value=0.0, value=50_000_000.0, step=500_000.0)
    with col2:
        annual_interest = st.number_input("Annual loan interest rate (%)", min_value=0.0, max_value=150.0, value=28.0, step=0.5)
    with col3:
        loan_term_months = st.number_input("Loan term (months)", min_value=1, max_value=120, value=60, step=1)

    col4, col5, col6 = st.columns(3)
    with col4:
        monthly_income = st.number_input("Current monthly income (UZS)", min_value=1.0, value=5_000_000.0, step=100_000.0)
    with col5:
        essential_expenses = st.number_input("Essential monthly expenses (UZS)", min_value=0.0, value=2_000_000.0, step=100_000.0)
    with col6:
        existing_payments = st.number_input("Existing loan payments / month (UZS)", min_value=0.0, value=500_000.0, step=100_000.0)

    col7, col8, col9 = st.columns(3)
    with col7:
        current_balance = st.number_input("Current balance / savings (UZS)", min_value=0.0, value=800_000.0, step=100_000.0)
    with col8:
        salary_day = st.slider("Salary day", 1, 31, 15)
    with col9:
        payment_due_day = st.slider("Payment due day", 1, 31, 10)

    col10, col11, col12 = st.columns(3)
    with col10:
        salary_growth_mode = st.selectbox(
            "Salary projection mode",
            ["In line with average salary forecast", "Fixed annual growth assumption", "No salary growth"],
            index=0,
        )
    with col11:
        expected_salary_growth = st.number_input("Expected annual salary growth (%)", min_value=-50.0, max_value=100.0, value=10.0, step=0.5)
    with col12:
        expenses_grow = st.checkbox("Expenses grow with inflation", value=True)

    submitted = st.form_submit_button("Analyse 5-year repayment burden", use_container_width=True)


if not submitted:
    st.markdown(
        """
<div class="info-box">
<b>How it works:</b> the app forecasts macroeconomic context over the loan term, including inflation, policy/refinancing rate and average salary. Then it combines this forecast with your personal inputs to estimate whether the loan may become burdensome month by month.
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# Build user object
user = UserInput(
    loan_amount=loan_amount,
    annual_interest_rate_pct=annual_interest,
    loan_term_months=int(loan_term_months),
    monthly_income=monthly_income,
    essential_expenses=essential_expenses,
    existing_loan_payments=existing_payments,
    current_balance=current_balance,
    salary_day=int(salary_day),
    payment_due_day=int(payment_due_day),
    salary_growth_mode=salary_growth_mode,
    expected_annual_salary_growth_pct=expected_salary_growth,
    expenses_grow_with_inflation=expenses_grow,
)

loan_start = pd.Timestamp(date.today().replace(day=1))
macro_forecast = build_macro_forecast(macro_df, loan_start, user.loan_term_months)
loan_simulation, burden_distribution, long_term_summary = simulate_user_loan_burden(user, macro_forecast)
recommendation = build_recommendation(long_term_summary)

# Extract key summary metrics
summary_dict = dict(zip(long_term_summary["metric"], long_term_summary["value"]))
long_term_result = summary_dict["long_term_result"]
monthly_payment = float(summary_dict["monthly_payment"])
total_repayable = float(summary_dict["total_repayable"])
total_interest = float(summary_dict["total_interest_cost"])
high_months = int(summary_dict["high_burden_months"])
medium_months = int(summary_dict["medium_burden_months"])
min_cash = float(summary_dict["min_monthly_cash_flow_after_payment"])
avg_pti = float(summary_dict["avg_payment_to_income_pct"])
max_debt = float(summary_dict["max_total_debt_burden_pct"])

if "High" in long_term_result:
    result_level = "High"
elif "Medium" in long_term_result:
    result_level = "Medium"
else:
    result_level = "Low"

st.markdown("---")
st.subheader("2. Long-term forecast result")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Monthly payment", fmt_uzs(monthly_payment))
c2.metric("Total repayable", fmt_uzs(total_repayable))
c3.metric("Total interest cost", fmt_uzs(total_interest))
c4.markdown(f"**Long-term result**<br>{user_burden_badge(result_level)}", unsafe_allow_html=True)

c5, c6, c7, c8 = st.columns(4)
c5.metric("High-burden months", high_months)
c6.metric("Medium-burden months", medium_months)
c7.metric("Avg payment-to-income", fmt_pct(avg_pti))
c8.metric("Min monthly cash flow", fmt_uzs(min_cash))

st.markdown(
    f"""
<div class="soft-card">
<b>Recommendation:</b><br>{recommendation}
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("3. Why this result?")
reason_counts = loan_simulation["reasons"].str.get_dummies(sep="; ").sum().sort_values(ascending=False)
if not reason_counts.empty:
    for reason, count in reason_counts.head(6).items():
        st.write(f"- {reason} — {int(count)} months")
else:
    st.write("No high-risk reasons were detected.")

st.subheader("4. Forecasted macro context during the loan term")

macro_chart_df = macro_forecast.set_index("date")[[
    "inflation_yoy_pct",
    "policy_rate_pct",
    "real_policy_rate_pct",
    "real_wage_growth_pct",
]]
st.line_chart(macro_chart_df)

salary_chart = macro_forecast.set_index("date")[["nominal_wage_monthly_approx"]]
st.line_chart(salary_chart)

pressure_chart = macro_forecast.set_index("date")[["macro_pressure_numeric"]]
st.line_chart(pressure_chart)
st.caption("Macro pressure numeric scale: 0 = Low, 1 = Medium, 2 = High.")

st.subheader("5. User repayment burden forecast")

burden_chart_df = loan_simulation.set_index("date")[[
    "payment_to_income_pct",
    "total_debt_burden_pct",
]]
st.line_chart(burden_chart_df)

cash_chart_df = loan_simulation.set_index("date")[[
    "monthly_cash_flow_after_payment",
]]
st.line_chart(cash_chart_df)

user_burden_chart = loan_simulation.set_index("date")[["user_burden_numeric"]]
st.line_chart(user_burden_chart)
st.caption("User burden numeric scale: 0 = Low, 1 = Medium, 2 = High.")

st.subheader("6. Burden distribution over the loan term")
st.dataframe(burden_distribution, use_container_width=True, hide_index=True)

st.subheader("7. Monthly simulation table")
show_cols = [
    "date",
    "loan_month",
    "monthly_payment",
    "projected_user_income",
    "projected_essential_expenses",
    "payment_to_income_pct",
    "total_debt_burden_pct",
    "monthly_cash_flow_after_payment",
    "forecast_inflation_yoy_pct",
    "forecast_policy_rate_pct",
    "forecast_average_salary",
    "forecast_macro_pressure",
    "predicted_user_repayment_burden",
    "reasons",
]
st.dataframe(loan_simulation[show_cols], use_container_width=True, hide_index=True)

# Downloads
st.subheader("8. Download results")

csv_sim = loan_simulation.to_csv(index=False).encode("utf-8")
csv_macro = macro_forecast.to_csv(index=False).encode("utf-8")
csv_summary = long_term_summary.to_csv(index=False).encode("utf-8")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.download_button("Download monthly simulation CSV", csv_sim, "user_loan_burden_simulation.csv", "text/csv")
with col_b:
    st.download_button("Download macro forecast CSV", csv_macro, "macro_forecast_loan_term.csv", "text/csv")
with col_c:
    st.download_button("Download summary CSV", csv_summary, "long_term_summary.csv", "text/csv")

with st.expander("Methodology notes"):
    st.markdown(
        """
- The app uses historical monthly macro-financial data for Uzbekistan.
- Forecasts are model-based extrapolations and are not official economic forecasts.
- Macro pressure is classified using transparent Decision Tree rules derived from the dissertation analysis.
- User burden is calculated monthly using payment-to-income ratio, total debt burden, cash flow, income compared with forecasted average salary, payment timing, and forecasted macro pressure.
- The output estimates repayment pressure, not legal default.
- The app does not approve or reject credit.
"""
    )

st.markdown(
    """
<div class="disclaimer">
<b>Important.</b> This tool is an exploratory research MVP. The forecast is based on historical data and simplified assumptions. It should be used for dissertation demonstration and responsible borrowing education only.
</div>
""",
    unsafe_allow_html=True,
)
