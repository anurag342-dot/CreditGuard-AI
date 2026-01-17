import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="CreditGuard AI", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        color: #333333;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card h1 {
        margin: 0;
        font-size: 2.5rem;
        color: #333333;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        color: #666666;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. LOAD SAVED ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('xgb_credit_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        selector = joblib.load('feature_selector.pkl')
        return model, scaler, selector
    except FileNotFoundError:
        return None, None, None


model, scaler, selector = load_assets()

if model is None:
    st.error(
        "⚠️ Error: .pkl files not found! Make sure xgb_credit_model.pkl, robust_scaler.pkl, and feature_selector.pkl are in the same folder.")
    st.stop()

# --- 3. MAPPINGS ---
occupation_map = {
    'Accountant': 0, 'Architect': 1, 'Developer': 2, 'Doctor': 3,
    'Engineer': 4, 'Entrepreneur': 5, 'Journalist': 6, 'Lawyer': 7,
    'Manager': 8, 'Mechanic': 9, 'Media_Manager': 10, 'Musician': 11,
    'Scientist': 12, 'Teacher': 13, 'Writer': 14, '_______': 15
}

credit_mix_map = {'Bad': 0, 'Good': 1, 'Standard': 2, '_': 3}
min_pay_map = {'NM': 0, 'No': 1, 'Yes': 2}
pay_behaviour_map = {
    '!@9#%8': 0,
    'High_spent_Large_value_payments': 1,
    'High_spent_Medium_value_payments': 2,
    'High_spent_Small_value_payments': 3,
    'Low_spent_Large_value_payments': 4,
    'Low_spent_Medium_value_payments': 5,
    'Low_spent_Small_value_payments': 6
}
loan_map = {
    'Not Specified': 3464, 'Auto Loan': 60, 'Credit-Builder Loan': 684,
    'Debt Consolidation Loan': 1410, 'Home Equity Loan': 2100, 'Mortgage Loan': 2779,
    'Payday Loan': 4145, 'Personal Loan': 4879, 'Student Loan': 5592
}

# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("👤 Customer Profile")


def user_input_features():
    # --- Financials ---
    st.sidebar.subheader("💰 Financials")
    annual_income = st.sidebar.number_input("Annual Income", value=50000.0)
    monthly_inhand = st.sidebar.number_input("Monthly Inhand Salary", value=4000.0)
    outstanding_debt = st.sidebar.number_input("Outstanding Debt", value=1000.0)
    monthly_balance = st.sidebar.number_input("Avg. Monthly Balance", value=300.0)
    total_emi = st.sidebar.number_input("Total EMI per Month", value=50.0)
    amount_invested = st.sidebar.number_input("Amount Invested Monthly", value=80.0)

    # --- Accounts ---
    st.sidebar.subheader("💳 Accounts")
    num_bank_acc = st.sidebar.number_input("Num Bank Accounts", min_value=0, value=2)
    num_credit_card = st.sidebar.number_input("Num Credit Cards", min_value=0, value=3)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", value=15)
    num_loan = st.sidebar.number_input("Number of Loans", min_value=0, value=1)

    # --- History ---
    st.sidebar.subheader("📅 History")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    delay_due_date = st.sidebar.number_input("Days Delayed", min_value=0, value=5)
    num_delayed_payment = st.sidebar.number_input("Num Delayed Payments", min_value=0, value=2)
    changed_limit = st.sidebar.number_input("Changed Credit Limit", value=0.0)
    num_inquiries = st.sidebar.number_input("Num Credit Inquiries", min_value=0, value=1)
    credit_utilization = st.sidebar.number_input("Credit Utilization Ratio", value=30.0)
    credit_history_age = st.sidebar.number_input("Credit History Age (Months)", value=120.0)

    # --- Categorical Dropdowns ---
    st.sidebar.subheader("📂 Details")
    occ_label = st.sidebar.selectbox("Occupation", options=list(occupation_map.keys()))
    mix_label = st.sidebar.selectbox("Credit Mix", options=list(credit_mix_map.keys()), index=1)
    pay_min_label = st.sidebar.selectbox("Pays Minimum Amount?", options=list(min_pay_map.keys()))
    pay_beh_label = st.sidebar.selectbox("Payment Behaviour", options=list(pay_behaviour_map.keys()))
    loan_label = st.sidebar.selectbox("Type of Loan", options=list(loan_map.keys()))

    # --- Pack Data ---
    data = {
        'Age': age,
        'Occupation': occupation_map[occ_label],
        'Annual_Income': annual_income,
        'Monthly_Inhand_Salary': monthly_inhand,
        'Num_Bank_Accounts': num_bank_acc,
        'Num_Credit_Card': num_credit_card,
        'Interest_Rate': interest_rate,
        'Num_of_Loan': num_loan,
        'Type_of_Loan': loan_map[loan_label],
        'Delay_from_due_date': delay_due_date,
        'Num_of_Delayed_Payment': num_delayed_payment,
        'Changed_Credit_Limit': changed_limit,
        'Num_Credit_Inquiries': num_inquiries,
        'Credit_Mix': credit_mix_map[mix_label],
        'Outstanding_Debt': outstanding_debt,
        'Credit_Utilization_Ratio': credit_utilization,
        'Credit_History_Age': credit_history_age,
        'Payment_of_Min_Amount': min_pay_map[pay_min_label],
        'Total_EMI_per_month': total_emi,
        'Amount_invested_monthly': amount_invested,
        'Payment_Behaviour': pay_behaviour_map[pay_beh_label],
        'Monthly_Balance': monthly_balance
    }
    return pd.DataFrame(data, index=[0])


# --- 5. MAIN PAGE LOGIC ---
st.title("🏦 CreditGuard AI System")
st.markdown("### Intelligent Credit Scoring & Risk Assessment")

input_df = user_input_features()

col1, col2 = st.columns([2, 1])
with col1:
    st.info("System Ready. Adjust inputs in the sidebar and click Analyze.")
    with st.expander("View Raw Input Data"):
        st.dataframe(input_df)

if st.button("Analyze Credit Risk"):
    # 1. Scale
    input_scaled = scaler.transform(input_df)

    # 2. Select Features
    input_selected = selector.transform(input_scaled)

    # 3. Predict
    prediction = model.predict(input_selected)[0]
    prediction_proba = model.predict_proba(input_selected)[0]

    # --- 6. SAFETY RULES (Sanity Check) ---
    is_hard_fail = False
    fail_reason = ""

    # READ VALUES FROM DATAFRAME
    val_delay = input_df['Delay_from_due_date'].values[0]
    val_debt = input_df['Outstanding_Debt'].values[0]
    val_income = input_df['Annual_Income'].values[0]
    val_util = input_df['Credit_Utilization_Ratio'].values[0]

    # Rule 1: Extreme Delays
    if val_delay > 90:
        prediction = 0  # Force POOR
        prediction_proba = np.array([0.99, 0.01, 0.00])
        is_hard_fail = True
        fail_reason = "Extreme Payment Delays (>90 Days)"

    # Rule 2: Debt Trap
    elif (val_debt > val_income) and (val_util > 90):
        prediction = 0  # Force POOR
        prediction_proba = np.array([0.95, 0.05, 0.00])
        is_hard_fail = True
        fail_reason = "Severe Debt Trap (Debt > Income & Maxed Cards)"

    # --- 7. DISPLAY RESULTS ---
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    score_labels = {0: "POOR", 1: "STANDARD", 2: "GOOD"}
    score_colors = {0: "#FF4B4B", 1: "#FFA500", 2: "#4CAF50"}

    result_label = score_labels[prediction]
    result_color = score_colors[prediction]
    confidence = prediction_proba[prediction] * 100

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {result_color};">
            <h3>Credit Score</h3>
            <h1 style="color: {result_color};">{result_label}</h1>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Confidence</h3>
            <h1>{confidence:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model Engine</h3>
            <h1>{'Safety Rule Override' if is_hard_fail else 'XGBoost'}</h1>
        </div>
        """, unsafe_allow_html=True)

    if not is_hard_fail:
        st.write("### Probability Breakdown")
        chart_data = pd.DataFrame(
            prediction_proba.reshape(1, -1),
            columns=["Poor", "Standard", "Good"]
        )
        st.bar_chart(chart_data)
    else:
        st.error(f"⚠️ **Automatic Rejection:** {fail_reason}")

    # Custom Advice
    if prediction == 0:
        st.error("⚠️ **High Risk:** This customer shows strong signs of financial instability.")
    elif prediction == 1:
        st.warning("⚠️ **Caution:** Customer is stable but has some risk factors.")
    else:
        st.success("✅ **Approved:** Excellent credit profile.")