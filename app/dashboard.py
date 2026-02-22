import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================================================
# PAGE CONFIG  (MUST BE FIRST STREAMLIT COMMAND)
# =====================================================
st.set_page_config(
    page_title="AI Customer Intelligence",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/telco_segmented.csv")

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    churn_model = joblib.load("models/churn_model.pkl")
    kmeans = joblib.load("models/kmeans_segmentation.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return churn_model, kmeans, scaler

df = load_data()
churn_model, kmeans, scaler = load_models()

# =====================================================
# TITLE
# =====================================================
st.title("AI Customer Intelligence Dashboard")
st.write("Customer Segmentation & Churn Prediction System")

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
page = st.sidebar.selectbox(
    "Select Module",
    ["Overview", "Segmentation Explorer", "Churn Prediction"]
)

# =====================================================
# OVERVIEW
# =====================================================
if page == "Overview":

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Distribution")
        st.bar_chart(df["Segment"].value_counts())

    with col2:
        st.subheader("Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    st.subheader("Key Statistics")
    st.write(df.describe())

# =====================================================
# SEGMENTATION
# =====================================================
elif page == "Segmentation Explorer":

    st.subheader("Customer Segments")

    segment = st.selectbox(
        "Select Segment",
        sorted(df["Segment"].unique())
    )

    seg_df = df[df["Segment"] == segment]

    st.write(f"Customers in Segment {segment}: {len(seg_df)}")

    st.dataframe(seg_df.head(20))

    st.subheader("Segment Statistics")
    st.write(seg_df.describe())

    st.subheader("Monthly Charges Distribution")
    st.bar_chart(seg_df["MonthlyCharges"])

# =====================================================
# CHURN PREDICTION
# =====================================================
elif page == "Churn Prediction":

    st.subheader("Predict Customer Churn")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        services = st.slider("Total Services", 0, 9, 3)
        engagement = st.slider("Engagement Score", 0.0, 20.0, 5.0)

    with col2:
        contract_risk = st.selectbox("Contract Risk", [0, 1, 2])
        autopay = st.selectbox("AutoPay (0=Yes,1=No)", [0, 1])
        high_value = st.selectbox("High Value Customer", [0, 1])

    if st.button("Predict Churn"):

        # Derived values
        total_charges = monthly * max(tenure, 1)
        avg_monthly = monthly

        input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
            "TotalServices": services,
            "EngagementScore": engagement,
            "AvgMonthlyValue": avg_monthly,
            "ContractRisk": contract_risk,
            "AutoPay": autopay,
            "HighValue": high_value
        }

        input_df = pd.DataFrame([input_dict])

        # Match churn model features
        input_df = input_df.reindex(
            columns=churn_model.feature_names_in_,
            fill_value=0
        )

        prob = churn_model.predict_proba(input_df)[0, 1]

        st.metric("Churn Probability", f"{prob:.2%}")

        if prob > 0.7:
            st.error("High Churn Risk")
        elif prob > 0.4:
            st.warning("Medium Risk")
        else:
            st.success("Low Churn Risk")

        # ===============================
        # SEGMENT PREDICTION
        # ===============================
        # take a template row from dataset
template = df.iloc[0].copy()

# update only user fields
template["tenure"] = tenure
template["MonthlyCharges"] = monthly
template["TotalServices"] = services
template["EngagementScore"] = engagement
template["ContractRisk"] = contract_risk
template["AutoPay"] = autopay
template["HighValue"] = high_value
template["TotalCharges"] = monthly * max(tenure,1)
template["AvgMonthlyValue"] = monthly

# drop target
input_df = template.drop("Churn").to_frame().T

# match model columns
input_df = input_df.reindex(columns=churn_model.feature_names_in_, fill_value=0)

prob = churn_model.predict_proba(input_df)[0,1]