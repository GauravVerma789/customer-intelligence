import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

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
# SIDEBAR
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

        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])

    # =================================================
    # PREDICTION BUTTON
    # =================================================
    if st.button("Predict Churn"):

        base_row = df.iloc[0].copy()

        # ---------- numeric ----------
        base_row["tenure"] = tenure
        base_row["MonthlyCharges"] = monthly
        base_row["TotalServices"] = services
        base_row["EngagementScore"] = engagement
        base_row["TotalCharges"] = monthly * max(tenure, 1)
        base_row["AvgMonthlyValue"] = monthly
        base_row["HighValue"] = high_value

        # ---------- CONTRACT ----------
        base_row["Contract_Month-to-month"] = 0
        base_row["Contract_One year"] = 0
        base_row["Contract_Two year"] = 0

        if contract_risk == 2:
            base_row["Contract_Month-to-month"] = 1
        elif contract_risk == 1:
            base_row["Contract_One year"] = 1
        else:
            base_row["Contract_Two year"] = 1

        # ---------- PAYMENT ----------
        base_row["PaymentMethod_Electronic check"] = 0
        base_row["PaymentMethod_Credit card (automatic)"] = 0
        base_row["PaymentMethod_Bank transfer (automatic)"] = 0
        base_row["PaymentMethod_Mailed check"] = 0

        if autopay == 1:
            base_row["PaymentMethod_Electronic check"] = 1
        else:
            base_row["PaymentMethod_Credit card (automatic)"] = 1

        # ---------- INTERNET ----------
        base_row["InternetService_DSL"] = 0
        base_row["InternetService_Fiber optic"] = 0
        base_row["InternetService_No"] = 0

        if internet_service == "Fiber optic":
            base_row["InternetService_Fiber optic"] = 1
        elif internet_service == "DSL":
            base_row["InternetService_DSL"] = 1
        else:
            base_row["InternetService_No"] = 1

        # ---------- TECH SUPPORT ----------
        base_row["TechSupport_Yes"] = 0
        base_row["TechSupport_No"] = 0

        if tech_support == "Yes":
            base_row["TechSupport_Yes"] = 1
        else:
            base_row["TechSupport_No"] = 1

        # ---------- ONLINE SECURITY ----------
        base_row["OnlineSecurity_Yes"] = 0
        base_row["OnlineSecurity_No"] = 0

        if online_security == "Yes":
            base_row["OnlineSecurity_Yes"] = 1
        else:
            base_row["OnlineSecurity_No"] = 1

        # ---------- STREAMING ----------
        base_row["StreamingTV_Yes"] = 0
        base_row["StreamingTV_No"] = 0

        if streaming_tv == "Yes":
            base_row["StreamingTV_Yes"] = 1
        else:
            base_row["StreamingTV_No"] = 1

        # ---------- CHURN PREDICTION ----------
        input_df = base_row.drop("Churn").to_frame().T
        input_df = input_df.reindex(
            columns=churn_model.feature_names_in_,
            fill_value=0
        )

        prob = churn_model.predict_proba(input_df)[0, 1]

        st.metric("Churn Probability", f"{prob:.2%}")

        if prob > 0.40:
            st.error("High Churn Risk")
        elif prob > 0.25:
            st.warning("Medium Risk")
        else:
            st.success("Low Churn Risk")

        # ---------- SEGMENT ----------
        seg_input = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": monthly * max(tenure, 1),
            "TotalServices": services,
            "EngagementScore": engagement,
            "AvgMonthlyValue": monthly,
            "ContractRisk": contract_risk,
            "AutoPay": autopay,
            "HighValue": high_value
        }])

        seg_scaled = scaler.transform(seg_input)
        segment_pred = kmeans.predict(seg_scaled)[0]

        st.info(f"Predicted Customer Segment: {segment_pred}")