# AI Customer Intelligence Platform

## Objective
Predict customer churn and segment customers to improve retention strategies.

## Dataset
Telco Customer Churn Dataset

## Key EDA Insights

1. Customers with short tenure show highest churn risk.
2. Month-to-month contract users churn significantly more.
3. High monthly charge customers are more likely to churn.
4. Fiber optic service users show elevated churn.
5. Long-term contract customers show strong retention.

## Engineered Features

TenureGroup → lifecycle stage  
TotalServices → service usage intensity  
AvgMonthlyValue → customer revenue density  
ContractRisk → churn probability proxy  
AutoPay → payment stability  
EngagementScore → behavioral loyalty score  
HighValue → premium indicator  
AtRisk → churn-prone segment

Segment 0:
- High tenure
- High charges
- High engagement
→ Loyal Premium Customers

Segment 1:
- Low tenure
- High contract risk
→ New At-Risk Customers

Segment 2:
- Medium value
- Moderate engagement
→ Standard Customers

Segment 3:
- Low services
- Low value
→ Low Engagement Customers

# AI Customer Intelligence Platform

## Overview
End-to-end machine learning system for customer segmentation and churn prediction.

## Features
- Customer segmentation using KMeans
- Churn prediction using Random Forest
- Explainable AI (SHAP)
- Interactive Streamlit dashboard
- Business insights & recommendations

## Tech Stack
Python, Scikit-learn, Pandas, Streamlit, SHAP

## Live Demo
[Streamlit App Link]

## Results
- Churn ROC-AUC: 0.85+
- Segmentation clusters: 4 customer personas
- Key churn drivers: tenure, contract type, engagement

## Author
Gaurav Verma