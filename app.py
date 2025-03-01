import streamlit as st
import pandas as pd
import joblib

# Load models and encoder
lgb_model = joblib.load("lightgbm_fraud_detection.pkl")
xgb_model = joblib.load("xgboost_fraud_detection.pkl")
encoder = joblib.load("label_encoder.pkl")

# Hardcoded sample data with additional columns
sample_data = pd.DataFrame({
    "step": [1, 1, 1, 1, 1],
    "type": ["PAYMENT", "PAYMENT", "TRANSFER", "CASH_OUT", "PAYMENT"],
    "amount": [9839.64, 1864.28, 181.00, 181.00, 11668.14],
    "nameOrig": ["C1231006815", "C1666544295", "C1305486145", "C840083671", "C2048537720"],
    "oldbalanceOrg": [170136.0, 21249.0, 181.0, 181.0, 41554.0],
    "newbalanceOrig": [160296.36, 19384.72, 0.00, 0.00, 29885.86],
    "nameDest": ["M1979787155", "M2044282225", "C553264065", "C38997010", "M1230701703"],
    "oldbalanceDest": [0.0, 0.0, 0.0, 21182.0, 0.0],
    "newbalanceDest": [0.0, 0.0, 0.0, 0.0, 0.0],
    "isFraud": [0, 0, 1, 1, 0],
    "isFlaggedFraud": [0, 0, 0, 0, 0]
})

st.title("Fraud Detection System")

# Model selection
model_choice = st.radio("Select a Model:", ["LightGBM", "XGBoost"])
model = lgb_model if model_choice == "LightGBM" else xgb_model

# Data input method
input_choice = st.radio("How would you like to enter data?", ["Use Sample Data", "Enter Manually"])

if input_choice == "Use Sample Data":
    # Let user select a sample row and display full data
    selected_index = st.selectbox("Select a transaction:", sample_data.index)
    selected_row = sample_data.iloc[selected_index:selected_index + 1]
    st.write("### Selected Transaction")
    st.table(selected_row)
    
    # Show expected fraud result from sample data
    expected = selected_row["isFraud"].values[0]
    st.write(f"**Expected Fraud Result:** {'Fraudulent' if expected == 1 else 'Non-Fraudulent'}")
    
    # Prepare input data for prediction by dropping only the extra columns (keep isFlaggedFraud)
    input_data = selected_row.drop(columns=["nameOrig", "nameDest", "isFraud"])
else:
    # Manual entry fields
    step = st.number_input("Step", min_value=1, value=1)
    # Create a list of transaction types with their encoded values
    type_options = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    # Display the dropdown
    type_option = st.selectbox("Transaction Type", type_options)
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=500.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=400.0)
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)
    # Set isFlaggedFraud to a default value (e.g., 0) for manual entry
    isFlaggedFraud = st.selectbox("isFlaggedFraud", [0, 1])
    
    # Create input dataframe with 8 features
    input_data = pd.DataFrame([[step, type_option, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud]],
                              columns=["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"])

# Encode transaction type using the saved encoder
input_data["type"] = encoder.transform(input_data["type"])

# Predict and display results
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    fraud_probability = model.predict_proba(input_data)[0][1]
    
    st.write("## Prediction Result")
    st.write(f"Fraud Probability: {fraud_probability:.2%}")
    st.write("Fraudulent Transaction" if prediction == 1 else "Non-Fraudulent Transaction")
