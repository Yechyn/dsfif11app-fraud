

api_url = "http://127.0.0.1:8503"

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

st.title("Fraud Detection App")

# Display site header
st.header("Upload Transactions CSV for Batch Predictions")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

image_path = "../images/dsif header 2.jpeg"
try:
    img = Image.open(image_path)
    st.image(img, use_column_width=True)
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

transaction_amount = st.number_input("Transaction Amount")
customer_age = st.number_input("Customer Age")
customer_balance = st.number_input("Customer Balance")

data = {
    "transaction_amount": transaction_amount,
    "customer_age": customer_age,
    "customer_balance": customer_balance
}

if st.button("Show Feature Importance"):
    response = requests.get(f"{api_url}/feature-importance")
    feature_importance = response.json().get('feature_importance', {})

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

if st.button("Predict and show prediction confidence"):
    response = requests.post(f"{api_url}/predict/", json=data)
    result = response.json()
    confidence = result['confidence']

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    labels = ['Not Fraudulent', 'Fraudulent']
    fig, ax = plt.subplots()
    ax.bar(labels, confidence, color=['green', 'red'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

if st.button("Predict and show SHAP values"):
    response = requests.post(f"{api_url}/predict/", json=data)
    result = response.json()

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    shap_values = np.array(result['shap_values'])
    features = result['features']

    st.subheader("SHAP Values Explanation")
    fig, ax = plt.subplots()
    ax.barh(features, shap_values[0])
    ax.set_xlabel('SHAP Value (Impact on Model Output)')
    st.pyplot(fig)

# CSV File processing and batch predictions
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_columns = ['transaction_amount', 'customer_age', 'customer_balance']

    if all(column in df.columns for column in required_columns):
        #creating a new transaction_amount_to_balance_ratio feature
        df['transaction_amount_to_balance_ratio'] = df['transaction_amount'] / df['customer_balance']

        if st.button("Run Batch Predictions"):
            predictions = []

            for _, row in df.iterrows():
                row_data = {
                    "transaction_amount": row['transaction_amount'],
                    "customer_age": row['customer_age'],
                    "customer_balance": row['customer_balance']
                }

                response = requests.post(f"{api_url}/predict/", json=row_data)
                prediction_result = response.json()

                predictions.append({
                    "transaction_amount": row['transaction_amount'],
                    "customer_age": row['customer_age'],
                    "customer_balance": row['customer_balance'],
                    "fraud_prediction": prediction_result['fraud_prediction'],
                    "confidence_fraud": prediction_result['confidence'][1],
                    "transaction_amount_to_balance_ratio": row['transaction_amount_to_balance_ratio']
                })

            predictions_df = pd.DataFrame(predictions)
            st.subheader("Batch Prediction Results")
            st.dataframe(predictions_df.head())

            csv_output = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv_output,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        # Visualization of transactions
        st.subheader("Transaction Insights Visualization")

        numeric_columns = ['transaction_amount', 'customer_age', 'customer_balance', 'transaction_amount_to_balance_ratio']

        x_axis = st.selectbox("Select X-axis column", numeric_columns)
        y_axis = st.selectbox("Select Y-axis column", numeric_columns, index=1)

        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis], alpha=0.7)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"Scatter plot: {x_axis} vs {y_axis}")
        st.pyplot(fig)
    else:
        missing_cols = [col for col in required_columns if col not in df.columns]
        st.error(f"Missing columns: {', '.join(missing_cols)}")

