import streamlit as st
import requests

st.title("Currency Classifier")

# Input fields for the 4 features
variance = st.number_input("Variance", format="%.5f")
skewness = st.number_input("Skewness", format="%.5f")
curtosis = st.number_input("Curtosis", format="%.5f")
entropy = st.number_input("Entropy", format="%.5f")

if st.button("Predict"):
    # Build the input for the FastAPI POST
    input_data = {
        "values": [variance, skewness, curtosis, entropy]
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Exception occurred: {e}")
