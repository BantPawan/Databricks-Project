import streamlit as st
import pandas as pd
import joblib
from azure.storage.blob import BlobServiceClient
import zipfile
import os

# Azure Storage Credentials
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=nycdemov1;AccountKey=KQVhKXYfyjKTg4ZNQjxIDyOXkhOEpGvdgP6Dq8A8jwgzIZ9N9hNLwj5yig4hoa+eaDtqi95kj+FP+AStXe5FiA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "nycdatabrick"

# Streamlit Cache: Download Model
@st.cache_resource
def download_and_load_model():
    model_blob_name = "models/random_forest_model.zip"
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=model_blob_name)

    # Download and extract the model
    temp_zip_path = "/tmp/random_forest_model.zip"
    with open(temp_zip_path, "wb") as file:
        file.write(blob_client.download_blob().readall())
    
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall("/tmp/random_forest_model")
    
    # Load the model (assuming it's in .pkl format)
    model_path = "/tmp/random_forest_model/model.pkl"
    model = joblib.load(model_path)

    return model

# Load model
model = download_and_load_model()

# Borough and passenger count options
PICKUP_BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]
DROPOFF_BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]
PASSENGER_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Streamlit UI
st.title("NYC Taxi Price Prediction")

# User input
pickup_borough = st.selectbox("Select Pickup Borough", PICKUP_BOROUGHS)
dropoff_borough = st.selectbox("Select Dropoff Borough", DROPOFF_BOROUGHS)
passenger_count = st.selectbox("Select Passenger Count", PASSENGER_COUNTS)

# Prediction logic
if st.button("Predict Price"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        "pickup_borough": [pickup_borough],
        "dropoff_borough": [dropoff_borough],
        "passenger_count": [passenger_count]
    })

    # Make prediction
    try:
        predicted_price = model.predict(input_data)[0]
        st.write(f"**Predicted Price:** ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Add an info message
st.info("The price prediction is based on a trained Random Forest model.")
