import streamlit as st
import pandas as pd
import joblib
from azure.storage.blob import BlobServiceClient
import os

# Azure Blob Storage configuration
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=nycdemov1;AccountKey=YOUR_ACCOUNT_KEY;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "nycdatabrick"
MODEL_BLOB_NAME = "models/random_forest_model.pkl"

# Temporary file path to download the model
TEMP_MODEL_PATH = "/tmp/random_forest_model.pkl"

# Download the model from Azure Blob Storage
@st.cache_resource
def load_model():
    # Connect to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=MODEL_BLOB_NAME)
    
    # Download the model file if not already downloaded
    if not os.path.exists(TEMP_MODEL_PATH):
        with open(TEMP_MODEL_PATH, "wb") as model_file:
            model_file.write(blob_client.download_blob().readall())
    
    # Load the model with joblib
    return joblib.load(TEMP_MODEL_PATH)

# Load the trained model
model = load_model()

# Define borough and passenger count options
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
    # Create a sample input DataFrame
    input_data = pd.DataFrame({
        "pickup_borough": [pickup_borough],
        "dropoff_borough": [dropoff_borough],
        "passenger_count": [passenger_count]
    })

    # Make predictions using the loaded model
    predicted_price = model.predict(input_data)[0]

    # Display the predicted price
    st.write(f"**Predicted Price:** ${predicted_price:.2f}")

# Add a note for the user
st.info("The price prediction is based on a trained Random Forest regression model.")
