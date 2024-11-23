import streamlit as st
import pandas as pd
import joblib
import zipfile
from azure.storage.blob import BlobServiceClient
import os

# Azure Storage Credentials
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=nycdemov1;AccountKey=KQVhKXYfyjKTg4ZNQjxIDyOXkhOEpGvdgP6Dq8A8jwgzIZ9N9hNLwj5yig4hoa+eaDtqi95kj+FP+AStXe5FiA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "nycdatabrick"
MODEL_ZIP_BLOB_NAME = "models/random_forest_model.zip"  # Path for zipped model
PARAMS_BLOB_NAME = "models/random_forest_model_params.pkl"  # Path for feature importances

# Temporary file paths to download the model and parameters
TEMP_MODEL_ZIP_PATH = "/tmp/random_forest_model.zip"
TEMP_PARAMS_PATH = "/tmp/random_forest_model_params.pkl"
TEMP_MODEL_PATH = "/tmp/random_forest_model.pkl"  # Extracted model file path

# Function to download blobs from Azure Blob Storage
def download_blob_from_azure(blob_name, temp_path):
    try:
        # Connect to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        # Download the blob if not already downloaded
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            with open(temp_path, "wb") as file:
                file.write(blob_client.download_blob().readall())
        
        # Validate file integrity
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"File not found at {temp_path}")
        
        st.success(f"Downloaded {blob_name} successfully!")
    except Exception as e:
        st.error(f"Failed to download the blob: {e}")
        raise

# Function to unzip the model file
def unzip_model(zip_path, extract_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        st.success(f"Model extracted successfully from {zip_path}!")
    except Exception as e:
        st.error(f"Failed to unzip the model file: {e}")
        raise

# Load model and parameters
@st.cache_resource
def load_model_and_params():
    # Download model and parameters from Azure Blob Storage
    download_blob_from_azure(MODEL_ZIP_BLOB_NAME, TEMP_MODEL_ZIP_PATH)
    download_blob_from_azure(PARAMS_BLOB_NAME, TEMP_PARAMS_PATH)
    
    # Unzip the model file
    unzip_model(TEMP_MODEL_ZIP_PATH, "/tmp/")
    
    # Load the model from the extracted .pkl file
    model = joblib.load(TEMP_MODEL_PATH)
    
    # Load the parameters
    params = joblib.load(TEMP_PARAMS_PATH)
    
    return model, params

# Load the trained model and parameters
model, params = load_model_and_params()

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

    # Display feature importances (if needed)
    st.write("**Feature Importances:**")
    feature_importances = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": params['feature_importances']  # Assuming params contains feature importances
    }).sort_values(by="Importance", ascending=False)

    st.write(feature_importances)

# Add a note for the user
st.info("The price prediction is based on a trained Random Forest regression model.")
