import streamlit as st
import joblib
import os
import shutil
from azure.storage.blob import BlobServiceClient

# Azure Storage Connection Settings
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=nycdemov1;AccountKey=KQVhKXYfyjKTg4ZNQjxIDyOXkhOEpGvdgP6Dq8A8jwgzIZ9N9hNLwj5yig4hoa+eaDtqi95kj+FP+AStXe5FiA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "nycdatabrick"

# Define Model Paths
DBFS_MODEL_DIR = "/dbfs/tmp/random_forest_model"
LOCAL_MODEL_DIR = "/tmp/random_forest_model"
DBFS_MODEL_ZIP_PATH = "/dbfs/tmp/random_forest_model.zip"
LOCAL_MODEL_ZIP_PATH = "/tmp/random_forest_model.zip"
PARAMS_BLOB_NAME = "models/random_forest_model_params.pkl"
MODEL_BLOB_NAME = "models/random_forest_model.zip"

# Streamlit App
st.title("Random Forest Model Management")

# Upload Model Feature Importances
st.header("Upload Feature Importances")

# Azure Blob Client Initialization
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

# Load Feature Importances
try:
    params_file_path = f"{DBFS_MODEL_DIR}/params.pkl"
    if os.path.exists(params_file_path):
        feature_importances = joblib.load(params_file_path)
        st.success("Feature importances loaded successfully!")
        st.write("Feature Importances:", feature_importances)
    else:
        st.warning("Feature importance file not found.")
except Exception as e:
    st.error(f"Error loading feature importances: {e}")

# Upload Feature Importances to Azure Blob
if st.button("Upload Feature Importances to Azure"):
    try:
        # Blob Client
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=PARAMS_BLOB_NAME)
        
        # Upload the file
        with open(params_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        st.success(f"Feature importances uploaded successfully to: {CONTAINER_NAME}/{PARAMS_BLOB_NAME}")
    except Exception as e:
        st.error(f"Error uploading feature importances: {e}")

# Upload Model Pipeline to Azure
st.header("Upload Random Forest Model")

# Display Extracted Model Files
if os.path.exists(DBFS_MODEL_DIR):
    st.write("Available Model Files:")
    model_files = os.listdir(DBFS_MODEL_DIR)
    st.write(model_files)
else:
    st.warning("Model directory not found.")

# Zip and Upload Model
if st.button("Zip and Upload Model to Azure"):
    try:
        # Clean Local Directory
        if os.path.exists(LOCAL_MODEL_DIR):
            shutil.rmtree(LOCAL_MODEL_DIR)

        # Copy and Zip Model Directory
        shutil.copytree(DBFS_MODEL_DIR, LOCAL_MODEL_DIR)
        shutil.make_archive(LOCAL_MODEL_ZIP_PATH.replace(".zip", ""), "zip", LOCAL_MODEL_DIR)

        # Move Zip to DBFS Path
        shutil.move(LOCAL_MODEL_ZIP_PATH, DBFS_MODEL_ZIP_PATH)

        # Upload to Azure
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=MODEL_BLOB_NAME)
        with open(DBFS_MODEL_ZIP_PATH, "rb") as model_zip:
            blob_client.upload_blob(model_zip, overwrite=True)

        st.success(f"Model uploaded successfully to Azure Blob: {CONTAINER_NAME}/{MODEL_BLOB_NAME}")
    except Exception as e:
        st.error(f"Error zipping or uploading model: {e}")

# Download Model from Azure
st.header("Download Model from Azure")
if st.button("Download Model from Azure"):
    try:
        # Blob Client
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=MODEL_BLOB_NAME)

        # Download the file
        with open(LOCAL_MODEL_ZIP_PATH, "wb") as file:
            file.write(blob_client.download_blob().readall())
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading model: {e}")
