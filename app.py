  import streamlit as st
import pandas as pd
import joblib
import zipfile
import os
from azure.storage.blob import BlobServiceClient
from pyspark.sql import SparkSession


# Initialize Spark
@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("Streamlit Cloud App") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    return spark


spark = init_spark()

# Azure Storage Credentials
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=nycdemov1;AccountKey=KQVhKXYfyjKTg4ZNQjxIDyOXkhOEpGvdgP6Dq8A8jwgzIZ9N9hNLwj5yig4hoa+eaDtqi95kj+FP+AStXe5FiA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "nycdatabrick"

# Download RandomForest model
model_blob_name = "models/random_forest_model.zip"
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=model_blob_name)

# Save the model locally
with open("/tmp/random_forest_model.zip", "wb") as file:
    file.write(blob_client.download_blob().readall())

# Extract and load the model
with zipfile.ZipFile("/tmp/random_forest_model.zip", 'r') as zip_ref:
    zip_ref.extractall("/tmp/random_forest_model")

TEMP_MODEL_PATH = "/tmp/random_forest_model/model.pkl"  # Specify the extracted model path
TEMP_PARAMS_PATH = "/tmp/random_forest_model/params.pkl"  # Specify parameters path


# Load model and parameters
@st.cache_resource
def load_model_and_params():
    # Load the model
    model = joblib.load(TEMP_MODEL_PATH)

    # Load the parameters
    if os.path.exists(TEMP_PARAMS_PATH):
        params = joblib.load(TEMP_PARAMS_PATH)
    else:
        params = {"feature_importances": [0] * 3}  # Default values if params.pkl is missing

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

    # Display feature importances (if available)
    st.write("**Feature Importances:**")
    feature_importances = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": params.get('feature_importances', [0] * len(input_data.columns))
    }).sort_values(by="Importance", ascending=False)

    st.write(feature_importances)

# Add a note for the user
st.info("The price prediction is based on a trained Random Forest model.")
