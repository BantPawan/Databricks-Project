import os
import shutil
import requests
from pyspark.ml import PipelineModel
import streamlit as st
from pyspark.sql import SparkSession

# URL to the model in Azure Blob Storage
MODEL_URL = "https://nycdemov1.blob.core.windows.net/nycdatabrick/models/random_forest_model.zip"

# Local paths in Streamlit
local_model_dir = "random_forest_model"
local_zip_path = f"{local_model_dir}.zip"

# Initialize Spark session (only once)
spark = SparkSession.builder.master("local").appName("StreamlitApp").getOrCreate()

# Download the model zip
def download_model():
    st.write("Downloading model from Azure Blob Storage...")
    try:
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(local_zip_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            st.write("Model downloaded successfully.")
        else:
            st.error("Failed to download model. Please check the URL or network connection.")
    except Exception as e:
        st.error(f"An error occurred while downloading the model: {e}")

# Extract the zip file
def extract_model():
    st.write("Extracting model files...")
    try:
        shutil.unpack_archive(local_zip_path, local_model_dir)
        st.write("Model extracted successfully.")
    except Exception as e:
        st.error(f"Error extracting the model: {e}")

# Load the model
def load_model():
    try:
        model_path = os.path.join(local_model_dir, "random_forest_model")  # Adjust path if needed
        loaded_model = PipelineModel.load(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Streamlit UI for predictions
st.title("Price Prediction App")

# Display loading steps
if not os.path.exists(local_model_dir):
    download_model()
    extract_model()

# Load the model
if 'model' not in st.session_state:
    model = load_model()
    if model:
        st.session_state.model = model

# User inputs
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
payment_type = st.selectbox("Payment Type", ["Credit Card", "Cash", "Other"])
trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, value=20)
distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0)
pickup_borough = st.selectbox("Pickup Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
dropoff_borough = st.selectbox("Dropoff Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])

# Predict button
if st.button("Predict"):
    if 'model' in st.session_state:
        # Prepare input data
        data = [(passenger_count, payment_type, trip_duration, distance_km, pickup_borough, dropoff_borough)]
        columns = ["passenger_count", "payment_type", "trip_duration", "distance_km", "pickup_borough", "dropoff_borough"]
        input_df = spark.createDataFrame(data, columns)

        # Perform prediction
        predictions = st.session_state.model.transform(input_df)
        predicted_price = predictions.select("prediction").collect()[0]["prediction"]
        
        # Show result
        st.success(f"Predicted Price: ${predicted_price:.2f}")
    else:
        st.error("Model not loaded. Please try again later.")
