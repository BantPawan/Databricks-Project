import os
import shutil
import requests
from pyspark.ml import PipelineModel
import streamlit as st

# URL to the model in Azure Blob Storage
MODEL_URL = "https://nycdemov1.blob.core.windows.net/nycdatabrick/models/random_forest_model.zip"

# Local paths in Streamlit
local_model_dir = "random_forest_model"
local_zip_path = f"{local_model_dir}.zip"

# Download the model zip
st.write("Downloading model from Azure Blob Storage...")
response = requests.get(MODEL_URL, stream=True)
with open(local_zip_path, 'wb') as f:
    shutil.copyfileobj(response.raw, f)

# Extract the zip file
st.write("Extracting model files...")
shutil.unpack_archive(local_zip_path, local_model_dir)

# Load the model
st.write("Loading the model...")
model_path = os.path.join(local_model_dir, "random_forest_model")  # Adjust path if needed
loaded_model = PipelineModel.load(model_path)

st.success("Model loaded successfully!")

# Streamlit UI for predictions
st.title("Price Prediction App")

# User inputs
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
payment_type = st.selectbox("Payment Type", ["Credit Card", "Cash", "Other"])
trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, value=20)
distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0)
pickup_borough = st.selectbox("Pickup Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
dropoff_borough = st.selectbox("Dropoff Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])

# Predict button
if st.button("Predict"):
    # Initialize Spark session
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("StreamlitApp").getOrCreate()
    
    # Prepare input data
    data = [(passenger_count, payment_type, trip_duration, distance_km, pickup_borough, dropoff_borough)]
    columns = ["passenger_count", "payment_type", "trip_duration", "distance_km", "pickup_borough", "dropoff_borough"]
    input_df = spark.createDataFrame(data, columns)
    
    # Perform prediction
    predictions = loaded_model.transform(input_df)
    predicted_price = predictions.select("prediction").collect()[0]["prediction"]
    
    # Show result
    st.success(f"Predicted Price: ${predicted_price:.2f}")
