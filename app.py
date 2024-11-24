import os
import streamlit as st
from pyspark.sql import SparkSession

# Azure Storage Configuration
STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Fetch from environment variable
CONTAINER_NAME = "nycdatabrick"
BLOB_NAME = "random_forest_model"

if not STORAGE_CONNECTION_STRING:
    st.error("Azure Storage Connection String is not set. Please configure it.")
else:
    @st.cache_resource
    def init_spark_session():
        """Initialize a Spark session."""
        return SparkSession.builder.appName("ModelPrediction").getOrCreate()

    # Initialize Spark session
    spark = init_spark_session()

    # Set Azure Blob Storage connection string for accessing data
    spark.conf.set("spark.hadoop.fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem")
    spark.conf.set("spark.hadoop.fs.azure.account.key.nycdemov1.blob.core.windows.net", STORAGE_CONNECTION_STRING)

    # Model path on Azure Blob Storage
    model_path = f"wasbs://{CONTAINER_NAME}@nycdemov1.blob.core.windows.net/{BLOB_NAME}"

    # Load the model from Azure Blob Storage
    try:
        model = spark.read.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from Azure Blob: {e}")
        model = None

    # Ensure model is loaded
    if model:
        st.title("Taxi Fare Prediction")

        # Input fields for user data
        passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
        payment_type = st.selectbox("Payment Type", [1, 2])  # 1 for Card, 2 for Cash
        trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, step=1)
        distance_km = st.number_input("Distance (km)", min_value=0.0, step=0.1)
        pickup_day_of_week = st.selectbox("Pickup Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        pickup_hour = st.slider("Pickup Hour", 0, 23, value=12)
        pickup_month = st.slider("Pickup Month", 1, 12, value=1)
        pickup_borough = st.selectbox("Pickup Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
        dropoff_borough = st.selectbox("Dropoff Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
        is_holiday = st.selectbox("Is Holiday?", ["Yes", "No"])
        distance_bin = st.selectbox("Distance Bin", ["Short", "Medium", "Long"], index=1)
        time_of_day_bin = st.selectbox("Time of Day Bin", ["Morning", "Afternoon", "Evening", "Night"], index=0)
        near_airport = st.selectbox("Near Airport?", ["Yes", "No"])

        # Create a DataFrame for the input query
        single_query = [(passenger_count, payment_type, trip_duration, distance_km, 
                         pickup_day_of_week, pickup_hour, pickup_month, pickup_borough, 
                         dropoff_borough, is_holiday, distance_bin, time_of_day_bin, near_airport)]
        columns = [
            "passenger_count", "payment_type", "trip_duration", "distance_km",
            "pickup_day_of_week", "pickup_hour", "pickup_month", "pickup_borough", 
            "dropoff_borough", "is_holiday", "distance_bin", "time_of_day_bin", "near_airport"
        ]
        test_data_df = spark.createDataFrame(single_query, columns)

        # Show the input data
        st.write("Input Data:")
        st.dataframe(test_data_df.toPandas())

        # Make predictions
        if st.button("Predict"):
            try:
                predictions = model.transform(test_data_df)
                prediction_result = predictions.select("prediction").collect()[0]["prediction"]
                st.success(f"Predicted Fare: ${prediction_result:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.error("Model could not be loaded. Please check your Azure Blob Storage setup.")
