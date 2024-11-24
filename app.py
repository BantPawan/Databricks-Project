import os
import tempfile
import streamlit as st
from azure.storage.blob import BlobServiceClient
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# Azure Storage Configuration
STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Securely fetch from environment
CONTAINER_NAME = "nycdatabrick"
BLOB_NAME = "random_forest_model"

if not STORAGE_CONNECTION_STRING:
    st.error("Azure Storage Connection String is not set. Please configure it.")
else:
    @st.cache_resource
    def init_spark_session():
        """Initialize a Spark session."""
        return SparkSession.builder.appName("ModelPrediction").getOrCreate()

    @st.cache_resource
    def load_model_from_blob():
        """Load the model directly from Azure Blob Storage."""
        try:
            st.info("Fetching model from Azure Blob Storage...")
            blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

            # Download blob content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
                temp_model_file.write(blob_client.download_blob().readall())
                temp_model_path = temp_model_file.name

            st.success("Model fetched successfully!")
            return PipelineModel.load(temp_model_path)
        except Exception as e:
            st.error(f"Error loading model from Azure Blob: {e}")
            return None

    # Initialize Spark session and load model
    spark = init_spark_session()
    model = load_model_from_blob()

    # Ensure model is loaded
    if model:
        st.title("Taxi Fare Prediction")

        # Input fields for user data
        st.sidebar.header("Enter Trip Details:")
        passenger_count = st.number_input(
            "Passenger Count", min_value=1, max_value=6, value=1, help="Number of passengers for the trip"
        )
        payment_type = st.sidebar.selectbox(
            "Payment Type", [1, 2], help="1 for Card, 2 for Cash"
        )
        trip_duration = st.number_input(
            "Trip Duration (minutes)", min_value=1, step=1, help="Duration of the trip in minutes"
        )
        distance_km = st.number_input(
            "Distance (km)", min_value=0.0, step=0.1, help="Distance of the trip in kilometers"
        )
        pickup_day_of_week = st.selectbox(
            "Pickup Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        pickup_hour = st.slider("Pickup Hour", 0, 23, value=12, help="Hour of the day for pickup (0-23)")
        pickup_month = st.slider("Pickup Month", 1, 12, value=1, help="Month of the year for pickup")
        pickup_borough = st.selectbox(
            "Pickup Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        )
        dropoff_borough = st.selectbox(
            "Dropoff Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        )
        is_holiday = st.selectbox("Is Holiday?", ["Yes", "No"])
        distance_bin = st.sidebar.selectbox("Distance Bin", ["Short", "Medium", "Long"], index=1)
        time_of_day_bin = st.sidebar.selectbox("Time of Day Bin", ["Morning", "Afternoon", "Evening", "Night"], index=0)
        near_airport = st.sidebar.selectbox("Near Airport?", ["Yes", "No"])

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
