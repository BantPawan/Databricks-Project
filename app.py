import os
import requests
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.pipeline import PipelineModel

# Model path
MODEL_DIR = os.path.join("rf_model", "random_forest_model")

# URL to the model in your GitHub repository
MODEL_URL = "https://github.com/yourusername/yourrepository/raw/main/rf_model/random_forest_model"

# Initialize Spark session
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("ModelPrediction") \
        .getOrCreate()

spark = init_spark()

# Function to download the model from GitHub if not already present
def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(os.path.join(MODEL_DIR, "random_forest_model"), "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                st.success("Model downloaded successfully.")
            else:
                st.error(f"Failed to download model. HTTP Status: {response.status_code}")
        except Exception as e:
            st.error(f"Error downloading model: {e}")

# Function to load the pipeline model
@st.cache_resource
def load_model():
    try:
        # Load the pipeline model from the directory
        model = PipelineModel.load(MODEL_DIR)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to map categorical inputs to numerical values
def map_categorical_inputs(input_value, mapping_dict):
    return mapping_dict.get(input_value, -1)  # Return -1 if value is not in mapping

# Main Streamlit app
def main():
    st.title("Optimized Random Forest Model for Predictions")

    # Step 1: Download model if not present
    download_model()

    # Step 2: Load the model
    model = load_model()
    if model:
        st.success("Model Loaded Successfully!")

        # Step 3: User Input for Prediction
        with st.form("prediction_form"):
            passenger_count = st.selectbox("Passenger Count", [1, 2, 3, 4, 5, 6], index=0)
            payment_type = st.selectbox("Payment Type", ["Card", "Cash"])
            trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, step=1)
            pickup_day_of_week = st.selectbox("Pickup Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            pickup_hour = st.slider("Pickup Hour", 0, 23, value=12)
            pickup_month = st.slider("Pickup Month", 1, 12, value=1)
            pickup_borough = st.selectbox("Pickup Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
            dropoff_borough = st.selectbox("Dropoff Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
            is_holiday = st.selectbox("Is Holiday?", ["Yes", "No"])
            distance_bin = st.selectbox("Distance Bin", ["Short", "Medium", "Long"], index=1)
            time_of_day_bin = st.selectbox("Time of Day Bin", ["Morning", "Afternoon", "Evening", "Night"], index=0)
            near_airport = st.selectbox("Near Airport?", ["Yes", "No"])

            # Submit button for form
            submitted = st.form_submit_button("Predict")

        # Step 4: Prediction
        if submitted:
            # Map categorical inputs to numerical values
            payment_type_map = {"Card": 1, "Cash": 2}
            is_holiday_map = {"Yes": 1, "No": 0}
            pickup_day_of_week_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

            single_query = [(
                passenger_count,
                map_categorical_inputs(payment_type, payment_type_map),
                trip_duration,
                map_categorical_inputs(pickup_day_of_week, pickup_day_of_week_map),
                pickup_hour,
                pickup_month,
                pickup_borough,
                dropoff_borough,
                map_categorical_inputs(is_holiday, is_holiday_map),
                distance_bin,
                time_of_day_bin,
                near_airport
            )]

            # Define schema for input DataFrame
            schema = StructType([
                StructField("passenger_count", IntegerType(), True),
                StructField("payment_type", IntegerType(), True),
                StructField("trip_duration", IntegerType(), True),
                StructField("pickup_day_of_week", IntegerType(), True),
                StructField("pickup_hour", IntegerType(), True),
                StructField("pickup_month", IntegerType(), True),
                StructField("pickup_borough", StringType(), True),
                StructField("dropoff_borough", StringType(), True),
                StructField("is_holiday", IntegerType(), True),
                StructField("distance_bin", StringType(), True),
                StructField("time_of_day_bin", StringType(), True),
                StructField("near_airport", StringType(), True)
            ])

            try:
                test_data_df = spark.createDataFrame(single_query, schema=schema)
                predictions = model.transform(test_data_df)
                prediction_result = predictions.select("prediction").collect()

                # Display the prediction
                st.write("### Prediction Result:")
                for row in prediction_result:
                    st.write(f"Predicted Value: {row['prediction']}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
