import os
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel

# Model path
MODEL_DIR = "rf_model/random_forest_model"  # Adjust this to your correct path

# Initialize Spark session
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("ModelPrediction") \
        .getOrCreate()

spark = init_spark()

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
    return mapping_dict.get(input_value, 0)

# Main Streamlit app
def main():
    st.title("Nyc Taxi Fare App")

    model = load_model()
    if model:
        st.success("Model Loaded Successfully!")

        # Step 1: User Input for Prediction
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

        # Step 2: Prediction
        if submitted:
            # Map categorical inputs to numerical values
            payment_type_map = {"Card": 1, "Cash": 2}
            is_holiday_map = {"Yes": 1, "No": 0}
            pickup_day_of_week_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            near_airport_map = {"Yes": 1, "No": 0}  # Add mapping for near_airport

            # Prepare the single query
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
                map_categorical_inputs(near_airport, near_airport_map)  # Map near_airport
            )]

            columns = [
                "passenger_count", "payment_type", "trip_duration", "pickup_day_of_week",
                "pickup_hour", "pickup_month", "pickup_borough", "dropoff_borough",
                "is_holiday", "distance_bin", "time_of_day_bin", "near_airport"
            ]

            try:
                # Create a DataFrame with the mapped values
                test_data_df = spark.createDataFrame(single_query, columns)

                # Explicitly cast columns that are categorical to 'int' for correct processing
                test_data_df = test_data_df.withColumn("payment_type", test_data_df["payment_type"].cast("int"))
                test_data_df = test_data_df.withColumn("is_holiday", test_data_df["is_holiday"].cast("int"))
                test_data_df = test_data_df.withColumn("pickup_day_of_week", test_data_df["pickup_day_of_week"].cast("int"))
                test_data_df = test_data_df.withColumn("near_airport", test_data_df["near_airport"].cast("int"))

                # Perform prediction
                predictions = model.transform(test_data_df)
                prediction_result = predictions.select("prediction").collect()

                # Display the prediction result
                st.write("### Prediction Result:")
                for row in prediction_result:
                    st.write(f"Predicted Value: {row['prediction']}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
