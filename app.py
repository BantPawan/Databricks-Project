import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os

# Azure Storage Credentials
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=nycdemov1;AccountKey=KQVhKXYfyjKTg4ZNQjxIDyOXkhOEpGvdgP6Dq8A8jwgzIZ9N9hNLwj5yig4hoa+eaDtqi95kj+FP+AStXe5FiA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "nycdatabrick"
DATA_BLOB_NAME = "taxi_data_parquet/taxi_data.parquet"  # Correct path to the parquet file inside the container
MODEL_BLOB_NAME = "models/random_forest_model.pkl"  # Path for model
PARAMS_BLOB_NAME = "models/random_forest_model_params.pkl"  # Path for feature importances

# Temporary directory for caching downloaded files
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Predefined lists for pickup and dropoff boroughs, and passenger count
PICKUP_BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]
DROPOFF_BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]
PASSENGER_COUNTS = [1, 2, 3, 4, 0, 5, 6, 8, 7, 9]

# Helper function to download files to local temp directory
def download_to_local(blob_client, file_name):
    # Ensure the full directory path exists
    file_path = os.path.join(TEMP_DIR, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    if not os.path.exists(file_path):  # Only download if file does not exist
        with open(file_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
    return file_path

# Load dataset from Azure Blob Storage
@st.cache_data
def load_dataset():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=DATA_BLOB_NAME)
        file_path = download_to_local(blob_client, "taxi_data.parquet")
        dataset = pd.read_parquet(file_path)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load model and parameters from Azure Blob Storage
@st.cache_resource
def load_model_and_params():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

        # Load model
        model_blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=MODEL_BLOB_NAME)
        model_path = download_to_local(model_blob_client, MODEL_BLOB_NAME)
        model = joblib.load(model_path)

        # Load feature importances
        params_blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=PARAMS_BLOB_NAME)
        params_path = download_to_local(params_blob_client, PARAMS_BLOB_NAME)
        feature_importances = joblib.load(params_path)

        return model, feature_importances
    except Exception as e:
        st.error(f"Error loading model or parameters: {e}")
        return None, None

# Plot map function
def plot_map(pickup, dropoff, dataset):
    try:
        # Get coordinates for pickup and dropoff locations
        pickup_coords = dataset.loc[dataset['pickup_borough'] == pickup, ['pickup_lat', 'pickup_long']].iloc[0]
        dropoff_coords = dataset.loc[dataset['dropoff_borough'] == dropoff, ['dropoff_lat', 'dropoff_long']].iloc[0]

        # Create map
        m = folium.Map(location=[pickup_coords['pickup_lat'], pickup_coords['pickup_long']], zoom_start=12)
        folium.Marker([pickup_coords['pickup_lat'], pickup_coords['pickup_long']], tooltip="Pickup",
                      icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([dropoff_coords['dropoff_lat'], dropoff_coords['dropoff_long']], tooltip="Dropoff",
                      icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine([(pickup_coords['pickup_lat'], pickup_coords['pickup_long']),
                         (dropoff_coords['dropoff_lat'], dropoff_coords['dropoff_long'])], color="blue").add_to(m)

        return m
    except Exception as e:
        st.error(f"Error plotting map: {e}")
        return None

# Streamlit app main function
def main():
    st.title("Taxi Fare Prediction App")

    # Load dataset and model
    dataset = load_dataset()
    if dataset is None:
        st.stop()  # Stop the app if dataset couldn't be loaded

    model, feature_importances = load_model_and_params()
    if model is None or feature_importances is None:
        st.stop()  # Stop the app if model or feature importances couldn't be loaded

    # Display feature importances
    st.subheader("Model Feature Importances")
    st.write(feature_importances)

    # User input for pickup and dropoff locations (using predefined lists)
    pickup = st.selectbox("Select Pickup Borough", PICKUP_BOROUGHS)
    dropoff = st.selectbox("Select Dropoff Borough", DROPOFF_BOROUGHS)
    passenger_count = st.selectbox("Select Passenger Count", PASSENGER_COUNTS)

    # Predict button
    if st.button("Predict Price"):
        try:
            # Validate if required columns exist
            required_columns = ['pickup_borough', 'dropoff_borough', 'passenger_count']
            for col in required_columns:
                if col not in dataset.columns:
                    raise ValueError(f"Column '{col}' is missing from the dataset.")

            # Prepare input for prediction
            input_data = pd.DataFrame({
                'pickup_borough': [pickup],
                'dropoff_borough': [dropoff],
                'passenger_count': [passenger_count]
            })

            # Make prediction
            predicted_price = model.predict(input_data)[0]
            st.success(f"Predicted Fare: ₹{predicted_price:.2f}")

            # Map visualization
            st.subheader("Route Map")
            route_map = plot_map(pickup, dropoff, dataset)
            if route_map:
                folium_static(route_map)
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
