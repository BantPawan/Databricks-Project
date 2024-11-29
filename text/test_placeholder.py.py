# Databricks notebook source
import boto3
import os

aws_access_key = 'access_key'
aws_secret_key = 'secret_key'

s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

s3_bucket = 'nyc-databricks-bucket'
s3_key = '/tmp/rf_model.tar.gz'
local_path = '/tmp/rf_model.tar.gz'

os.environ['AWS_ACCESS_KEY_ID'] = 'access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'secret_key'

s3.upload_file(local_path, s3_bucket, s3_key)


# COMMAND ----------

# Define local path to save the file
local_path_download = '/tmp/rf_model_downloaded.tar.gz'

# Download the file from S3 to local machine
s3.download_file(s3_bucket, s3_key, local_path_download)

print(f"File downloaded successfully to {local_path_download}")

# COMMAND ----------

import tarfile
import os

# Path to the downloaded .tar.gz file in DBFS
tar_file_path = '/dbfs/tmp/rf_model.tar.gz'

# Directory to extract the contents to
extract_dir = '/dbfs/tmp/random_forest_model/'

# Check if the tar file exists
if os.path.exists(tar_file_path):
    # Ensure the extraction directory exists
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Extract the .tar.gz file
    try:
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)

        print(f"Model extracted to {extract_dir}")
    except Exception as e:
        print(f"Error while extracting the tar file: {e}")
else:
    print(f"Error: The file {tar_file_path} does not exist.")


# COMMAND ----------

# List files in the dbfs:/tmp/ directory
files = dbutils.fs.ls("dbfs:/tmp/rf_model/random_forest_model/")
for file in files:
    print(file.path)

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Example single query input data
single_query = [
    (1, 1, 20.37, 2, 8, 10, "Manhattan", "Brooklyn", 0, "Medium", "Morning", 0)
]

# Define the schema of your input data based on the model's feature columns
columns = [
    "passenger_count", "payment_type", "trip_duration", 
    "pickup_day_of_week", "pickup_hour", "pickup_month", "pickup_borough", 
    "dropoff_borough", "is_holiday", "distance_bin", "time_of_day_bin", "near_airport"
]

# Initialize SparkSession
spark = SparkSession.builder.appName("ModelPrediction").getOrCreate()

# Convert the input query to a DataFrame
test_data_df = spark.createDataFrame(single_query, columns)

# Show the test data to verify the schema
print("Test Data:")
test_data_df.show()

# COMMAND ----------

from pyspark.ml.pipeline import PipelineModel

# Define the correct path to the model directory
model_path = "dbfs:/tmp/rf_model/random_forest_model"

# Load the model
try:
    model = PipelineModel.load(model_path)
    print(f"Loaded the model from {model_path}")
except Exception as e:
    print(f"Failed to load the model: {str(e)}")

# COMMAND ----------

transform_data = model.transform(test_data_df)
# Show the transformed data with predictions
print("Transformed Data with Predictions:")
transform_data.select("scaledFeatures", "prediction").show()
