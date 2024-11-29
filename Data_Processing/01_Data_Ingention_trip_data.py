# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, unix_timestamp, dayofweek, hour
from pyspark.sql.types import StructType, StructField, TimestampType, IntegerType, FloatType, StringType

# Create a Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Trip Data Analysis") \
    .getOrCreate()
    
# Define paths
folder_path = "dbfs:/FileStore/tables/"  # Folder where files are stored
delta_path = "dbfs:/FileStore/tables/nyc_taxi_data_delta/"  # Destination Delta path

# Generate a list of parquet files for each month of 2023 with the updated filename format
months = [f"yellow_tripdata_2023_{str(i).zfill(2)}.parquet" for i in range(1, 13)]

# Load the first month's data with all necessary columns
trip_data = spark.read.parquet(folder_path + months[0]) \
                      .select("tpep_pickup_datetime", "tpep_dropoff_datetime", 
                              "PULocationID", "DOLocationID", 
                              "passenger_count", "trip_distance", 
                              "payment_type", "fare_amount", 
                              "tolls_amount", "improvement_surcharge", 
                              "total_amount") \
                      .withColumn("month", lit(months[0][21:28])) \
                      .withColumn("trip_duration", 
                                  (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60) \
                      .withColumn("pickup_day_of_week", dayofweek("tpep_pickup_datetime")) \
                      .withColumn("pickup_hour", hour("tpep_pickup_datetime"))

# Union the rest of the months
for month in months[1:]:
    temp_df = spark.read.parquet(folder_path + month) \
                        .select("tpep_pickup_datetime", "tpep_dropoff_datetime", 
                                "PULocationID", "DOLocationID", 
                                "passenger_count", "trip_distance", 
                                "payment_type", "fare_amount", 
                                "tolls_amount", "improvement_surcharge", 
                                "total_amount") \
                        .withColumn("month", lit(month[21:28])) \
                        .withColumn("trip_duration", 
                                    (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60) \
                        .withColumn("pickup_day_of_week", dayofweek("tpep_pickup_datetime")) \
                        .withColumn("pickup_hour", hour("tpep_pickup_datetime"))
    
    # Union the new month's data with the existing data
    trip_data = trip_data.union(temp_df)

# Check the schema and data types
trip_data.printSchema()

# Convert data types if necessary
nyc_taxi_df = trip_data \
    .withColumn("tpep_pickup_datetime", trip_data["tpep_pickup_datetime"].cast(TimestampType())) \
    .withColumn("tpep_dropoff_datetime", trip_data["tpep_dropoff_datetime"].cast(TimestampType())) \
    .withColumn("PULocationID", trip_data["PULocationID"].cast(IntegerType())) \
    .withColumn("DOLocationID", trip_data["DOLocationID"].cast(IntegerType())) \
    .withColumn("passenger_count", trip_data["passenger_count"].cast(IntegerType())) \
    .withColumn("trip_distance", trip_data["trip_distance"].cast(FloatType())) \
    .withColumn("payment_type", trip_data["payment_type"].cast(StringType())) \
    .withColumn("fare_amount", trip_data["fare_amount"].cast(FloatType())) \
    .withColumn("tolls_amount", trip_data["tolls_amount"].cast(FloatType())) \
    .withColumn("improvement_surcharge", trip_data["improvement_surcharge"].cast(FloatType())) \
    .withColumn("total_amount", trip_data["total_amount"].cast(FloatType())) \
    .withColumn("month", trip_data["month"].cast(StringType())) \
    .withColumn("trip_duration", trip_data["trip_duration"].cast(FloatType())) \
    .withColumn("pickup_day_of_week", trip_data["pickup_day_of_week"].cast(IntegerType())) \
    .withColumn("pickup_hour", trip_data["pickup_hour"].cast(IntegerType()))

# Check the first few rows to verify
display(nyc_taxi_df.limit(5))

# COMMAND ----------


# Write the combined DataFrame in Delta Lake format
nyc_taxi_df.write.format("delta").mode("overwrite").save(delta_path)

# COMMAND ----------

loaded_df = spark.read.format("delta").load(delta_path)
loaded_df.show()

# COMMAND ----------

# Display count of rows
print(f"Total records: {nyc_taxi_df.count()}")

# Describe dataset to get an overview of each column
nyc_taxi_df.describe().show()


# COMMAND ----------


