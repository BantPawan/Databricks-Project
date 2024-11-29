# Databricks notebook source
dbfs_subfolder_path = "/FileStore/tables/taxi_zones/"
dbutils.fs.mkdirs(dbfs_subfolder_path)
dbutils.fs.ls("/FileStore/tables/")

# COMMAND ----------

import zipfile
import os

# Define paths
dbfs_zip_path = "/FileStore/tables/taxi_zones/taxi_zones.zip"
local_zip_path = "/tmp/taxi_zones.zip"
local_extract_path = "/tmp/taxi_zones/"

# Copy the zip file from DBFS to a local path
dbutils.fs.cp(dbfs_zip_path, "file:" + local_zip_path)

# Unzip the file locally and list the contents
with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
    zip_ref.extractall(local_extract_path)

# Verify extracted files
extracted_files = os.listdir(local_extract_path)
print("Extracted files:", extracted_files)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import radians, cos, sin, asin, sqrt, col, udf, hour, dayofweek
from pyspark.sql.types import FloatType

# Create a Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Trip Analysis") \
    .getOrCreate()

# COMMAND ----------

# Check for the .shp file in the extracted directory
if "taxi_zones.shp" in extracted_files:
    import geopandas as gpd
    # Load the shapefile
    zones = gpd.read_file(os.path.join(local_extract_path, "taxi_zones.shp"))
    
    # Convert to WGS 84 coordinate system if necessary
    zones = zones.to_crs(epsg=4326)
else:
    print("Error: 'taxi_zones.shp' not found in extracted files.")

# COMMAND ----------

zones

# COMMAND ----------

# Extract 'LocationID', 'latitude', and 'longitude' (using centroids of the polygons)
zones['latitude'] = zones.geometry.centroid.y
zones['longitude'] = zones.geometry.centroid.x

# COMMAND ----------

# Select 'LocationID', 'latitude', and 'longitude'
location_lat_long_df = zones[['LocationID', 'latitude', 'longitude']]

# COMMAND ----------

# Convert GeoDataFrame to Spark DataFrame
location_lat_long_spark_df = spark.createDataFrame(location_lat_long_df)

# COMMAND ----------

display(location_lat_long_spark_df)

# COMMAND ----------

# List all files in the specified DBFS directory
dbutils.fs.ls("dbfs:/dbfs/FileStore/tables/data_processed_trip_data/")

# COMMAND ----------

# Load the Delta table
taxi_zones_df = spark.read.format("delta").load("dbfs:/dbfs/FileStore/tables/data_processed_taxi_zones/")

# Show the first few rows to verify
display(taxi_zones_df)

# COMMAND ----------

# Load the Delta table
taxi_df = spark.read.format("delta").load("dbfs:/dbfs/FileStore/tables/data_processed_trip_data/")

# Show the first few rows to verify
display(taxi_df)

# COMMAND ----------

location_lat_long_spark_df = location_lat_long_spark_df \
    .withColumnRenamed("latitude", "latitude") \
    .withColumnRenamed("longitude", "longitude") \
    .withColumnRenamed("LocationID", "LocationID")

# COMMAND ----------

# Step 2: Join on PULocationID for pickup coordinates
taxi_df = taxi_df.join(
    location_lat_long_spark_df.withColumnRenamed("LocationID", "PULocationID")
    .withColumnRenamed("latitude", "pickup_latitude")
    .withColumnRenamed("longitude", "pickup_longitude"),
    on="PULocationID",
    how="left"
)

# COMMAND ----------

# Step 3: Join on DOLocationID for dropoff coordinates
taxi_df = taxi_df.join(
    location_lat_long_spark_df.withColumnRenamed("LocationID", "DOLocationID")
    .withColumnRenamed("latitude", "dropoff_latitude")
    .withColumnRenamed("longitude", "dropoff_longitude"),
    on="DOLocationID",
    how="left"
)

# COMMAND ----------

# Show result to verify joins
taxi_df.show(5)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import math

# COMMAND ----------

# Define the Haversine function
def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0  
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(dlon / 2) ** 2))
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c  # Result in kilometers
    return distance

# COMMAND ----------

# Register the Haversine UDF
haversine_udf = udf(haversine, DoubleType())

# COMMAND ----------

# Add 'distance_km' column using the Haversine UDF, ensuring no null values
taxi_df = taxi_df.withColumn(
    "distance_km",
    haversine_udf(
        col("pickup_latitude"),
        col("pickup_longitude"),
        col("dropoff_latitude"),
        col("dropoff_longitude")
    )
).na.fill({"distance_km": 0})  
# Replace null distances with 0 if needed

# COMMAND ----------

taxi_df.show(5)

# COMMAND ----------

# Check the Shape and Schema
# Get the number of rows and columns
num_rows = taxi_df.count()
num_cols = len(taxi_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

# Get DataFrame information
taxi_df.printSchema()

# COMMAND ----------

taxi_df.printSchema()
taxi_zones_df.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F

# Renaming columns in taxi_zones_df to avoid conflicts when joining
taxi_zones_df_renamed = taxi_zones_df \
    .withColumnRenamed("Borough", "pickup_borough") \
    .withColumnRenamed("Zone", "pickup_zone") \
    .withColumnRenamed("service_zone", "pickup_service_zone")

# COMMAND ----------

# Join on PULocationID to get pickup information
taxi_df = taxi_df.join(
    taxi_zones_df_renamed,
    taxi_df.PULocationID == taxi_zones_df_renamed.LocationID,
    "left"
).drop(taxi_zones_df_renamed.LocationID)

# COMMAND ----------

# Renaming columns again for dropoff information
taxi_zones_df_renamed = taxi_zones_df \
    .withColumnRenamed("Borough", "dropoff_borough") \
    .withColumnRenamed("Zone", "dropoff_zone") \
    .withColumnRenamed("service_zone", "dropoff_service_zone")

# COMMAND ----------

# Join on DOLocationID to get dropoff information
taxi_df = taxi_df.join(
    taxi_zones_df_renamed,
    taxi_df.DOLocationID == taxi_zones_df_renamed.LocationID,
    "left"
).drop(taxi_zones_df_renamed.LocationID)

# COMMAND ----------

# Show the result
taxi_df.show(truncate=False)

# COMMAND ----------

# Check for Missing Values in each column
# Check for missing values
missing_values = taxi_df.select([((taxi_df[column].isNull()).cast("int")).alias(column) for column in taxi_df.columns]) \
                                .agg({column: 'sum' for column in taxi_df.columns})

display(missing_values.limit(5))


# COMMAND ----------

taxi_df.select("PULocationID", "DOLocationID").filter(col("PULocationID").isNull() | col("DOLocationID").isNull()).show()

# COMMAND ----------

# Get distinct pickup locations and rename for joining
pickup_locations = taxi_df.select("PULocationID").distinct().withColumnRenamed("PULocationID", "LocationID")

# Get distinct dropoff locations and rename for joining
dropoff_locations = taxi_df.select("DOLocationID").distinct().withColumnRenamed("DOLocationID", "LocationID")

# Find missing pickup locations
missing_pickup_locations = pickup_locations.join(location_lat_long_spark_df, on="LocationID", how="left_anti")

# Find missing dropoff locations
missing_dropoff_locations = dropoff_locations.join(location_lat_long_spark_df, on="LocationID", how="left_anti")

# COMMAND ----------

# Display the results
print("Missing Pickup Locations:")
missing_pickup_locations.show()

# COMMAND ----------

print("Missing Dropoff Locations:")
missing_dropoff_locations.show()


# COMMAND ----------

# Count distinct LocationID values in location_lat_long_spark_df
distinct_count = location_lat_long_spark_df.select("LocationID").distinct().count()
print(f"Distinct LocationID count: {distinct_count}")

# COMMAND ----------

# Display all distinct LocationID values in sorted order
display(location_lat_long_spark_df.select("LocationID").distinct().orderBy("LocationID"))

# COMMAND ----------

# Drop the geometry column if it's not needed
zones_cleaned = zones.drop(columns=['geometry'])
zone_df = spark.createDataFrame(zones_cleaned)

# COMMAND ----------

# Convert geometry to string representation (e.g., WKT format)
zones['geometry'] = zones['geometry'].apply(lambda geom: geom.wkt if geom else None)  # If using Shapely
zone_geometry_df = spark.createDataFrame(zones)

# COMMAND ----------

zone_df=zone_df[['zone','LocationID', 'latitude', 'longitude']]

# COMMAND ----------

display(zone_df)

# COMMAND ----------

# List of Location IDs to search for
location_ids_to_find = [57, 264, 265, 105]

# Filter zone_df for these Location IDs
zone_df_filtered = zone_df.filter(zone_df.LocationID.isin(location_ids_to_find))

# Show the results
zone_df_filtered.show(truncate=False)

# COMMAND ----------

# Specify the IDs to check
ids_to_check = [57, 264, 265, 105]

# Total number of rows in the DataFrame
total_count = taxi_df.count()

# Count occurrences of specified IDs in pulocationid and dulocationid
pulocationid_count = taxi_df.filter(taxi_df.PULocationID.isin(ids_to_check)).count()
dulocationid_count = taxi_df.filter(taxi_df.DOLocationID.isin(ids_to_check)).count()

# Calculate percentages
pulocationid_percentage = (pulocationid_count / total_count) * 100
dulocationid_percentage = (dulocationid_count / total_count) * 100

print(f"Percentage of specified IDs in pulocationid: {pulocationid_percentage:.2f}%")
print(f"Percentage of specified IDs in dulocationid: {dulocationid_percentage:.2f}%")

# COMMAND ----------

# Remove rows containing the specified IDs in pulocationid or dulocationid
taxi_df = taxi_df.filter(~taxi_df.PULocationID.isin(ids_to_check) & ~taxi_df.DOLocationID.isin(ids_to_check))

# Show the filtered DataFrame
taxi_df.show()

# COMMAND ----------

# Check for Missing Values in each column
# Check for missing values
missing_values = taxi_df.select([((taxi_df[column].isNull()).cast("int")).alias(column) for column in taxi_df.columns]) \
                                .agg({column: 'sum' for column in taxi_df.columns})

display(missing_values.limit(5))


# COMMAND ----------

# Check the Shape and Schema
# Get the number of rows and columns
num_rows = taxi_df.count()
num_cols = len(taxi_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

display(taxi_df)

# COMMAND ----------

# Define a more descriptive Delta Lake storage path
delta_path = "/dbfs/FileStore/tables/data_processed_lat_long/"
# Write the DataFrame to Delta format
taxi_df.write.format("delta").mode("overwrite").save(delta_path)

# COMMAND ----------


