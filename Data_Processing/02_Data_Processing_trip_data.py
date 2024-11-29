# Databricks notebook source
# Load the Delta table
data_processing_df = spark.read.format("delta").load("dbfs:/FileStore/tables/nyc_taxi_data_delta/")

# Show the first few rows to verify
display(data_processing_df.limit(5))

# COMMAND ----------

# Check the Shape and Schema
# Get the number of rows and columns
num_rows = data_processing_df.count()
num_cols = len(data_processing_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# Get DataFrame information
data_processing_df.printSchema()

# COMMAND ----------

# Check for Duplicates
# Count the number of duplicate rows
duplicates_count = data_processing_df.count() - data_processing_df.distinct().count()
print(f"Number of duplicate rows: {duplicates_count}")

# COMMAND ----------

# Check for Missing Values in each column
# Check for missing values
missing_values = data_processing_df.select([((data_processing_df[column].isNull()).cast("int")).alias(column) for column in data_processing_df.columns]) \
                                .agg({column: 'sum' for column in data_processing_df.columns})

display(missing_values.limit(5))


# COMMAND ----------

# Get value counts for each column
for column in data_processing_df.columns:
    print(f"Value counts for {column}:")
    display(data_processing_df.groupBy(column).count())

# COMMAND ----------

# Function to display value counts for a specific column using display()
def display_value_counts(dataframe, column):
    print(f"Value counts for {column}:")
    value_counts = dataframe.groupBy(column).count().orderBy("count", ascending=False)
    display(value_counts)


# COMMAND ----------

# Example usage for a specific column
display_value_counts(data_processing_df, 'month')

# COMMAND ----------

from pyspark.sql.functions import regexp_extract, col
from pyspark.sql.types import IntegerType

# Assuming 'month' is the name of the column in data_processing_df that contains values like '10.parq'
data_processing_df = data_processing_df.withColumn("Month_Num", regexp_extract("month", r"(\d{2})", 1).cast(IntegerType()))

# Drop the original 'month' column if it's not needed
data_processing_df = data_processing_df.drop("month")

# Display the updated DataFrame
display(data_processing_df)

# COMMAND ----------

from pyspark.sql.functions import hour, dayofweek, month, weekofyear

data_processing_df = data_processing_df.withColumn("pickup_month", month("tpep_pickup_datetime"))

# COMMAND ----------

# Extract features from dropoff datetime
data_processing_df = data_processing_df.withColumn("dropoff_hour", hour("tpep_dropoff_datetime"))
data_processing_df = data_processing_df.withColumn("dropoff_day_of_week", dayofweek("tpep_dropoff_datetime"))
data_processing_df = data_processing_df.withColumn("dropoff_month", month("tpep_dropoff_datetime"))
data_processing_df = data_processing_df.withColumn("dropoff_week_of_year", weekofyear("tpep_dropoff_datetime"))

# COMMAND ----------

display(data_processing_df.limit(5))

# COMMAND ----------

# Remove duplicate rows across all columns
data_processing_df = data_processing_df.dropDuplicates()

# COMMAND ----------

# Example usage for a specific column
display_value_counts(data_processing_df, 'passenger_count')

# COMMAND ----------

# Filter rows with null values in passenger_count
null_passenger_count_df = data_processing_df.filter(data_processing_df.passenger_count.isNull())

# Show the rows with null passenger_count
display(null_passenger_count_df)

# COMMAND ----------

from pyspark.sql import functions as F

# Step 1: Count total rows in the DataFrame
total_rows = data_processing_df.count()

# Step 2: Count rows with null values in passenger_count
null_count = data_processing_df.filter(data_processing_df.passenger_count.isNull()).count()

# Step 3: Calculate the percentage of null values
null_percentage = (null_count / total_rows) * 100

print(f"Total rows: {total_rows}")
print(f"Null count in passenger_count: {null_count}")
print(f"Percentage of null values in passenger_count: {null_percentage:.2f}%")

# COMMAND ----------

# Drop rows with any missing values
data_processing_df = data_processing_df.dropna()

# COMMAND ----------

# Get the number of rows and columns
num_rows = data_processing_df.count()
num_cols = len(data_processing_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

from pyspark.sql.functions import date_format, col

# Extract date from the pickup datetime
data_processing_df = data_processing_df.withColumn("pickup_date", date_format("tpep_pickup_datetime", "yyyy-MM-dd"))

# COMMAND ----------

# Extract time from the pickup datetime
data_processing_df = data_processing_df.withColumn("pickup_time", date_format("tpep_pickup_datetime", "HH:mm:ss"))

# COMMAND ----------

# updated DataFrame with new columns
data_processing_df.select("tpep_pickup_datetime", "pickup_date", "pickup_time").show(truncate=False)

# COMMAND ----------

# Group by pickup_time and count occurrences
peak_hours = data_processing_df.groupBy("pickup_time").count().orderBy("count", ascending=False)

# COMMAND ----------

# Show peak hours
display(peak_hours)

# COMMAND ----------

# Extract date from the dropoff datetime
data_processing_df = data_processing_df.withColumn("dropoff_date", date_format("tpep_dropoff_datetime", "yyyy-MM-dd"))

# COMMAND ----------

# Extract time from the dropoff datetime
data_processing_df = data_processing_df.withColumn("dropoff_time", date_format("tpep_dropoff_datetime", "HH:mm:ss"))

# COMMAND ----------

# updated DataFrame with new columns
data_processing_df.select("tpep_dropoff_datetime", "dropoff_date", "dropoff_time").show(truncate=False)

# COMMAND ----------

from pyspark.sql.functions import year

data_processing_df = data_processing_df.filter(year("pickup_date") == 2023)

data_processing_df = data_processing_df.filter(year("dropoff_date") == 2023)

data_processing_df.show()

# COMMAND ----------

# Peak Pickup Analysis
peak_pickup_hours = data_processing_df.groupBy("pickup_time").count().orderBy("count", ascending=False)
display(peak_pickup_hours)

# COMMAND ----------

# Peak Drop-off Analysis
peak_dropoff_hours = data_processing_df.groupBy("dropoff_time").count().orderBy("count", ascending=False)
display(peak_dropoff_hours)

# COMMAND ----------

# Daily Pickup Analysis
daily_pickups = data_processing_df.groupBy("pickup_date").count().orderBy("pickup_date")
display(daily_pickups)

# COMMAND ----------

# Daily Drop-off Analysis
daily_dropoffs = data_processing_df.groupBy("dropoff_date").count().orderBy("dropoff_date")
display(daily_dropoffs)


# COMMAND ----------

# Renaming columns for better clarity
data_processing_df = data_processing_df \
    .withColumnRenamed("tpep_pickup_datetime", "pickup_datetime") \
    .withColumnRenamed("tpep_dropoff_datetime", "dropoff_datetime")

# COMMAND ----------

data_processing_df.printSchema()

# COMMAND ----------

display(data_processing_df)

# COMMAND ----------

num_rows = data_processing_df.count()
num_cols = len(data_processing_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

# Define a more descriptive Delta Lake storage path
delta_path = "/dbfs/FileStore/tables/data_processed_trip_data/"
# Write the DataFrame to Delta format
data_processing_df.write.format("delta").mode("overwrite").save(delta_path)
