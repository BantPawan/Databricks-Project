# Databricks notebook source
#  Loading Taxi Zone Lookup Data

# COMMAND ----------

from pyspark.sql import SparkSession

# Initialize Spark session (if not already created)
spark = SparkSession.builder.appName("NYC Taxi Zone Lookup").getOrCreate()

# COMMAND ----------

# Define the path for the taxi zone lookup file
file_path = "dbfs:/FileStore/tables/taxi_zone_lookup.csv"

# COMMAND ----------

 # Load the taxi zone lookup table
taxi_zones_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Display the schema
taxi_zones_df.printSchema()

# Display the first 5 rows
display(taxi_zones_df)

# COMMAND ----------

# Check the Shape and Schema
# Get the number of rows and columns
num_rows = taxi_zones_df.count()
num_cols = len(taxi_zones_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

# Check for Duplicates
# Count the number of duplicate rows
duplicates_count = taxi_zones_df.count() - taxi_zones_df.distinct().count()
print(f"Number of duplicate rows: {duplicates_count}")

# COMMAND ----------

# Check for Missing Values in each column
# Check for missing values
missing_values = taxi_zones_df.select([((taxi_zones_df[column].isNull()).cast("int")).alias(column) for column in taxi_zones_df.columns]) \
                                .agg({column: 'sum' for column in taxi_zones_df.columns})

display(missing_values.limit(5))

# COMMAND ----------

# Get value counts for each column
for column in taxi_zones_df.columns:
    print(f"Value counts for {column}:")
    display(taxi_zones_df.groupBy(column).count())


# COMMAND ----------

from pyspark.sql.functions import when, col

# Replace "N/A" with null across all columns
for column in taxi_zones_df.columns:
    taxi_zones_df = taxi_zones_df.withColumn(
        column, when(col(column) == "N/A", None).otherwise(col(column))
    )


# COMMAND ----------

# Check for Missing Values in each column
# Check for missing values
missing_values = taxi_zones_df.select([((taxi_zones_df[column].isNull()).cast("int")).alias(column) for column in taxi_zones_df.columns]) \
                                .agg({column: 'sum' for column in taxi_zones_df.columns})

display(missing_values.limit(5))

# COMMAND ----------

# Remove rows with LocationID 264 and 265
taxi_zones_df = taxi_zones_df.filter(~col("LocationID").isin([264, 265]))

# COMMAND ----------

from pyspark.sql.functions import col

# Filter rows where any column is null
null_rows_df = taxi_zones_df.filter(
  (col("Borough").isNull()) | (col("Zone").isNull()) | (col("LocationID").isNull()) | (col("service_zone").isNull())
)

# Show the rows with null values
null_rows_df.show(truncate=False)

# COMMAND ----------

# Check for Missing Values in each column
# Check for missing values
missing_values = taxi_zones_df.select([((taxi_zones_df[column].isNull()).cast("int")).alias(column) for column in taxi_zones_df.columns]) \
                                .agg({column: 'sum' for column in taxi_zones_df.columns})

display(missing_values.limit(5))

# COMMAND ----------

# Define a more descriptive Delta Lake storage path
delta_path = "/dbfs/FileStore/tables/data_processed_taxi_zones"
# Write the DataFrame to Delta format
taxi_zones_df.write.format("delta").mode("overwrite").save(delta_path)
