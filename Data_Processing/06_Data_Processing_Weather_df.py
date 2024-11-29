# Databricks notebook source
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("NYC Taxi Fare Prediction").getOrCreate()

# COMMAND ----------

from pyspark.sql import functions as F
import requests
import pandas as pd

# Replace with your API Key from Visual Crossing
api_key = '6ZBRRE47XM5UHNVS7X7G53ZU5'
url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/NewYork/2023-01-01/2023-12-31?unitGroup=metric&key={api_key}&include=days"

# COMMAND ----------

# List all files in the specified DBFS directory
dbutils.fs.ls("dbfs:/dbfs/FileStore/tables/Data_Processing_Pulic_Holiday/")

# COMMAND ----------

# Load the Delta table
taxi_with_holidays_df = spark.read.format("delta").load("dbfs:/dbfs/FileStore/tables/Data_Processing_Pulic_Holiday/")

# Show the first few rows to verify
display(taxi_with_holidays_df.limit(5))

# COMMAND ----------

# Fetch weather data
response = requests.get(url)

if response.status_code == 200:
    weather_data = response.json()
    # Convert the JSON response to Pandas DataFrame
    weather_df = pd.json_normalize(weather_data['days'])
    weather_df = pd.DataFrame(weather_df[['datetime', 'temp']])  # Select only date and temperature

    # Convert to Spark DataFrame
    weather_spark_df = spark.createDataFrame(weather_df)
    # Rename and format the date column
    weather_spark_df = weather_spark_df.withColumnRenamed('datetime', 'weather_date')
    weather_spark_df = weather_spark_df.withColumn("weather_date", F.to_date("weather_date"))

    # Join the weather data with taxi data on date columns
    taxi_with_weather_df = taxi_with_holidays_df.join(
        weather_spark_df,
        taxi_with_holidays_df["pickup_date"] == weather_spark_df["weather_date"],
        how="left"
    )

    # Drop the duplicate date column from weather data if needed
    taxi_with_weather_df = taxi_with_weather_df.drop("weather_date")

    # Show the updated DataFrame with weather info
    taxi_with_weather_df.show()
else:
    print("Failed to fetch weather data. Status code:", response.status_code)

# COMMAND ----------

display(taxi_with_weather_df)

# COMMAND ----------

# Function to display value counts for a specific column using display()
def display_value_counts(dataframe, column):
    print(f"Value counts for {column}:")
    value_counts = dataframe.groupBy(column).count().orderBy("count", ascending=False)
    display(value_counts)

# Example usage for a specific column
display_value_counts(taxi_with_weather_df, 'temp')

# COMMAND ----------

num_rows = taxi_with_weather_df.count()
num_cols = len(taxi_with_weather_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")


# COMMAND ----------

# Check for Missing Values in each column
# Check for missing values
missing_values = taxi_with_weather_df.select([((taxi_with_weather_df[column].isNull()).cast("int")).alias(column) for column in taxi_with_weather_df.columns]) \
                                .agg({column: 'sum' for column in taxi_with_weather_df.columns})

display(missing_values.limit(5))

# COMMAND ----------

# Check for Duplicates
# Count the number of duplicate rows
duplicates_count = taxi_with_weather_df.count() - taxi_with_weather_df.distinct().count()
print(f"Number of duplicate rows: {duplicates_count}")

# COMMAND ----------

taxi_with_weather_df.show(5)

# COMMAND ----------

# Define a more descriptive Delta Lake storage path
delta_path = "/dbfs/FileStore/tables/cleaned_nyc_taxi_fare/"
# Write the DataFrame to Delta format
taxi_with_weather_df.write.format("delta").mode("overwrite").save(delta_path)
