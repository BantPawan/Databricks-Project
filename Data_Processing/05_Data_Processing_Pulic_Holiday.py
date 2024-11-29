# Databricks notebook source
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("NYC Taxi Fare Predi ction").getOrCreate()

# COMMAND ----------

from azureml.opendatasets import PublicHolidays
from datetime import datetime

# Define the start and end dates for the year 2023
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

# COMMAND ----------

# Fetch public holidays data
hol = PublicHolidays(start_date=start_date, end_date=end_date)
hol_df = hol.to_spark_dataframe()

# COMMAND ----------

# Display holidays
hol_df.show()

# COMMAND ----------

# removing this column since it has only null values
hol_df = hol_df.drop("isPaidTimeOff")

# COMMAND ----------

display(hol_df)

# COMMAND ----------

# Load the Delta table
taxi_df = spark.read.format("delta").load("/dbfs/FileStore/tables/data_processed_lat_long/")

# Show the first few rows to verify
display(taxi_df.limit(5))

# COMMAND ----------

from pyspark.sql import functions as F

# Step 1: Filter for U.S. holidays in 2023 in holiday_df
hol_df = hol_df.filter(
    (hol_df["countryRegionCode"] == "US") &
    (F.year(F.to_date("date")) == 2023)
)

# COMMAND ----------

# Step 2: Convert date columns to compatible format
taxi_df = taxi_df.withColumn("pickup_date", F.to_date("pickup_datetime"))
hol_df = hol_df.withColumn("holiday_date", F.to_date("date"))

# COMMAND ----------

# Step 3: Perform a left join to add holiday information to taxi_df based on pickup date
taxi_with_holidays_df = taxi_df.join(
    hol_df,
    taxi_df["pickup_date"] == hol_df["holiday_date"],
    how="left"
)

# COMMAND ----------

# Step 4: Add a holiday indicator (1 if it's a holiday, otherwise 0)
taxi_with_holidays_df = taxi_with_holidays_df.withColumn(
    "is_holiday",
    F.when(F.col("holidayName").isNotNull(), 1).otherwise(0)
)

# COMMAND ----------

# Fill null values with a default value for holidayName and normalizeHolidayName
taxi_with_holidays_df = taxi_with_holidays_df.fillna({
    "holidayName": "No Holiday",
    "normalizeHolidayName": "No Holiday"
})

# COMMAND ----------

display(taxi_with_holidays_df)

# COMMAND ----------

# Get unique values in the 'holidayName' column
hol_df.select("holidayName").distinct().show(truncate=False)

# COMMAND ----------

# Get unique values in the 'normalizeHolidayName' column
hol_df.select("normalizeHolidayName").distinct().show(truncate=False)

# COMMAND ----------

taxi_with_holidays_df = taxi_with_holidays_df.drop("countryOrRegion", "countryRegionCode", "date", "holiday_date")

# COMMAND ----------

display(taxi_with_holidays_df.limit(5))

# COMMAND ----------

# Check the Shape and Schema
# Get the number of rows and columns
num_rows = taxi_with_holidays_df.count()
num_cols = len(taxi_with_holidays_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")


# COMMAND ----------

# Check for Missing Values in each column
# Check for missing values
missing_values = taxi_with_holidays_df.select([((taxi_with_holidays_df[column].isNull()).cast("int")).alias(column) for column in taxi_with_holidays_df.columns]) \
                                .agg({column: 'sum' for column in taxi_with_holidays_df.columns})

display(missing_values.limit(5))


# COMMAND ----------

# Get value counts for each column
for column in taxi_with_holidays_df.columns:
    print(f"Value counts for {column}:")
    display(taxi_with_holidays_df.groupBy(column).count())

# COMMAND ----------

# Check for Duplicates
# Count the number of duplicate rows
duplicates_count = taxi_with_holidays_df.count() - taxi_with_holidays_df.distinct().count()
print(f"Number of duplicate rows: {duplicates_count}")


# COMMAND ----------

# Define a more descriptive Delta Lake storage path
delta_path = "/dbfs/FileStore/tables/Data_Processing_Pulic_Holiday/"
# Write the DataFrame to Delta format
taxi_with_holidays_df.write.format("delta").mode("overwrite").save(delta_path)
