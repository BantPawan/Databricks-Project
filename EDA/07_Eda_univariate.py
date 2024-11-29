# Databricks notebook source
display(dbutils.fs.ls("/dbfs/FileStore/tables/"))

# COMMAND ----------

# Import necessary libraries
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Fare Prediction") \
    .getOrCreate()

# COMMAND ----------

# Load Delta table with PySpark
taxi_df = spark.read.format("delta").load("dbfs:/dbfs/FileStore/tables/cleaned_nyc_taxi_fare/")

# COMMAND ----------

display(taxi_df.limit(5))

# COMMAND ----------

taxi_df.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F

# Define the columns and the rounding precision
columns = {
    'trip_distance': 2,
    'fare_amount': 2,
    'tolls_amount': 2,
    'improvement_surcharge': 2,
    'total_amount': 2,
    'trip_duration': 2,
    'distance_km': 2,
    'pickup_latitude': 5,
    'pickup_longitude': 5,
    'dropoff_latitude': 5,
    'dropoff_longitude': 5,
    'temp': 1
}

# Apply rounding
for column, precision in columns.items():
    taxi_df = taxi_df.withColumn(column, F.format_number(F.col(column), precision).cast("double"))

# Show to verify
taxi_df.show()

# COMMAND ----------

display(taxi_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Fare Amount

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 1.	Fare Amount Distribution

# COMMAND ----------

# Basic descriptive stats for fare_amount
taxi_df.select("fare_amount").describe().show()

# COMMAND ----------

from pyspark.sql.functions import col, hour, dayofweek, month, year, mean
import matplotlib.pyplot as plt
import seaborn as sns

# sns.histplot(fare_data, kde=True)

# Convert fare amount to Pandas for plotting
fare_df = taxi_df.select("fare_amount").toPandas()

plt.figure(figsize=(10, 6))
plt.hist(fare_df['fare_amount'], bins=50, color='skyblue')
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Fare Amounts")
plt.show()

# COMMAND ----------

# Get row and column count
row_count = taxi_df.count()
column_count = len(taxi_df.columns)
print(f"Rows: {row_count}, Columns: {column_count}")

# COMMAND ----------

# Check for Duplicates
# Count the number of duplicate rows
duplicates_count = taxi_df.count() - taxi_df.distinct().count()
print(f"Number of duplicate rows: {duplicates_count}")

# COMMAND ----------

taxi_df = taxi_df.dropDuplicates()

# COMMAND ----------

# Summary Statistics for 'fare_amount'
taxi_df.describe("fare_amount").show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'fare_amount' column to a Pandas DataFrame or Series
fare_amount = taxi_df.select("fare_amount").toPandas()["fare_amount"]

# Assuming 'fare_amount' column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(fare_amount, bins=50, kde=True)
plt.title("Distribution of Fare Amounts")

plt.subplot(1, 2, 2)
sns.boxplot(x=fare_amount)
plt.title("Box Plot of Fare Amounts")
plt.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Filter out rows where fare_amount is less than or equal to zero
taxi_df = taxi_df.filter(col("fare_amount") > 0)

# COMMAND ----------

from pyspark.sql import functions as F

# Calculate Q1 and Q3
q1, q3 = taxi_df.approxQuantile("fare_amount", [0.25, 0.75], 0.05)
iqr = q3 - q1

# Define lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# COMMAND ----------

# Filter for outliers based on updated bounds
outliers = taxi_df.filter((col("fare_amount") < lower_bound) | (col("fare_amount") > upper_bound))
outliers.select("fare_amount").describe().show()

# COMMAND ----------

from pyspark.sql.functions import when

# Cap fare_amount values at the upper bound
taxi_df = taxi_df.withColumn("fare_amount", when(col("fare_amount") > upper_bound, upper_bound).otherwise(col("fare_amount")))

# COMMAND ----------

# Summary Statistics for 'fare_amount'
taxi_df.describe("fare_amount").show()

# COMMAND ----------

from pyspark.sql import functions as F

# Calculate skewness and kurtosis for the fare_amount column
fare_stats = taxi_df.select(
    F.skewness("fare_amount").alias("skewness"),
    F.kurtosis("fare_amount").alias("kurtosis")
)

# Display the results
fare_stats.show()

# COMMAND ----------

# MAGIC %md
# MAGIC A positive skewness (like 0.8633) suggests that the tail on the right side of the distribution (higher fare amounts) is longer or fatter than the left side. This means that there are more low fares and a few extremely high fares pulling the mean to the right.

# COMMAND ----------

# MAGIC %md
# MAGIC A negative kurtosis (like -0.4961) indicates that the distribution has lighter tails and a flatter peak than a normal distribution. In this case, it suggests that there are fewer extreme outliers compared to a normal distribution.

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'fare_amount' column to a Pandas DataFrame or Series
fare_amount = taxi_df.select("fare_amount").toPandas()["fare_amount"]

# Assuming 'fare_amount' column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(fare_amount, bins=50, kde=True)
plt.title("Distribution of Fare Amounts")

plt.subplot(1, 2, 2)
sns.boxplot(x=fare_amount)
plt.title("Box Plot of Fare Amounts")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Total Amount

# COMMAND ----------

# Basic descriptive stats for total_amount
taxi_df.select("total_amount").describe().show()

# COMMAND ----------

from pyspark.sql.functions import col, hour, dayofweek, month, year, mean
import matplotlib.pyplot as plt
import seaborn as sns

# sns.histplot(fare_data, kde=True)

# Convert fare amount to Pandas for plotting
amount_df = taxi_df.select("total_amount").toPandas()

plt.figure(figsize=(10, 6))
plt.hist(amount_df['total_amount'], bins=50, color='skyblue')
plt.xlabel("Total Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Total Amount")
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'Total Amount' column to a Pandas DataFrame or Series
amount_df = taxi_df.select("total_amount").toPandas()["total_amount"]

# Assuming Total Amount column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(amount_df, bins=50, kde=True)
plt.title("Distribution of total_amount")

plt.subplot(1, 2, 2)
sns.boxplot(x=amount_df)
plt.title("Box Plot of Total Amount")
plt.show()


# COMMAND ----------

from pyspark.sql.functions import col

# Filter out rows where total_amount is less than or equal to zero
taxi_df = taxi_df.filter(col("total_amount") > 0)

# COMMAND ----------

from pyspark.sql import functions as F

# Calculate Q1 and Q3
q1, q3 = taxi_df.approxQuantile("total_amount", [0.25, 0.75], 0.05)
iqr = q3 - q1

# Define lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# COMMAND ----------

from pyspark.sql.functions import when

# Cap total amount values at the upper bound
taxi_df = taxi_df.withColumn("total_amount", when(col("total_amount") > upper_bound, upper_bound).otherwise(col("total_amount")))

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'total_amount' column to a Pandas DataFrame or Series
total_amount = taxi_df.select("total_amount").toPandas()["total_amount"]

# Assuming 'total_amount' column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(total_amount, bins=50, kde=True)
plt.title("Distribution of total_amount")

plt.subplot(1, 2, 2)
sns.boxplot(x=fare_amount)
plt.title("Box Plot of total_amount")
plt.show()

# COMMAND ----------

# Summary Statistics for 'total_amount'
taxi_df.describe("total_amount").show()

# COMMAND ----------

from pyspark.sql import functions as F

# Calculate skewness and kurtosis for the total_amount column
total_amount_stats = taxi_df.select(
    F.skewness("total_amount").alias("skewness"),
    F.kurtosis("total_amount").alias("kurtosis")
)

# Display the results
total_amount_stats.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Passenger Count

# COMMAND ----------

# Distribution of passenger_count
passenger_dist = taxi_df.groupBy("passenger_count").count().orderBy("count", ascending=False)
passenger_dist.show()

# COMMAND ----------

# Bar plot for passenger count
passenger_count_data = taxi_df.groupBy("passenger_count").count().orderBy("passenger_count").collect()
x = [row['passenger_count'] for row in passenger_count_data]
y = [row['count'] for row in passenger_count_data]

plt.bar(x, y, color="purple")
plt.title("Passenger Count Distribution")
plt.xlabel("Passenger Count")
plt.ylabel("Frequency")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Most Common Values: The vast majority of records have 1-2 passengers, which seems reasonable for taxi rides in a city. Rows with values from 1 to 6 appear commonly and represent realistic group sizes for taxi usage.
# MAGIC
# MAGIC Zero Passengers: There are 569,086 records with a passenger_count of 0, which seems unrealistic for taxi rides. These could be errors or placeholder values. It might be best to remove these rows, as they likely don’t contribute meaningful information for fare prediction
# MAGIC
# MAGIC Outliers (Values of 7 to 9): Rows with passenger counts from 7 to 9 are very rare (a total of 263 occurrences in 36 million rows). These could be data entry errors, or they could represent unusual cases (e.g., shared rides). Given the low frequency, removing them should have a minimal impact, but you could keep them if you want a model that handles unusual cases.

# COMMAND ----------

# Filter out rows with passenger_count of 0 or greater than 6
taxi_df = taxi_df.filter((taxi_df["passenger_count"] > 0) & (taxi_df["passenger_count"] <= 6))

# Verify the distribution after filtering
taxi_df.groupBy("passenger_count").count().orderBy("passenger_count").show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'trip_distance' column to a Pandas DataFrame or Series
passenger_count_df = taxi_df.select("passenger_count").toPandas()["passenger_count"]

# Assuming Trip Distance column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
sns.boxplot(x=passenger_count_df)
plt.title("Box Plot of Passenger Count")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Payment Type

# COMMAND ----------

payment_mapping = {
    1: "Credit Card",
    2: "Cash",
    3: "No Charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided Trip"
}

# COMMAND ----------

# Flatten the dictionary items into a list of alternating keys and values
payment_type_expr = F.create_map([F.lit(x) for kv in payment_mapping.items() for x in kv])

# Use the map to create the new column
taxi_df = taxi_df.withColumn("payment_type_label", payment_type_expr[F.col("payment_type")])

# COMMAND ----------

payment_counts = taxi_df.groupBy("payment_type_label").count()

# COMMAND ----------

total_count = taxi_df.count()
payment_percentage = payment_counts.withColumn("percentage", (F.col("count") / total_count) * 100)

# COMMAND ----------

# Show the frequency and percentage distribution
payment_percentage.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Convert to Pandas DataFrame for visualization
payment_percentage_pd = payment_percentage.toPandas()

# Bar Plot
plt.figure(figsize=(10, 5))
plt.bar(payment_percentage_pd['payment_type_label'], payment_percentage_pd['count'], color='blue')
plt.title('Frequency Distribution of Payment Types')
plt.xlabel('Payment Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(payment_percentage_pd['percentage'], labels=payment_percentage_pd['payment_type_label'], autopct='%1.1f%%', startangle=140)
plt.title('Percentage Distribution of Payment Types')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Distribution of Trip Distance

# COMMAND ----------

# Summary Statistics for 'fare_amount'
taxi_df.describe("trip_distance").show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'trip_distance' column to a Pandas DataFrame or Series
trip_distance_df = taxi_df.select("trip_distance").toPandas()["trip_distance"]

# Assuming Trip Distance column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(trip_distance_df, bins=50, kde=True)
plt.title("Distribution of Trip Distance")

plt.subplot(1, 2, 2)
sns.boxplot(x=trip_distance_df)
plt.title("Box Plot of Trip Distance")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Zero values for trip_distance might indicate canceled trips, errors in data entry, or other anomalies. It’s essential to investigate these zero distances to determine if they are valid entries or should be excluded from analysis.

# COMMAND ----------

taxi_df = taxi_df.filter(taxi_df["trip_distance"] > 0)

# COMMAND ----------

from pyspark.sql import functions as F

# Calculate Q1 and Q3
q1, q3 = taxi_df.approxQuantile("trip_distance", [0.25, 0.75], 0.05)
iqr = q3 - q1

# Define lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# COMMAND ----------

# Filter for outliers based on updated bounds
outliers = taxi_df.filter((col("fare_amount") < lower_bound) | (col("trip_distance") > upper_bound))
outliers.select("trip_distance").describe().show()

# COMMAND ----------

from pyspark.sql.functions import when

# Cap trip_distance values at the upper bound
taxi_df = taxi_df.withColumn("trip_distance", when(col("trip_distance") > upper_bound, upper_bound).otherwise(col("trip_distance")))


# COMMAND ----------

from pyspark.sql import functions as F

# Calculate skewness and kurtosis for the trip_distance column
trip_distance_stats = taxi_df.select(
    F.skewness("trip_distance").alias("skewness"),
    F.kurtosis("trip_distance").alias("kurtosis")
)

# Display the results
trip_distance_stats.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'trip_distance' column to a Pandas DataFrame or Series
trip_distance_df = taxi_df.select("trip_distance").toPandas()["trip_distance"]

# Assuming Trip Distance column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(trip_distance_df, bins=50, kde=True)
plt.title("Distribution of Trip Distance")

plt.subplot(1, 2, 2)
sns.boxplot(x=trip_distance_df)
plt.title("Box Plot of Trip Distance")
plt.show()

# COMMAND ----------

# Summary Statistics for 'trip_distance'
taxi_df.describe("trip_distance").show()

# COMMAND ----------

# Frequency distribution for trip distance bins
frequency_bins = taxi_df.select(
    when(col("trip_distance") > 20, "Very High (>20 km)").when((col("trip_distance") >= 10) & (col("trip_distance") <= 20), "High (10-20 km)").when((col("trip_distance") >= 5) & (col("trip_distance") < 10), "Average (5-10 km)").when((col("trip_distance") > 1) & (col("trip_distance") < 5), "Low (1-5 km)").otherwise("Very Low (<1 km)").alias("distance_category")
).groupBy("distance_category").count()

frequency_bins.show()

# COMMAND ----------

from pyspark.sql import functions as F

# Define bin boundaries for trip_distance and fare_amount
trip_distance_bins = [2, 5, 10, 20]  # Example: 0-2 km, 2-5 km, 5-10 km, 10-20 km, 20+ km
fare_amount_bins = [10, 20, 50, 100]  # Example: $0-10, $10-20, $20-50, $50-100, $100+

# Function to create bins and calculate frequency for a given column
def create_frequency_bins(df, column_name, bins, bin_labels):
    # Start bin column using the first condition
    bin_col = F.when(F.col(column_name) < bins[0], bin_labels[0])
    
    # Iterate through bins to set conditions for each range
    for i in range(1, len(bins)):
        bin_col = bin_col.when((F.col(column_name) >= bins[i-1]) & (F.col(column_name) < bins[i]), bin_labels[i])
    
    # Add final bin for the last range (values greater than the last bin value)
    bin_col = bin_col.when(F.col(column_name) >= bins[-1], bin_labels[-1])

    # Add bin column to DataFrame
    df = df.withColumn(f"{column_name}_bin", bin_col)

    # Calculate frequency for each bin
    frequency_df = df.groupBy(f"{column_name}_bin").count().orderBy(F.col("count").desc())

    return frequency_df

# Define bin labels (must match the length of bins + 1)
trip_distance_labels = ["0-2 km", "2-5 km", "5-10 km", "10-20 km", "20+ km"]
fare_amount_labels = ["$0-10", "$10-20", "$20-50", "$50-100", "$100+"]

# Create frequency distributions
trip_distance_freq = create_frequency_bins(taxi_df, "trip_distance", trip_distance_bins, trip_distance_labels)
fare_amount_freq = create_frequency_bins(taxi_df, "fare_amount", fare_amount_bins, fare_amount_labels)

# Show the frequency tables
trip_distance_freq.show()
fare_amount_freq.show()

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

# Check for the .shp file in the extracted directory
if "taxi_zones.shp" in extracted_files:
    import geopandas as gpd
    # Load the shapefile
    zones = gpd.read_file(os.path.join(local_extract_path, "taxi_zones.shp"))
    
    # Convert to WGS 84 coordinate system if necessary
    zone_gdf = zones.to_crs(epsg=4326)
else:
    print("Error: 'taxi_zones.shp' not found in extracted files.")

# COMMAND ----------

from pyspark.sql import functions as F

# Count pickups by borough
pickup_borough_agg = (
    taxi_df.groupBy("pickup_borough")
    .agg(F.count("PULocationID").alias("pickup_count"))
)

pickup_borough_agg.show()

# COMMAND ----------

# Count dropoffs by borough
dropoff_borough_agg = (
    taxi_df.groupBy("dropoff_borough")
    .agg(F.count("DOLocationID").alias("dropoff_count"))
)

dropoff_borough_agg.show()


# COMMAND ----------

# Convert to Pandas DataFrames
pickup_borough_agg_pd = pickup_borough_agg.toPandas()
dropoff_borough_agg_pd = dropoff_borough_agg.toPandas()

# If you have a GeoDataFrame called zone_gdf, ensure it is in Pandas
zone_gdf_pd = zone_gdf.copy()  # Assuming zone_gdf is a GeoDataFrame

# COMMAND ----------

# Convert to Pandas DataFrame and rename columns for consistency
pickup_borough_agg_pd = pickup_borough_agg.toPandas().rename(columns={"pickup_borough": "borough"})
dropoff_borough_agg_pd = dropoff_borough_agg.toPandas().rename(columns={"dropoff_borough": "borough"})

# COMMAND ----------

# Merge pickup and dropoff counts with the original zone data for plotting
pickup_borough_merged = zone_gdf.merge(pickup_borough_agg_pd, on='borough', how='left')
dropoff_borough_merged = zone_gdf.merge(dropoff_borough_agg_pd, on='borough', how='left')

# COMMAND ----------

# Plot pickup and dropoff counts
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Plotting Pickup Counts
pickup_borough_merged.plot(column="pickup_count", cmap="Oranges", legend=True, ax=ax[0])
ax[0].set_title("NYC Pickup Counts by Borough")

# Plotting Dropoff Counts
dropoff_borough_merged.plot(column="dropoff_count", cmap="Blues", legend=True, ax=ax[1])
ax[1].set_title("NYC Dropoff Counts by Borough")

plt.show()


# COMMAND ----------

# Check the Shape and Schema
# Get the number of rows and columns
num_rows = taxi_df.count()
num_cols = len(taxi_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

# Define a more descriptive Delta Lake storage path
delta_path = "/dbfs/FileStore/tables/Eda_univariate_taxi_data/" 
# Write the DataFrame to Delta format
taxi_df.write.format("delta").mode("overwrite").save(delta_path)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
