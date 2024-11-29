# Databricks notebook source
# DBTITLE 1,v
display(dbutils.fs.ls("/dbfs/FileStore/tables/"))

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("OptimizedRF") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

# COMMAND ----------

# Load Delta table with PySpark
taxi_df = spark.read.format("delta").load("dbfs:/dbfs/FileStore/tables/cleaned_nyc_taxi_fare/")

# COMMAND ----------

from pyspark.sql.functions import col
# Summary Statistics for 'trip_distance'
taxi_df.describe("trip_distance").show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'trip_distance' column to a Pandas DataFrame or Series
trip_distance_df = taxi_df.select("trip_distance").toPandas()["trip_distance"]

# Assuming 'trip_distance' column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(trip_distance_df, bins=50, kde=True)
plt.title("Distribution of Trip Distance")

plt.subplot(1, 2, 2)
sns.boxplot(x=trip_distance_df)
plt.title("Box Plot of Trip Distance")
plt.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Filter out rows where trip_duration is less than or equal to zero
taxi_df = taxi_df.filter(col("trip_distance") > 0)

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
outliers = taxi_df.filter((col("trip_distance") < lower_bound) | (col("trip_distance") > upper_bound))
outliers.select("trip_distance").describe().show()

# COMMAND ----------

from pyspark.sql.functions import when

# Cap trip_distance values at the upper bound
taxi_df = taxi_df.withColumn("trip_distance", when(col("trip_distance") > upper_bound, upper_bound).otherwise(col("trip_distance")))

# COMMAND ----------

# Summary Statistics for 'trip_duration'
taxi_df.describe("trip_distance").show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
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

# Assuming 'trip_distance' column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(trip_distance_df, bins=50, kde=True)
plt.title("Distribution of Trip Distance")

plt.subplot(1, 2, 2)
sns.boxplot(x=trip_distance_df)
plt.title("Box Plot of Trip Distance")
plt.show()

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import col
# Summary Statistics for 'trip_duration'
taxi_df.describe("trip_duration").show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'trip_duration' column to a Pandas DataFrame or Series
trip_duration_df = taxi_df.select("trip_duration").toPandas()["trip_duration"]

# Assuming 'trip_duration' column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(trip_duration_df, bins=50, kde=True)
plt.title("Distribution of Trip Duration")

plt.subplot(1, 2, 2)
sns.boxplot(x=trip_duration_df)
plt.title("Box Plot of Trip Duration")
plt.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Filter out rows where trip_duration is less than or equal to zero
taxi_df = taxi_df.filter(col("trip_duration") > 0)

# COMMAND ----------

from pyspark.sql import functions as F

# Calculate Q1 and Q3
q1, q3 = taxi_df.approxQuantile("trip_duration", [0.25, 0.75], 0.05)
iqr = q3 - q1

# Define lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# COMMAND ----------

# Filter for outliers based on updated bounds
outliers = taxi_df.filter((col("trip_duration") < lower_bound) | (col("trip_duration") > upper_bound))
outliers.select("trip_duration").describe().show()

# COMMAND ----------

from pyspark.sql.functions import when

# Cap trip_duration values at the upper bound
taxi_df = taxi_df.withColumn("trip_duration", when(col("trip_duration") > upper_bound, upper_bound).otherwise(col("trip_duration")))

# COMMAND ----------

# Summary Statistics for 'trip_duration'
taxi_df.describe("trip_duration").show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# Calculate skewness and kurtosis for the trip_duration column
fare_stats = taxi_df.select(
    F.skewness("trip_duration").alias("skewness"),
    F.kurtosis("trip_duration").alias("kurtosis")
)

# Display the results
fare_stats.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the 'trip_duration' column to a Pandas DataFrame or Series
trip_duration_df = taxi_df.select("trip_duration").toPandas()["trip_duration"]

# Assuming 'trip_duration' column has been converted to Pandas Series for plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(trip_duration_df, bins=50, kde=True)
plt.title("Distribution of Trip Duration")

plt.subplot(1, 2, 2)
sns.boxplot(x=trip_duration_df )
plt.title("Box Plot of Trip Duration")
plt.show()

# COMMAND ----------

from pyspark.sql.functions import col
import matplotlib.pyplot as plt

# Define city border coordinates
city_long_border = (-74.1,73.8)
city_lat_border = (40.5, 40.9)

# Filter data for valid latitude and longitude values
taxi_df = taxi_df.filter(
    (col("pickup_longitude") >= city_long_border[0]) & 
    (col("pickup_longitude") <= city_long_border[1]) &
    (col("pickup_latitude") >= city_lat_border[0]) & 
    (col("pickup_latitude") <= city_lat_border[1]) &
    (col("dropoff_longitude") >= city_long_border[0]) & 
    (col("dropoff_longitude") <= city_long_border[1]) &
    (col("dropoff_latitude") >= city_lat_border[0]) & 
    (col("dropoff_latitude") <= city_lat_border[1]))

# COMMAND ----------

# import folium
# from folium.plugins import MarkerCluster

# # Initialize the map centered on New York City
# m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

# # Add a MarkerCluster layer
# marker_cluster = MarkerCluster().add_to(m)

# # Take a 1% random sample of the data and select only the required columns
# sampled_taxi_df = taxi_df.sample(fraction=0.01).select("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude").collect()

# # Add markers for each pickup and dropoff location in the sampled data
# for row in sampled_taxi_df:
#     # Add a marker for the pickup location
#     folium.Marker([row['pickup_latitude'], row['pickup_longitude']], 
#                   popup="Pickup Location").add_to(marker_cluster)
    
#     # Add a marker for the dropoff location
#     folium.Marker([row['dropoff_latitude'], row['dropoff_longitude']], 
#                   popup="Dropoff Location").add_to(marker_cluster)

# # Display the map
# m


# COMMAND ----------

display(taxi_df)

# COMMAND ----------

# Function to display value counts for a specific column using display()
def display_value_counts(dataframe, column):
    print(f"Value counts for {column}:")
    value_counts = dataframe.groupBy(column).count().orderBy("count", ascending=False)
    display(value_counts)

# Example usage for a specific column
display_value_counts(taxi_df, 'pickup_service_zone')

# COMMAND ----------



# COMMAND ----------

# Example usage for a specific column
display_value_counts(taxi_df, 'pickup_zone')

# COMMAND ----------

# MAGIC %md
# MAGIC | Column               | Type            | Purpose                                                    |
# MAGIC |----------------------|-----------------|------------------------------------------------------------|
# MAGIC | `DOLocationID`       | Categorical     | Destination location                                       |
# MAGIC | `PULocationID`       | Categorical     | Pickup location                                            |
# MAGIC | `passenger_count`    | Numerical       | Number of passengers                                       |
# MAGIC | `trip_distance`      | Numerical       | Trip distance                                              |
# MAGIC | `total_amount`       | Numerical       | Target variable (taxi fare)                                |
# MAGIC | `trip_duration`      | Numerical       | Trip duration                                              |
# MAGIC | `pickup_hour`        | Categorical     | Hour of pickup                                             |
# MAGIC | `pickup_day_of_week` | Categorical     | Day of the week (pickup)                                   |
# MAGIC | `pickup_month`       | Categorical     | Month of pickup                                            |
# MAGIC | `distance_km`        | Numerical       | Calculated trip distance in km                             |
# MAGIC | `pickup_borough`     | Categorical     | Pickup borough                                             |
# MAGIC | `dropoff_borough`    | Categorical     | Drop-off borough                                           |
# MAGIC | `is_holiday`         | Categorical     | Indicator for holidays                                     |
# MAGIC | `temp`               | Numerical       | Temperature (optional)                                     |

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Bins for Distance

# COMMAND ----------

# Binning trip_distance into categories: Short, Medium, Long
taxi_df = taxi_df.withColumn(
    "distance_bin", 
    when(col("trip_distance") <= 2, "Short")
    .when((col("trip_distance") > 2) & (col("trip_distance") <= 5), "Medium")
    .otherwise("Long")
)

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt

distance_distribution = (
    taxi_df.groupBy("distance_bin")
    .agg(F.count("*").alias("count"))
)

total_count = distance_distribution.agg(F.sum("count")).collect()[0][0]
distance_distribution = distance_distribution.withColumn("percentage", (F.col("count") / total_count) * 100)

distance_distribution_pd = distance_distribution.toPandas()

plt.figure(figsize=(8, 6))
plt.pie(distance_distribution_pd["percentage"], labels=distance_distribution_pd["distance_bin"], autopct='%1.1f%%')
plt.title("Distribution of Trip Distances")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ####  Bins for  Temperature

# COMMAND ----------


# Binning temperature into ranges: Cold, Cool, Warm, Hot
taxi_df = taxi_df.withColumn(
    "temp_bin", 
    when(col("temp") <= 10, "Cold")
    .when((col("temp") > 10) & (col("temp") <= 20), "Cool")
    .when((col("temp") > 20) & (col("temp") <= 30), "Warm")
    .otherwise("Hot")
)

# COMMAND ----------

temp_distribution = (
    taxi_df.groupBy("temp_bin")
    .agg(F.count("*").alias("count"))
)

total_count = temp_distribution.agg(F.sum("count")).collect()[0][0]
temp_distribution = temp_distribution.withColumn("percentage", (F.col("count") / total_count) * 100)

temp_distribution_pd = temp_distribution.toPandas()

plt.figure(figsize=(8, 6))
plt.pie(temp_distribution_pd["percentage"], labels=temp_distribution_pd["temp_bin"], autopct='%1.1f%%')
plt.title("Distribution of Temperature Bins")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Binning Time of Day

# COMMAND ----------

# Binning pickup_hour into time of day categories
taxi_df = taxi_df.withColumn(
    "time_of_day_bin",
    when((col("pickup_hour") >= 0) & (col("pickup_hour") < 6), "Late Night")
    .when((col("pickup_hour") >= 6) & (col("pickup_hour") < 12), "Morning")
    .when((col("pickup_hour") >= 12) & (col("pickup_hour") < 18), "Afternoon")
    .otherwise("Evening")
)

# COMMAND ----------

time_of_day_distribution = (
    taxi_df.groupBy("time_of_day_bin")
    .agg(F.count("*").alias("count"))
)

total_count = time_of_day_distribution.agg(F.sum("count")).collect()[0][0]
time_of_day_distribution = time_of_day_distribution.withColumn("percentage", (F.col("count") / total_count) * 100)

time_of_day_distribution_pd = time_of_day_distribution.toPandas()

plt.figure(figsize=(8, 6))
plt.pie(time_of_day_distribution_pd["percentage"], labels=time_of_day_distribution_pd["time_of_day_bin"], autopct='%1.1f%%')
plt.title("Distribution of Trips by Time of Day")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Binning Time of Day

# COMMAND ----------

# Binning pickup_hour into time of day categories
taxi_df = taxi_df.withColumn(
    "time_of_day_bin",
    when((col("pickup_hour") >= 0) & (col("pickup_hour") < 6), "Late Night")
    .when((col("pickup_hour") >= 6) & (col("pickup_hour") < 12), "Morning")
    .when((col("pickup_hour") >= 12) & (col("pickup_hour") < 18), "Afternoon")
    .otherwise("Evening")
)

# COMMAND ----------

time_of_day_distribution = (
    taxi_df.groupBy("time_of_day_bin")
    .agg(F.count("*").alias("count"))
)

total_count = time_of_day_distribution.agg(F.sum("count")).collect()[0][0]
time_of_day_distribution = time_of_day_distribution.withColumn("percentage", (F.col("count") / total_count) * 100)

time_of_day_distribution_pd = time_of_day_distribution.toPandas()

plt.figure(figsize=(8, 6))
plt.pie(time_of_day_distribution_pd["percentage"], labels=time_of_day_distribution_pd["time_of_day_bin"], autopct='%1.1f%%')
plt.title("Distribution of Trips by Time of Day")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Binning Date into Season

# COMMAND ----------

from pyspark.sql.functions import month, when

# Create season column based on the pickup month
taxi_df = taxi_df.withColumn(
    "season",
    when(month("pickup_datetime").isin([12, 1, 2]), "Winter")
    .when(month("pickup_datetime").isin([3, 4, 5]), "Spring")
    .when(month("pickup_datetime").isin([6, 7, 8]), "Summer")
    .otherwise("Fall")
)


# COMMAND ----------

season_distribution = (
    taxi_df.groupBy("season")
    .agg(F.count("*").alias("count"))
)

total_count = season_distribution.agg(F.sum("count")).collect()[0][0]
season_distribution = season_distribution.withColumn("percentage", (F.col("count") / total_count) * 100)

season_distribution_pd = season_distribution.toPandas()

plt.figure(figsize=(8, 6))
plt.pie(season_distribution_pd["percentage"], labels=season_distribution_pd["season"], autopct='%1.1f%%')
plt.title("Distribution of Trips by Season")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### near_airport

# COMMAND ----------

from pyspark.sql.functions import col, when

# Define near_airport based on pickup or dropoff being in an airport zone
taxi_df = taxi_df.withColumn(
    "near_airport",
    when(
        (col("pickup_service_zone") == "Airports") | 
        (col("dropoff_service_zone") == "Airports"),
        1
    ).otherwise(0)
)

# COMMAND ----------

airport_distribution = (
    taxi_df.groupBy("near_airport")
    .agg(F.count("*").alias("count"))
)

total_count = airport_distribution.agg(F.sum("count")).collect()[0][0]
airport_distribution = airport_distribution.withColumn("percentage", (F.col("count") / total_count) * 100)

airport_distribution_pd = airport_distribution.toPandas()

airport_distribution_pd["near_airport"] = airport_distribution_pd["near_airport"].map({1: "Near Airport", 0: "Not Near Airport"})

plt.figure(figsize=(8, 6))
plt.pie(airport_distribution_pd["percentage"], labels=airport_distribution_pd["near_airport"], autopct='%1.1f%%')
plt.title("Distribution of Trips Near Airport")
plt.show()

# COMMAND ----------

display(taxi_df.limit(20))

# COMMAND ----------

# Get the number of rows and columns
num_rows = taxi_df.count()
num_cols = len(taxi_df.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

# Display column names of the DataFrame
taxi_df.columns

# COMMAND ----------

# Select only the columns needed for price prediction
taxi_df_cleaned = taxi_df.select(
    'DOLocationID',
    'PULocationID',
    'passenger_count',
    'payment_type',
    'total_amount',
    'trip_duration',
    'pickup_day_of_week',
    'pickup_hour',
    'pickup_month',
    'pickup_borough',
    'dropoff_borough',
    'temp',
    'distance_km',
    'is_holiday',
    'distance_bin',
    'temp_bin',
    'time_of_day_bin',
    'season',
    'near_airport'
)


# COMMAND ----------

display(taxi_df_cleaned.limit(10))

# COMMAND ----------

# Get the number of rows and columns
num_rows = taxi_df_cleaned.count()
num_cols = len(taxi_df_cleaned.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

# MAGIC %md
# MAGIC | Column Name         | Type        | Description                               | Preprocessing Steps                           |
# MAGIC |---------------------|-------------|-------------------------------------------|-----------------------------------------------|
# MAGIC | `DOLocationID`      | Categorical | Dropoff Location ID                       | Encode (One-Hot or Label Encoding)           |
# MAGIC | `PULocationID`      | Categorical | Pickup Location ID                        | Encode (One-Hot or Label Encoding)           |
# MAGIC | `passenger_count`   | Numerical   | Number of passengers                      | Scale (Min-Max or Standard Scaling)          |
# MAGIC | `payment_type`      | Categorical | Type of payment (e.g., cash, card)      | Encode (One-Hot or Label Encoding)           |
# MAGIC | `total_amount`      | Numerical   | Total fare amount                         | Scale (Min-Max or Standard Scaling)          |
# MAGIC | `trip_duration`     | Numerical   | Duration of the trip (in seconds)        | Scale (Min-Max or Standard Scaling)          |
# MAGIC | `pickup_day_of_week`| Categorical | Day of the week (0-6, where 0 = Sunday) | Encode (One-Hot Encoding)                    |
# MAGIC | `pickup_hour`       | Numerical   | Hour of pickup (0-23)                    | Scale (Min-Max or Standard Scaling)          |
# MAGIC | `pickup_month`      | Numerical   | Month of pickup (1-12)                   | Encode (One-Hot Encoding)                     |
# MAGIC | `pickup_borough`    | Categorical | Borough where pickup occurs               | Encode (One-Hot Encoding)                     |
# MAGIC | `dropoff_borough`   | Categorical | Borough where dropoff occurs              | Encode (One-Hot Encoding)                     |
# MAGIC | `temp`              | Numerical   | Temperature (in °C)                      | Scale (Min-Max or Standard Scaling)          |
# MAGIC | `distance_km`     | Numerical   | Distance of the trip (in km)             | Scale (Min-Max or Standard Scaling)          |
# MAGIC | `is_holiday`        | Categorical | Whether the trip occurs on a holiday     | Encode (Binary Encoding)                      |
# MAGIC | `distance_bin`      | Categorical | Binned distance categories (Short, Medium, Long) | Encode (One-Hot Encoding)         |
# MAGIC | `temp_bin`          | Categorical | Binned temperature categories             | Encode (One-Hot Encoding)                     |
# MAGIC | `time_of_day_bin`   | Categorical | Binned time of day (e.g., Morning, Afternoon, Night) | Encode (One-Hot Encoding)    |
# MAGIC | `season`            | Categorical | Season during the trip                    | Encode (One-Hot Encoding)                     |
# MAGIC | `near_airport`      | Categorical | Proximity to airport (0 or 1)            | Encode (Binary Encoding)                      |

# COMMAND ----------

taxi_df_cleaned.printSchema()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression

# Define categorical and numerical columns
categorical_cols = [
    'payment_type', 'pickup_day_of_week',
    'pickup_month', 'pickup_borough', 'dropoff_borough', 'is_holiday',
    'distance_bin', 'temp_bin', 'time_of_day_bin', 'season', 'near_airport'
]

numerical_cols = ['passenger_count', 'trip_duration', 'pickup_hour', 'temp', 'distance_km']

# COMMAND ----------

# MAGIC %md
# MAGIC the column needs to be transformed into a vector format before it can be used in your pipeline.

# COMMAND ----------

# Apply StringIndexer for ordinal encoding on categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]

# COMMAND ----------


# Assemble all features into a single vector, using ordinal indices for categorical features
assembler = VectorAssembler(inputCols=[col + "_index" for col in categorical_cols] + numerical_cols, outputCol="features")

# COMMAND ----------

# Scale the features
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

# COMMAND ----------

# Create the pipeline without one-hot encoding
pipeline = Pipeline(stages=indexers + [assembler, scaler])

# COMMAND ----------

# Fit the pipeline to the DataFrame
pipeline_model = pipeline.fit(taxi_df_cleaned)

# COMMAND ----------

# Transform the DataFrame using the fitted pipeline
preprocessed_df = pipeline_model.transform(taxi_df_cleaned)

# COMMAND ----------

preprocessed_df.printSchema()

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col

# Compute the correlation matrix using Pearson correlation (default)
correlation_matrix = Correlation.corr(preprocessed_df, "scaled_features").head()[0]

# Convert correlation matrix to a readable format
correlation_array = correlation_matrix.toArray()

# Display the correlation matrix
print("Correlation Matrix:\n", correlation_array)


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.types import NumericType
from pyspark.sql import functions as F
from pyspark.sql.functions import col

# COMMAND ----------

from pyspark.sql.types import NumericType

# Define the target column
target_column = "total_amount"

# Automatically identify numerical and indexed categorical features
numerical_features = [
    col for col in preprocessed_df.columns 
    if isinstance(preprocessed_df.schema[col].dataType, NumericType) and col != target_column
]

# Identify ordinally encoded categorical features (those ending in '_index')
indexed_features = [col for col in preprocessed_df.columns if col.endswith('_index')]

# COMMAND ----------


# Combine all feature columns (numerical + indexed)
all_features = numerical_features + indexed_features

# COMMAND ----------

# Sample data to optimize runtime
sample_fraction = 0.05
sampled_df = preprocessed_df.sample(fraction=sample_fraction, seed=42).cache()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructType, StructField, StringType

# Updated helper function to extract feature importances and normalize them
def get_feature_importance(model, features, importance_col):
    importances = model.stages[-1].featureImportances.toArray()
    
    # Specify schema with types
    schema = StructType([
        StructField("feature", StringType(), True),
        StructField(importance_col, FloatType(), True)
    ])
    
    importance_df = spark.createDataFrame([(feature, float(importance)) for feature, importance in zip(features, importances)], schema)
    total_importance = importance_df.select(F.sum(importance_col)).collect()[0][0]
    return importance_df.withColumn(importance_col, F.col(importance_col) / total_importance)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 1: Correlation Analysis

# COMMAND ----------

# Define categorical and numerical columns
categorical_cols = [
    'payment_type', 'pickup_day_of_week',
    'pickup_month', 'pickup_borough', 'dropoff_borough', 'is_holiday',
    'distance_bin', 'temp_bin', 'time_of_day_bin', 'season', 'near_airport'
]

numerical_cols = ['passenger_count', 'trip_duration', 'pickup_hour', 'temp', 'distance_km']

# Define the target column
target_column = 'total_amount'

# Collect all feature columns (numerical and categorical index columns)
all_feature_cols = numerical_cols + [col + "_index" for col in categorical_cols]

# COMMAND ----------

# Prepare a dictionary to hold correlations
correlations = {}

# Calculate correlations for numerical features
for col in numerical_cols:
    if col in preprocessed_df.columns:  # Check if the column exists
        correlation = preprocessed_df.stat.corr(col, target_column)
        correlations[col] = correlation

# COMMAND ----------

# Calculate correlations for ordinally encoded categorical features
for col in categorical_cols:
    indexed_col = col + "_index"
    if indexed_col in preprocessed_df.columns:  # Check if the indexed column exists
        correlation = preprocessed_df.stat.corr(indexed_col, target_column)
        correlations[indexed_col] = correlation

# COMMAND ----------

# Calculate mean target for each category in categorical features
for col in categorical_cols:
    if col in preprocessed_df.columns:  # Check if the column exists
        mean_target = preprocessed_df.groupBy(col).agg({target_column: 'mean'}).collect()
        for row in mean_target:
            category = row[col]
            mean_value = row[f'avg({target_column})']  # Access the mean value
            correlations[f'Mean {target_column} for {col} = {category}'] = mean_value

# COMMAND ----------

# Convert correlations dictionary to a Spark DataFrame
correlation_importance_df = (
    spark.createDataFrame([(k, abs(v)) for k, v in correlations.items()], ["feature", "correlation_importance"])
)

# COMMAND ----------

# Display correlations
display(correlation_importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 2: Random Forest Feature Importance

# COMMAND ----------

# 2. Random Forest Feature Importance
assembler_rf = VectorAssembler(inputCols=all_features, outputCol="rf_features")

# COMMAND ----------

rf = RandomForestRegressor(labelCol=target_column, featuresCol="rf_features", numTrees=50)

# COMMAND ----------

pipeline_rf = Pipeline(stages=[assembler_rf, rf])

# COMMAND ----------

rf_model = pipeline_rf.fit(sampled_df)

# COMMAND ----------

rf_importance_df = get_feature_importance(rf_model, all_features, "rf_importance")

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert specific numerical feature columns from both dataframes to pandas for visualization
feature = numerical_features[0]  # Replace this with the desired feature name
original_feature_data = preprocessed_df.select(feature).toPandas()
sampled_feature_data = sampled_df.select(feature).toPandas()

# Add a column to differentiate original vs. sampled data
original_feature_data['Source'] = 'Original'
sampled_feature_data['Source'] = 'Sampled'

# Concatenate both DataFrames for easy plotting
combined_data = pd.concat([original_feature_data, sampled_feature_data], ignore_index=True)

# Plot distributions side by side on the same plot
plt.figure(figsize=(10, 6))
sns.histplot(data=combined_data, x=feature, hue='Source', kde=True, stat="density", common_norm=False, bins=30)

plt.title(f"Comparison of {feature} Distribution: Original vs Sampled Data")
plt.xlabel(f"{feature}")
plt.ylabel("Density")
plt.legend(title="Data Source")
plt.show()


# COMMAND ----------

display(rf_importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 3: Gradient Boosting Feature Importance

# COMMAND ----------

# 3. Gradient Boosting Feature Importance
assembler_gbt = VectorAssembler(inputCols=all_features, outputCol="gbt_features")

# COMMAND ----------

gbt = GBTRegressor(labelCol=target_column, featuresCol="gbt_features", maxIter=50)

# COMMAND ----------

pipeline_gbt = Pipeline(stages=[assembler_gbt, gbt])

# COMMAND ----------

gbt_model = pipeline_gbt.fit(sampled_df)

# COMMAND ----------

gbt_importance_df = get_feature_importance(gbt_model, all_features, "gbt_importance")

# COMMAND ----------

display(gbt_importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 4: Lasso Regression

# COMMAND ----------

# 4. Lasso Regression (Linear Regression with L1 regularization) Feature Importance
assembler_lasso = VectorAssembler(inputCols=all_features, outputCol="lasso_features")

# COMMAND ----------

lasso = LinearRegression(labelCol=target_column, featuresCol="lasso_features", regParam=0.1, elasticNetParam=1.0)

# COMMAND ----------

pipeline_lasso = Pipeline(stages=[assembler_lasso, lasso])

# COMMAND ----------

lasso_model = pipeline_lasso.fit(sampled_df)

# COMMAND ----------

lasso_coefficients = lasso_model.stages[-1].coefficients.toArray()

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import col

# Define the schema explicitly
schema = StructType([
    StructField("feature", StringType(), True),
    StructField("lasso_importance", DoubleType(), True)
])

# Convert the coefficients to native Python float before creating the DataFrame
lasso_importance_data = [(feature, float(abs(coeff))) for feature, coeff in zip(all_features, lasso_coefficients)]

# COMMAND ----------

# Create the lasso importance DataFrame with the explicit schema
lasso_importance_df = spark.createDataFrame(lasso_importance_data, schema)

# COMMAND ----------

# Normalize Lasso importances
total_lasso_importance = lasso_importance_df.select(F.sum("lasso_importance")).collect()[0][0]

lasso_importance_df = lasso_importance_df.withColumn("lasso_importance", col("lasso_importance") / total_lasso_importance)

# COMMAND ----------

# Show the final DataFrame with normalized importance values
display(lasso_importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 5: Recursive Feature Elimination (RFE)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col

# List of all feature names
rfe_features = all_features.copy()  
numTrees = 30

# COMMAND ----------

# Assemble the features into a single vector column
assembler_rfe = VectorAssembler(inputCols=rfe_features, outputCol="rfe_features")
sampled_df_transformed = assembler_rfe.transform(sampled_df)

# COMMAND ----------

# Define and fit the Random Forest model on the transformed data
rf_rfe = RandomForestRegressor(labelCol=target_column, featuresCol="rfe_features", numTrees=numTrees)
rfe_model = rf_rfe.fit(sampled_df_transformed.select("rfe_features", target_column))

# COMMAND ----------

# Capture feature importances
importances = rfe_model.featureImportances.toArray()

# COMMAND ----------

# Convert all importance scores to Python's native float type
fi_data = [(feature, float(importance)) for feature, importance in zip(rfe_features, importances)]

# COMMAND ----------

# Now create the DataFrame with the specified schema
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Define schema with expected types
schema = StructType([
    StructField("feature", StringType(), True),
    StructField("rfe_score", FloatType(), True)
])

# Create DataFrame with the specified schema
rfe_importance_df = spark.createDataFrame(fi_data, schema=schema)

# COMMAND ----------

# Sort the DataFrame by importance scores in descending order
rfe_importance_df_sorted = rfe_importance_df.orderBy(col("rfe_score").desc())

# COMMAND ----------

display(rfe_importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 6: SHAP Values

# COMMAND ----------

# DBTITLE 1,5
import xgboost as xgb
import shap
import numpy as np
import pandas as pd

# Convert the Spark DataFrame to Pandas for model training
sampled_pd_df = sampled_df.select(all_features + [target_column]).toPandas()

# COMMAND ----------

# Define the features and target
X = sampled_pd_df[all_features]
y = sampled_pd_df[target_column]

# COMMAND ----------

# Ensure X and y are in the correct shape
X = np.array(X)
y = np.array(y).ravel()

# COMMAND ----------

# Train an XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
xgb_model.fit(X, y)

# COMMAND ----------

# Use SHAP to explain the model with correct array format
explainer = shap.Explainer(xgb_model, X)

# COMMAND ----------

shap_values = explainer(X).values

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, FloatType
# Compute mean absolute SHAP values for each feature (importance scores) and convert to Python floats
shap_importances = [(feature, float(np.abs(shap_values[:, i]).mean())) for i, feature in enumerate(all_features)]

# COMMAND ----------

# Define schema and create a PySpark DataFrame for SHAP importances
schema = StructType([
    StructField("feature", StringType(), True),
    StructField("shap_importance", FloatType(), True)
])

# COMMAND ----------

# Convert list to PySpark DataFrame with the specified schema
shap_importance_df = spark.createDataFrame(shap_importances, schema=schema)

# COMMAND ----------

# Show SHAP feature importances
display(shap_importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 7: Statistical Tests

# COMMAND ----------

# MAGIC %md
# MAGIC #### Chi-Square Test for categorical features

# COMMAND ----------

# df.printSchema()

# COMMAND ----------

# from pyspark.ml.stat import ChiSquareTest
# from pyspark.ml.feature import VectorAssembler

# # Step 1: Select relevant categorical columns for the Chi-square test
# categorical_columns = [
#     "pickup_day_of_week_index",
#     "distance_bin_index",
#     "temp_bin_index",
#     "season_index",
#     "near_airport_index"
# ]

# # Step 2: Vectorize the categorical features (you might need to ensure 'payment_type_index' is numeric)
# assembler = VectorAssembler(
#     inputCols=categorical_columns,
#     outputCol="features"
# )

# # Apply vector assembler to the dataframe
# df = assembler.transform(df)

# # Step 3: Perform the Chi-square test for each feature individually against the label
# for column in categorical_columns:
#     # Perform the Chi-square test on the vectorized features and label (payment_type_index)
#     chi_square_result = ChiSquareTest.test(df, "features", "payment_type_index")
    
#     # Step 4: Show the results for the current feature
#     chi_square_result.select("pValues", "degreesOfFreedom", "statistics").show(truncate=False)
    
#     # Step 5: Extract and print the chi-square statistics and p-values
#     result = chi_square_result.collect()[0]
#     p_values = result['pValues']
#     statistics = result['statistics']
    
#     print(f"Feature: {column}")
#     print(f"Chi-Square Statistic: {statistics[0]}")
#     print(f"P-Value: {p_values[0]}\n")


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### ANOVA F-Test for continuous features

# COMMAND ----------

# from pyspark.ml.stat import ANOVA
# import pandas as pd

# # Define categorical features
# categorical_cols = ['pickup_day_of_week', 'payment_type']

# # Calculate ANOVA F-Test for each categorical feature
# anova_results = {}
# for col in categorical_cols:
#     groups = sampled_df.groupBy(col).agg(F.collect_list(target_col).alias("values"))
#     # Create a DataFrame for ANOVA
#     anova_df = groups.toPandas()

#     f_stat, p_val = ANOVA.test(anova_df['values'], sampled_df[target_col])
#     anova_results[col] = (f_stat, p_val)

# # Display ANOVA results
# for feature, (f_stat, p_val) in anova_results.items():
#     print(f"Feature: {feature}, F-Statistic: {f_stat}, P-Value: {p_val}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Mutual Information for non-linear associations

# COMMAND ----------

#  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Technique 8: Feature Selection Based on Statistical Variance

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import NumericType

# Define the variance threshold
variance_threshold = 0.05  # Adjust as needed

# Identify numerical columns
numerical_columns = [col for col, dtype in sampled_df.dtypes if isinstance(sampled_df.schema[col].dataType, NumericType) and col != target_col]

# Calculate variance for each numerical column
variance_df = sampled_df.select(
    *[F.variance(F.col(c)).alias(c) for c in numerical_columns]
).toPandas()

# Filter columns based on the variance threshold
selected_features = [col for col, var in variance_df.loc[0].items() if var > variance_threshold]

# Display selected features
print("Selected Features based on Variance:")
print(selected_features)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge Feature Importances

# COMMAND ----------

final_importance_df = rf_importance_df \
    .join(gbt_importance_df, on="feature", how="outer") \
    .join(lasso_importance_df, on="feature", how="outer") \
    .join(rfe_importance_df, on="feature", how="outer") \
    .join(shap_importance_df, on="feature", how="outer") \
    .fillna(0)

# COMMAND ----------

from pyspark.sql.functions import col
# Calculate average importance across all methods
final_importance_df = final_importance_df.withColumn(
    "average_importance", 
    (col("rf_importance") + col("gbt_importance") + col("lasso_importance") +
     col("rfe_score") + col("shap_importance")) / 5
)

# COMMAND ----------

# Show the final DataFrame with average feature importance
display(final_importance_df.orderBy("average_importance", ascending=False))

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Collect the Spark DataFrame into Pandas for plotting
final_importance_pd = final_importance_df.select("feature", "average_importance").toPandas()

# Sort values by average_importance in descending order to have the highest on top
final_importance_pd = final_importance_pd.sort_values(by="average_importance", ascending=False)

# Create the horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=final_importance_pd, y='feature', x='average_importance', palette='viridis')

# Set the plot labels and title
plt.title('Average Feature Importance Across Methods', fontsize=16)
plt.xlabel('Average Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)

# Show the plot
plt.tight_layout()  # Adjust layout to avoid label overlap
plt.show()

# COMMAND ----------

display(final_importance_pd)

# COMMAND ----------

# # Export reduced feature dataset if needed
# selected_features = [row["feature"] for row in final_fi_df.select("feature").collect()]
# reduced_data = transformed_data.select(["total_amount", "features"] + selected_features)


# COMMAND ----------

display(taxi_df_cleaned)

# COMMAND ----------

# Select only the columns needed for price prediction
taxi_final_df_cleaned = taxi_df_cleaned.select(
    'passenger_count',
    'payment_type',
    'total_amount',
    'trip_duration',
    'pickup_day_of_week',
    'pickup_hour',
    'pickup_month',
    'pickup_borough',
    'dropoff_borough',
    'is_holiday',
    'distance_bin',
    'time_of_day_bin',
    'near_airport'
)


# COMMAND ----------

display(taxi_final_df_cleaned)

# COMMAND ----------

# Check the Shape and Schema
# Get the number of rows and columns
num_rows = taxi_final_df_cleaned.count()
num_cols = len(taxi_final_df_cleaned.columns)
print(f"Shape: ({num_rows}, {num_cols})")

# COMMAND ----------

# Define a more descriptive Delta Lake storage path
delta_path = "/dbfs/FileStore/tables/taxi_final_df_cleaned/"
# Write the DataFrame to Delta format
taxi_final_df_cleaned.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(delta_path)
