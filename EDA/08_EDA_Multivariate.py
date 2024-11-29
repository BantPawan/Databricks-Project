# Databricks notebook source
display(dbutils.fs.ls("/dbfs/FileStore/tables/"))

# COMMAND ----------

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Fare Analysis") \
    .getOrCreate()

# COMMAND ----------

# Load the Delta table
taxi_df = spark.read.format("delta").load("/dbfs/FileStore/tables/Eda_univariate_taxi_data/")

# Show the first few rows to verify
display(taxi_df.limit(5))

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

# MAGIC %md
# MAGIC ### Geographic Data Analysis (Latitude and Longitude)

# COMMAND ----------

from pyspark.sql.functions import col
import matplotlib.pyplot as plt

# Define city border coordinates
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)

# COMMAND ----------

# Filter data for valid latitude and longitude values
taxi_df_filtered = taxi_df.filter(
    (col("pickup_longitude") >= city_long_border[0]) & 
    (col("pickup_longitude") <= city_long_border[1]) &
    (col("pickup_latitude") >= city_lat_border[0]) & 
    (col("pickup_latitude") <= city_lat_border[1]) &
    (col("dropoff_longitude") >= city_long_border[0]) & 
    (col("dropoff_longitude") <= city_long_border[1]) &
    (col("dropoff_latitude") >= city_lat_border[0]) & 
    (col("dropoff_latitude") <= city_lat_border[1])
)

# COMMAND ----------

# Convert filtered DataFrame to Pandas for plotting
taxi_df_filtered_pd = taxi_df_filtered.select(
    "pickup_longitude", "pickup_latitude", 
    "dropoff_longitude", "dropoff_latitude"
).toPandas()

# COMMAND ----------

# Plot dropoffs
plt.figure(figsize=(10, 6))
plt.scatter(taxi_df_filtered_pd['dropoff_longitude'], taxi_df_filtered_pd['dropoff_latitude'],
            color='green', s=0.02, alpha=0.6)
plt.title("Dropoffs")
plt.xlim(city_long_border)
plt.ylim(city_lat_border)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary Statistics

# COMMAND ----------

# Display summary statistics for numerical columns
numeric_cols = ["fare_amount", "trip_distance", "tolls_amount", "total_amount", "trip_duration", "distance_km", "temp"]
display(taxi_df.select(numeric_cols).describe())

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select numerical columns and assemble them into a feature vector
numeric_cols = ["fare_amount", "trip_distance", "tolls_amount", "total_amount", "trip_duration", "distance_km", "temp"]
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
taxi_df_vector = assembler.transform(taxi_df).select("features")

# Calculate correlation matrix
correlation_matrix = Correlation.corr(taxi_df_vector, "features").head()[0]
correlation_values = correlation_matrix.toArray().tolist()

# Convert to a DataFrame for seaborn heatmap
corr_df = pd.DataFrame(correlation_values, index=numeric_cols, columns=numeric_cols)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Numerical Features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fare Amount

# COMMAND ----------

from pyspark.sql.functions import hour

# Extract hour from timestamp and compute average fare
hourly_fare_df = taxi_df.withColumn("hour", hour("pickup_datetime")) \
                        .groupBy("hour") \
                        .avg("fare_amount") \
                        .orderBy("hour")

# Convert to Pandas for visualization
hourly_fare_pd = hourly_fare_df.toPandas()

plt.figure(figsize=(10, 6))
plt.plot(hourly_fare_pd["hour"], hourly_fare_pd["avg(fare_amount)"], marker='o')
plt.xlabel("Hour of Day")
plt.ylabel("Average Fare Amount")
plt.title("Average Fare Amount by Hour of Day")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Average trip distance by day of the week

# COMMAND ----------

avg_distance_by_day = taxi_df.groupBy("pickup_day_of_week").agg(F.mean("trip_distance").alias("avg_distance")).orderBy("pickup_day_of_week")

# Convert to pandas for plotting
daily_distance_df = avg_distance_by_day.toPandas()

# Plot
plt.figure(figsize=(10, 6))
plt.bar(daily_distance_df["pickup_day_of_week"], daily_distance_df["avg_distance"], color="forestgreen")
plt.title("Average Trip Distance by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Trip Distance (miles)")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Average trips by weekdays

# COMMAND ----------

avg_trip_by_weekday = taxi_df.groupBy("pickup_day_of_week").count().orderBy("pickup_day_of_week")

# Convert to Pandas for visualization
avg_trip_by_weekday_pd = avg_trip_by_weekday.toPandas()

# Plot
plt.bar(avg_trip_by_weekday_pd["pickup_day_of_week"], avg_trip_by_weekday_pd["count"])
plt.xlabel("Weekday")
plt.ylabel("Average Trip Count")
plt.title("Average Trip by Weekdays")
plt.show()

# COMMAND ----------

# Trip proportion by weekday
trip_by_weekday = taxi_df.groupBy("pickup_day_of_week").count()
trip_by_weekday_pd = trip_by_weekday.toPandas()

# Plot as donut chart
plt.pie(trip_by_weekday_pd["count"], labels=trip_by_weekday_pd["pickup_day_of_week"], autopct="%1.1f%%", startangle=90)
plt.gca().add_artist(plt.Circle((0,0),0.7, color="white"))  # For donut chart
plt.title("Trip Proportion by Weekdays")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Proportion of trips for each weekday.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fare by Payment Type

# COMMAND ----------

# Group by payment type and calculate average fare
payment_fare_df = taxi_df.groupBy("payment_type").avg("fare_amount")

# Convert to Pandas for plotting
payment_fare_pd = payment_fare_df.toPandas()

plt.figure(figsize=(10, 6))
sns.barplot(x="payment_type", y="avg(fare_amount)", data=payment_fare_pd, palette="viridis")
plt.xlabel("Payment Type")
plt.ylabel("Average Fare Amount")
plt.title("Average Fare Amount by Payment Type")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Total Amount vs. Trip Duration

# COMMAND ----------

# Sample and plot data for Total Amount vs Trip Duration
sample_df = taxi_df.select("trip_duration", "total_amount").sample(fraction=0.01).toPandas()

plt.figure(figsize=(8, 6))
plt.scatter(sample_df["trip_duration"], sample_df["total_amount"], alpha=0.5, color="orange")
plt.title("Trip Duration vs. Total Amount")
plt.xlabel("Trip Duration (seconds)")
plt.ylabel("Total Amount")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trip Distance vs. Fare Amount

# COMMAND ----------

# Sample data
sample_df = taxi_df.select("trip_distance", "fare_amount").sample(fraction=0.01).toPandas()

# COMMAND ----------

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(sample_df["trip_distance"], sample_df["fare_amount"], alpha=0.5, color="teal")
plt.title("Trip Distance vs. Fare Amount")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Fare Amount")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Monthly trip distribution

# COMMAND ----------

monthly_trip_distribution = taxi_df.groupBy("Month_Num").count().orderBy("Month_Num")

# Convert to Pandas for visualization
monthly_trip_distribution_pd = monthly_trip_distribution.toPandas()

# Plot in matplotlib or seaborn
import matplotlib.pyplot as plt

plt.plot(monthly_trip_distribution_pd["Month_Num"], monthly_trip_distribution_pd["count"])
plt.xlabel("Month")
plt.ylabel("Total Trip Count")
plt.title("Total Trip Distribution by Month")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Total Trips by Day

# COMMAND ----------

# Daily trip distribution
daily_trip_distribution = taxi_df.groupBy("pickup_date").count().orderBy("pickup_date")

# Convert to Pandas for visualization
daily_trip_distribution_pd = daily_trip_distribution.toPandas()

# Plot
plt.plot(daily_trip_distribution_pd["pickup_date"], daily_trip_distribution_pd["count"])
plt.xlabel("Date")
plt.ylabel("Total Trip Count")
plt.title("Total Trip by Day")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Total Trip by Hour

# COMMAND ----------

import seaborn as sns

# Total trip by hour and weekday
hourly_trip = taxi_df.groupBy("pickup_hour", "pickup_day_of_week").count().orderBy("pickup_hour", "pickup_day_of_week")
hourly_trip_pd = hourly_trip.toPandas()

# Pivot the data
hourly_trip_pivot = hourly_trip_pd.pivot("pickup_hour", "pickup_day_of_week", "count")

# Plot heatmap
sns.heatmap(hourly_trip_pivot, cmap="YlGnBu")
plt.xlabel("Day of Week")
plt.ylabel("Hour of Day")
plt.title("Total Trip by Hour")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scatter Plot of Fare vs. Trip Distance

# COMMAND ----------

import seaborn as sns

# Sample data to reduce memory load
sample_df = taxi_df.select("fare_amount", "trip_distance").sample(fraction=0.01).toPandas()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="trip_distance", y="fare_amount", data=sample_df, alpha=0.5)
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Fare Amount")
plt.title("Fare Amount vs. Trip Distance")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Duration in minutes

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp

taxi_df = taxi_df.withColumn("trip_duration", 
                             (unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime")) / 60)

# Convert to Pandas for histogram
duration_pd = taxi_df.select("trip_duration").sample(fraction=0.1).toPandas()

plt.figure(figsize=(10, 6))
plt.hist(duration_pd['trip_duration'], bins=50, color='purple')
plt.xlabel("Trip Duration (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Trip Durations")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pickup and Dropoff Hour

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analyze hourly pickup distribution

# COMMAND ----------

pickup_hour_data = taxi_df.groupBy("pickup_hour").count().orderBy("pickup_hour").collect()
x = [row['pickup_hour'] for row in pickup_hour_data]
y = [row['count'] for row in pickup_hour_data]

plt.plot(x, y, marker="o", color="blue")
plt.title("Pickup Hour Distribution")
plt.xlabel("Hour of Day")
plt.ylabel("Frequency")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Analyze hourly dropoff distribution

# COMMAND ----------

pickup_hour_data = taxi_df.groupBy("dropoff_hour").count().orderBy("dropoff_hour").collect()
x = [row['dropoff_hour'] for row in pickup_hour_data]
y = [row['count'] for row in pickup_hour_data]

plt.plot(x, y, marker="o", color="blue")
plt.title("Dropoff Hour Distribution")
plt.xlabel("Hour of Day")
plt.ylabel("Frequency")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Zone-Based Insights

# COMMAND ----------

# MAGIC %md
# MAGIC #### Distribution of Total and Median Fare Amount based on Borough

# COMMAND ----------

# Calculate median fare amount by borough
borough_fare = taxi_df.groupBy("pickup_borough").agg(F.avg("fare_amount").alias("avg_fare"), F.expr("percentile_approx(fare_amount, 0.5)").alias("median_fare"))

# Convert to Pandas for visualization
borough_fare_pd = borough_fare.toPandas()

import squarify

# Plot as a tree map
squarify.plot(sizes=borough_fare_pd["avg_fare"], label=borough_fare_pd["pickup_borough"], alpha=0.7)
plt.title("Distribution of Average Fare Amount by Borough")
plt.axis("off")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Median Fare Amount based on Time and Borough

# COMMAND ----------

# Calculate median fare amount by hour and borough
time_borough_fare = taxi_df.groupBy("pickup_hour", "pickup_borough").agg(F.expr("percentile_approx(fare_amount, 0.5)").alias("median_fare"))

# Convert to Pandas for visualization
time_borough_fare_pd = time_borough_fare.toPandas()

# Plot
sns.barplot(x="pickup_hour", y="median_fare", hue="pickup_borough", data=time_borough_fare_pd)
plt.title("Median Fare Amount by Hour and Borough")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Total Trip by Pickup Borough and Pickup Zone

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, desc

top_trips_df = (
    taxi_df.groupBy("pickup_zone", "dropoff_zone")
    .count()
    .orderBy(desc("count"))
    .limit(10)
)

top_trips_pd = top_trips_df.toPandas()

top_trips_pivot = top_trips_pd.pivot(index="pickup_zone", columns="dropoff_zone", values="count")

plt.figure(figsize=(12, 8))
sns.heatmap(
    top_trips_pivot,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor="gray",
    cbar_kws={"label": "Trip Count"},
)
plt.title("Top 10 Trips by Pickup and Dropoff Zone")
plt.xlabel("Dropoff Zone")
plt.ylabel("Pickup Zone")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Top Pickup and Drop-off Zones

# COMMAND ----------


# Count the number of trips per pickup and dropoff zones
pickup_zones_df = taxi_df.groupBy("pickup_zone").count().orderBy("count", ascending=False).limit(10)
dropoff_zones_df = taxi_df.groupBy("dropoff_zone").count().orderBy("count", ascending=False).limit(10)

# Convert to Pandas for plotting
pickup_zones_pd = pickup_zones_df.toPandas()
dropoff_zones_pd = dropoff_zones_df.toPandas()

plt.figure(figsize=(10, 6))
sns.barplot(x="pickup_zone", y="count", data=pickup_zones_pd, color="skyblue")
plt.xlabel("Pickup Zone")
plt.ylabel("Number of Trips")
plt.title("Top 10 Pickup Zones")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------


plt.figure(figsize=(10, 6))
sns.barplot(x="dropoff_zone", y="count", data=dropoff_zones_pd, color="salmon")
plt.xlabel("Dropoff Zone")
plt.ylabel("Number of Trips")
plt.title("Top 10 Dropoff Zones")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passenger Count Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Effect of Passenger Number on Fare

# COMMAND ----------

# Convert the necessary columns from PySpark DataFrame to Pandas DataFrame
taxi_pd = taxi_df.select("passenger_count", "fare_amount").toPandas()

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=taxi_pd, x='passenger_count', y='fare_amount', ax=ax)
ax.set_title("Effect of Passenger Count on Fare Amount")
ax.set_xlabel("Passenger Count")
ax.set_ylabel("Fare Amount")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Passenger Count

# COMMAND ----------

# Count unique values in 'passenger_count'
taxi_df.groupBy("passenger_count").count().orderBy("count", ascending=False).show()

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
# MAGIC #### Payment Type vs Passenger Count

# COMMAND ----------

# Group by payment type and passenger count to see their interaction
payment_passenger_df = taxi_df.groupBy("payment_type", "passenger_count").count().orderBy("payment_type", "passenger_count").toPandas()

# Pivot for a heatmap
pivot_df = payment_passenger_df.pivot(index="passenger_count", columns="payment_type", values="count").fillna(0)

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".0f")
plt.title("Payment Type vs. Passenger Count")
plt.xlabel("Payment Type")
plt.ylabel("Passenger Count")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Payment Type Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC **1 = Credit Card**
# MAGIC
# MAGIC **2 = Cash**
# MAGIC
# MAGIC **3 = No Charge**
# MAGIC
# MAGIC **4 = Dispute**
# MAGIC
# MAGIC **5 = Unknown**
# MAGIC
# MAGIC **6 = Voided Trip**
# MAGIC
# MAGIC
# MAGIC 4 = Dispute
# MAGIC 5 = Unknown
# MAGIC 6 = Voided Trip
# MAGIC
# MAGIC
# MAGIC **4 = Dispute**: 
# MAGIC     This indicates that the fare was disputed, possibly due to a disagreement over the amount charged or a service issue. In these cases, the passenger or driver raised a concern about the payment.
# MAGIC
# MAGIC **5 = Unknown**: 
# MAGIC     This is used when the payment status is unclear or unrecorded.
# MAGIC     It might mean that the payment method was not identified or there was an error in recording it.
# MAGIC
# MAGIC **6 = Voided Trip**: 
# MAGIC     This indicates that the trip was voided, canceled, or invalid for some reason,
# MAGIC     so the fare amount associated with this trip isn’t charged or is set to zero.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Payment Type

# COMMAND ----------

# Count unique values in 'payment_type'
taxi_df.groupBy("payment_type").count().orderBy("count", ascending=False).show()

# Bar plot for payment type
payment_type_data = taxi_df.groupBy("payment_type").count().orderBy("payment_type").collect()
x = [row['payment_type'] for row in payment_type_data]
y = [row['count'] for row in payment_type_data]

plt.bar(x, y, color="green")
plt.title("Payment Type Distribution")
plt.xlabel("Payment Type")
plt.ylabel("Frequency")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Count and Percentage of Trips by Payment Method

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Define a window to calculate the total trip count across all rows
window_spec = Window.partitionBy()

# Count of trips by payment method
payment_counts = taxi_df.groupBy("payment_type").agg(
    F.count("*").alias("trip_count")
).withColumn(
    "percentage", (F.col("trip_count") / F.sum("trip_count").over(window_spec)) * 100
).orderBy("trip_count", ascending=False)

payment_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Average Fare Amount by Payment Method

# COMMAND ----------

# Average fare amount by payment method
average_fare = taxi_df.groupBy("payment_type").agg(
    F.avg("fare_amount").alias("average_fare"),
    F.median("fare_amount").alias("median_fare"),
    F.sum("fare_amount").alias("total_fare")
)

average_fare.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Effect of Date and Time on Fare

# COMMAND ----------

# Group by hour and calculate the average fare
avg_fare_by_hour = (
    taxi_df.groupBy("pickup_hour")
    .agg(F.avg("fare_amount").alias("average_fare"))
    .orderBy("pickup_hour")
)

# Convert to Pandas DataFrame for visualization
avg_fare_by_hour_pd = avg_fare_by_hour.toPandas()

# COMMAND ----------

# Group by day of the week and calculate the average fare
avg_fare_by_day = (
    taxi_df.groupBy("pickup_day_of_week")
    .agg(F.avg("fare_amount").alias("average_fare"))
    .orderBy("pickup_day_of_week")
)

# Convert to Pandas DataFrame for visualization
avg_fare_by_day_pd = avg_fare_by_day.toPandas()


# COMMAND ----------

# Group by month and calculate the average fare
avg_fare_by_month = (
    taxi_df.groupBy("pickup_month")
    .agg(F.avg("fare_amount").alias("average_fare"))
    .orderBy("pickup_month")
)

# Convert to Pandas DataFrame for visualization
avg_fare_by_month_pd = avg_fare_by_month.toPandas()

# COMMAND ----------

# Set up subplots
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

# Plot Average Fare by Hour
sns.lineplot(data=avg_fare_by_hour_pd, x="pickup_hour", y="average_fare", marker="o", ax=ax[0], color="b")
ax[0].set_title("Average Fare by Hour of Day")
ax[0].set_xlabel("Hour of Day")
ax[0].set_ylabel("Average Fare")

# Plot Average Fare by Day of Week
sns.lineplot(data=avg_fare_by_day_pd, x="pickup_day_of_week", y="average_fare", marker="o", ax=ax[1], color="g")
ax[1].set_title("Average Fare by Day of the Week")
ax[1].set_xlabel("Day of the Week")
ax[1].set_ylabel("Average Fare")
ax[1].set_xticks(range(1, 8))
ax[1].set_xticklabels(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])

# Plot Average Fare by Month
sns.lineplot(data=avg_fare_by_month_pd, x="pickup_month", y="average_fare", marker="o", ax=ax[2], color="r")
ax[2].set_title("Average Fare by Month")
ax[2].set_xlabel("Month")
ax[2].set_ylabel("Average Fare")

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pickup and Drop-off Density Across NYC Boroughs

# COMMAND ----------

pickup_counts = (taxi_df
                 .groupBy('pickup_borough')
                 .agg(F.count('*').alias('pickup_count'))
                 .orderBy('pickup_borough'))

dropoff_counts = (taxi_df
                  .groupBy('dropoff_borough')
                  .agg(F.count('*').alias('dropoff_count'))
                  .orderBy('dropoff_borough'))

pickup_counts_pd = pickup_counts.toPandas()
dropoff_counts_pd = dropoff_counts.toPandas()

density_df = (pickup_counts_pd
              .merge(dropoff_counts_pd, left_on='pickup_borough', right_on='dropoff_borough', how='outer')
              .fillna(0))  # Fill NaN values with 0

# Set the figure size
plt.figure(figsize=(14, 6))

# Bar plot for pickups
sns.barplot(data=density_df, x='pickup_borough', y='pickup_count', color='blue', alpha=0.6, label='Pickups')

# Bar plot for drop-offs (using the same x-axis)
sns.barplot(data=density_df, x='dropoff_borough', y='dropoff_count', color='orange', alpha=0.6, label='Drop-offs')

plt.title('Pickup and Drop-off Density Across NYC Boroughs')
plt.xlabel('Borough')
plt.ylabel('Count')
plt.legend()

plt.show()


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data to reduce load
sampled_df = taxi_df.sample(fraction=0.1, seed=42)  # Use 10% of the data

# Select relevant columns and drop rows with null values
filtered_df = sampled_df.select("pickup_latitude", "pickup_longitude", "pickup_borough").dropna()

# Ensure numeric columns are properly cast
filtered_df = filtered_df.withColumn("pickup_latitude", filtered_df["pickup_latitude"].cast("double"))
filtered_df = filtered_df.withColumn("pickup_longitude", filtered_df["pickup_longitude"].cast("double"))

# Assemble the features into a vector
vector_assembler = VectorAssembler(inputCols=["pickup_latitude", "pickup_longitude"], outputCol="features")
vector_df = vector_assembler.transform(filtered_df)

# Persist data to optimize performance
vector_df.cache()

# Apply KMeans clustering with fewer clusters
kmeans = KMeans(k=10, seed=42)  # Use 10 clusters for simplicity
model = kmeans.fit(vector_df)

# Make predictions
predictions = model.transform(vector_df)

# Limit rows for Pandas conversion
predicted_pd = predictions.select("pickup_latitude", "pickup_longitude", "pickup_borough", "prediction").limit(5000).toPandas()

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=predicted_pd, 
                x='pickup_longitude', 
                y='pickup_latitude', 
                hue='prediction', 
                palette='tab10', 
                s=10, 
                alpha=0.7)

# Overlay borough information
for borough in predicted_pd['pickup_borough'].unique():
    subset = predicted_pd[predicted_pd['pickup_borough'] == borough]
    plt.scatter(subset['pickup_longitude'], subset['pickup_latitude'], label=borough, alpha=0.3)

plt.title("Clustering NYC Taxi Pickups into Major Zones")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title='Boroughs')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather Impact Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Average Fare by Temperature Range

# COMMAND ----------

# Create temperature ranges and calculate average fare amount
temp_ranges = taxi_df.withColumn("temp_range", F.floor(F.col("temp") / 5) * 5)
avg_fare_by_temp = temp_ranges.groupBy("temp_range").agg(F.mean("fare_amount").alias("avg_fare")).orderBy("temp_range")

# Plot average fare amount by temperature range
temp_fare_df = avg_fare_by_temp.toPandas()

plt.figure(figsize=(10, 6))
plt.plot(temp_fare_df["temp_range"], temp_fare_df["avg_fare"], marker="o", color="firebrick")
plt.title("Average Fare Amount by Temperature Range")
plt.xlabel("Temperature Range")
plt.ylabel("Average Fare Amount")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Holiday Analysis

# COMMAND ----------

# Analyze trips based on holidays
holiday_data = taxi_df.groupBy("holidayName").count().orderBy("count", ascending=False).show()

# Bar plot for holidays
holiday_counts = taxi_df.groupBy("holidayName").count().orderBy("holidayName").collect()
x = [row['holidayName'] for row in holiday_counts]
y = [row['count'] for row in holiday_counts]

plt.barh(x, y, color="darkcyan")
plt.title("Holiday Trip Count")
plt.xlabel("Frequency")
plt.ylabel("Holiday Name")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Temperature Analysis

# COMMAND ----------

import matplotlib.pyplot as plt

temp_summary = taxi_df.select("temp").summary("count", "mean", "stddev", "min", "max").show()

temp_data = taxi_df.select("temp").toPandas()

plt.hist(temp_data['temp'], bins=30, color="plum", edgecolor="black")
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()

# COMMAND ----------


