# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler, StandardScaler

# Initialize Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Fare Prediction") \
    .getOrCreate()

# COMMAND ----------

display(dbutils.fs.ls("/dbfs/FileStore/tables/"))

# COMMAND ----------

taxi_df_cleaned = spark.read.format("delta").load("/dbfs/FileStore/tables/taxi_final_df_cleaned/")

# COMMAND ----------

taxi_df_cleaned = taxi_df_cleaned.withColumnRenamed("total_amount", "label")

# COMMAND ----------

# Define categorical and numerical columns
categorical_cols = [
    'payment_type', 'pickup_day_of_week', 'pickup_month', 
    'pickup_borough', 'dropoff_borough', 'is_holiday',
    'distance_bin', 'time_of_day_bin', 'near_airport'
]
standard_scaler_cols = ['trip_duration', 'label']
minmax_scaler_cols = ['pickup_hour', 'pickup_month', 'passenger_count']

# COMMAND ----------

# StringIndexer and OneHotEncoder for categorical variables
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]
one_hot_encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_ohe") for col in categorical_cols]

# COMMAND ----------

# StandardScaler for selected numerical columns
standard_assembler = VectorAssembler(inputCols=standard_scaler_cols, outputCol="standard_features")
standard_scaler = StandardScaler(inputCol="standard_features", outputCol="standard_scaled_features")

# COMMAND ----------

# MinMaxScaler for other numerical columns
minmax_assembler = VectorAssembler(inputCols=minmax_scaler_cols, outputCol="minmax_features")
minmax_scaler = MinMaxScaler(inputCol="minmax_features", outputCol="minmax_scaled_features")

# COMMAND ----------

# Assemble all features into a single vector column
assembler = VectorAssembler(
    inputCols=[col + "_ohe" for col in categorical_cols] + ["standard_scaled_features", "minmax_scaled_features"],
    outputCol="features"
)

# COMMAND ----------

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline(stages=indexers + one_hot_encoders + [standard_assembler, standard_scaler, minmax_assembler, minmax_scaler, assembler])

# COMMAND ----------

# Fit and transform the data using the preprocessing pipeline
preprocessed_model = preprocessing_pipeline.fit(taxi_df_cleaned)
preprocessed_df = preprocessed_model.transform(taxi_df_cleaned).select("features", "label")

# COMMAND ----------

# Step 2: Model Training and Evaluation

# COMMAND ----------

from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Split data into training and test sets
train_df, test_df = preprocessed_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# Define the model (example: Linear Regression)
lr = LinearRegression(featuresCol="features", labelCol="label")

# COMMAND ----------

# Fit the model
lr_model = lr.fit(train_df)

# COMMAND ----------

# Make predictions
lr_predictions = lr_model.transform(test_df)

# COMMAND ----------

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(lr_predictions)
r2 = evaluator.evaluate(lr_predictions, {evaluator.metricName: "r2"})

print(f"Linear Regression - RMSE: {rmse}, R2: {r2}")

# COMMAND ----------

# ---- Try a different model, e.g., Decision Tree Regressor ----

# Define the Decision Tree Regressor
dt = DecisionTreeRegressor(featuresCol="features", labelCol="label")

# COMMAND ----------

# Fit the Decision Tree model
dt_model = dt.fit(train_df)

# COMMAND ----------

# Make predictions
dt_predictions = dt_model.transform(test_df)

# COMMAND ----------

# Evaluate the Decision Tree model
rmse_dt = evaluator.evaluate(dt_predictions)
r2_dt = evaluator.evaluate(dt_predictions, {evaluator.metricName: "r2"})

print(f"Decision Tree Regressor - RMSE: {rmse_dt}, R2: {r2_dt}")
