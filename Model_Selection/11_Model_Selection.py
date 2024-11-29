# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Initialize Spark session
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("S3 Integration with Spark") \
    .config("spark.hadoop.fs.s3a.access.key", "Access_key") \
    .config("spark.hadoop.fs.s3a.secret.key", "access_secret_key") \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# COMMAND ----------

dbutils.fs.ls("/dbfs/FileStore/tables/taxi_final_df_cleaned/")

# COMMAND ----------

# Load the cleaned taxi data
taxi_df_cleaned = spark.read.format("delta").load("/dbfs/FileStore/tables/taxi_final_df_cleaned/")

# COMMAND ----------

from pyspark.sql.functions import col
taxi_df_cleaned = taxi_df_cleaned.withColumn("payment_type", col("payment_type").cast("int"))

# COMMAND ----------

display(taxi_df_cleaned.limit(5))
taxi_df_cleaned.printSchema()

# COMMAND ----------

# Sample data to a smaller fraction to speed up processing
sample_data = taxi_df_cleaned.sample(fraction=0.1, seed=42).cache()
train_data, test_data = sample_data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# Define categorical and numerical columns
categorical_cols = ['pickup_borough', 'dropoff_borough', 'distance_bin', 'time_of_day_bin']
numerical_cols = ['pickup_day_of_week','payment_type','trip_duration', 'pickup_hour', 'pickup_month','is_holiday','passenger_count', 'near_airport']
target_col = "total_amount"

# COMMAND ----------

# MAGIC %md
# MAGIC ## One_Hot_Encoding

# COMMAND ----------

# Define feature transformations
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]
encoders = [
    OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded", handleInvalid="error", dropLast=False) 
    for col in categorical_cols
]
assembler = VectorAssembler(inputCols=[col + "_encoded" for col in categorical_cols] + numerical_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# COMMAND ----------

from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
# Define models to evaluate
models = {
    'linear_reg': LinearRegression(featuresCol="scaledFeatures", labelCol=target_col),
    'decision_tree': DecisionTreeRegressor(featuresCol="scaledFeatures", labelCol=target_col),
    'random_forest': RandomForestRegressor(featuresCol="scaledFeatures", labelCol=target_col),
    'gradient_boosting': GBTRegressor(featuresCol="scaledFeatures", labelCol=target_col)
}

# COMMAND ----------

# Evaluation metrics
r2_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")
mae_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")

# COMMAND ----------

# Define K-fold cross-validation function with optimized settings
def cross_validate_model(model_name, model, train_data, test_data):
    # Pipeline creation
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, model])
    
    # Set up K-fold cross-validator with 3 folds
    param_grid = ParamGridBuilder().build()
    cross_val = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=r2_evaluator, numFolds=3, seed=42)
    
    # Fit cross-validator on training data
    cv_model = cross_val.fit(train_data)
    
    # Get R² score and calculate MAE on the test set
    predictions = cv_model.transform(test_data)
    r2_score = cv_model.avgMetrics[0]
    mae = mae_evaluator.evaluate(predictions)
    
    return (model_name, float(r2_score), float(mae))

# Evaluate models sequentially
results = []
for name, model in models.items():
    result = cross_validate_model(name, model, train_data, test_data)
    results.append(result)
    print(f"Model: {result[0]}, R²: {result[1]}, MAE: {result[2]}")

# Convert results to a DataFrame and display
schema = StructType([
    StructField("model_name", StringType(), True),
    StructField("r2_score", FloatType(), True),
    StructField("mae", FloatType(), True)
])
results_df = spark.createDataFrame(results, schema=schema)
results_df.show()

# Unpersist the cached data to free memory
sample_data.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ordinal_Encoding

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

# Define feature transformations
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]
assembler = VectorAssembler(inputCols=[col + "_index" for col in categorical_cols] + numerical_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")


# COMMAND ----------

from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
# Define models to evaluate
models = {
    'linear_reg': LinearRegression(featuresCol="scaledFeatures", labelCol=target_col),
    'decision_tree': DecisionTreeRegressor(featuresCol="scaledFeatures", labelCol=target_col),
    'random_forest': RandomForestRegressor(featuresCol="scaledFeatures", labelCol=target_col),
    'gradient_boosting': GBTRegressor(featuresCol="scaledFeatures", labelCol=target_col)
}

# COMMAND ----------

# Evaluation metrics
r2_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")
mae_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Define K-fold cross-validation function
def cross_validate_model(model_name, model, train_data, test_data):
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(stages=indexers + [assembler, scaler, model])
    
    # Set up 3-fold cross-validation
    param_grid = ParamGridBuilder().build()  # No hyperparameter tuning
    cross_val = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=r2_evaluator, numFolds=3, seed=42)
    
    # Fit cross-validator on training data
    cv_model = cross_val.fit(train_data)
    
    # Evaluate model on the test set
    predictions = cv_model.transform(test_data)
    r2_score = cv_model.avgMetrics[0]
    mae = mae_evaluator.evaluate(predictions)
    
    return (model_name, float(r2_score), float(mae))

# Evaluate each model and capture results
results = []
for name, model in models.items():
    print(f"Starting cross-validation for model: {name}")
    result = cross_validate_model(name, model, train_data, test_data)
    results.append(result)
    print(f"Model: {result[0]}, R²: {result[1]}, MAE: {result[2]}")

# Convert results to a DataFrame and display
schema = StructType([
    StructField("model_name", StringType(), True),
    StructField("r2_score", FloatType(), True),
    StructField("mae", FloatType(), True)
])
results_df = spark.createDataFrame(results, schema=schema)
results_df.show()

# Unpersist the cached data to free memory
train_data.unpersist()
test_data.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target_Encoding

# COMMAND ----------

train_data.printSchema()
test_data.printSchema()

# COMMAND ----------

sample_data.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# List of categorical columns for target mean encoding
categorical_cols = ['pickup_borough', 'dropoff_borough', 'distance_bin', 'time_of_day_bin']

# Target column
target_col = "total_amount"

# Dynamically build feature columns list by performing target mean encoding
feature_cols = []

# Add target mean encoding to train_data and test_data
for col in categorical_cols:
    # Calculate mean target for each category
    category_means = train_data.groupBy(col).agg(F.mean(target_col).alias(f"{col}_target_mean"))
    
    # Join the means to train and test datasets
    train_data = train_data.join(category_means, on=col, how="left")
    test_data = test_data.join(category_means, on=col, how="left")
    
    # Add the new encoded column to the list of feature columns
    feature_cols.append(f"{col}_target_mean")

numerical_cols = ['pickup_day_of_week','payment_type','trip_duration', 'pickup_hour', 'pickup_month','is_holiday','passenger_count', 'near_airport']
feature_cols.extend(numerical_cols)

# Cache the data for efficiency
train_data.cache()
test_data.cache()

# COMMAND ----------

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# COMMAND ----------

# Define models to evaluate
models = {
    'linear_reg': LinearRegression(featuresCol="scaledFeatures", labelCol=target_col),
    'decision_tree': DecisionTreeRegressor(featuresCol="scaledFeatures", labelCol=target_col),
    'random_forest': RandomForestRegressor(featuresCol="scaledFeatures", labelCol=target_col),
    'gradient_boosting': GBTRegressor(featuresCol="scaledFeatures", labelCol=target_col)
}

# COMMAND ----------

# Define evaluation metrics
r2_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")
mae_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")

# COMMAND ----------

# Cross-validation function
def cross_validate_model(model_name, model, train_data, test_data):
    # Create pipeline with assembler, scaler, and model
    pipeline = Pipeline(stages=[assembler, scaler, model])
    
    # Set up cross-validation
    param_grid = ParamGridBuilder().build()  # No hyperparameter tuning in this example
    cross_val = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=r2_evaluator, numFolds=3, seed=42)
    
    # Fit cross-validator on training data
    cv_model = cross_val.fit(train_data)
    
    # Evaluate model on test data
    predictions = cv_model.transform(test_data)
    r2_score = cv_model.avgMetrics[0]
    mae = mae_evaluator.evaluate(predictions)
    
    return (model_name, float(r2_score), float(mae))

# Evaluate all models
results = []
for name, model in models.items():
    print(f"Starting cross-validation for model: {name}")
    result = cross_validate_model(name, model, train_data, test_data)
    results.append(result)
    print(f"Model: {result[0]}, R²: {result[1]}, MAE: {result[2]}")

# Convert results to a DataFrame and display
schema = StructType([
    StructField("model_name", StringType(), True),
    StructField("r2_score", FloatType(), True),
    StructField("mae", FloatType(), True)
])
results_df = spark.createDataFrame(results, schema=schema)
results_df.show()

train_data.unpersist()
test_data.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## One_Hot_Encoding_with_PCA

# COMMAND ----------

# Define feature transformations
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]
encoders = [
    OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded", handleInvalid="error", dropLast=False) 
    for col in categorical_cols
]
assembler = VectorAssembler(inputCols=[col + "_encoded" for col in categorical_cols] + numerical_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA

# Apply PCA after scaling
pca = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures")  # Adjust k to the desired number of principal components

# COMMAND ----------

# Define models to evaluate, using `pcaFeatures` as input
models = {
    'linear_reg': LinearRegression(featuresCol="pcaFeatures", labelCol=target_col),
    'decision_tree': DecisionTreeRegressor(featuresCol="pcaFeatures", labelCol=target_col),
    'random_forest': RandomForestRegressor(featuresCol="pcaFeatures", labelCol=target_col),
    'gradient_boosting': GBTRegressor(featuresCol="pcaFeatures", labelCol=target_col)
}

# COMMAND ----------

# Evaluation metrics
r2_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")
mae_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")

# COMMAND ----------

# Define K-fold cross-validation function
def cross_validate_model(model_name, model, train_data, test_data):
    # Pipeline creation with One-Hot Encoding, PCA, and model
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, pca, model])
    
    # Set up 3-fold cross-validation
    param_grid = ParamGridBuilder().build()  # No hyperparameter tuning
    cross_val = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=r2_evaluator, numFolds=3, seed=42)
    
    # Fit cross-validator on training data
    cv_model = cross_val.fit(train_data)
    
    # Evaluate model on the test set
    predictions = cv_model.transform(test_data)
    r2_score = cv_model.avgMetrics[0]
    mae = mae_evaluator.evaluate(predictions)
    
    return (model_name, float(r2_score), float(mae))

# Evaluate each model and capture results
results = []
for name, model in models.items():
    print(f"Starting cross-validation for model: {name}")
    result = cross_validate_model(name, model, train_data, test_data)
    results.append(result)
    print(f"Model: {result[0]}, R²: {result[1]}, MAE: {result[2]}")

# Convert results to a DataFrame and display
schema = StructType([
    StructField("model_name", StringType(), True),
    StructField("r2_score", FloatType(), True),
    StructField("mae", FloatType(), True)
])
results_df = spark.createDataFrame(results, schema=schema)
results_df.show()

# Unpersist the cached data to free memory
train_data.unpersist()
test_data.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# Define the Linear Regression model
linear_reg_model = LinearRegression(featuresCol="scaledFeatures", labelCol=target_col)

# For Linear Regression
print("Processing Linear Regression...")
linear_reg_pipeline = Pipeline(stages=indexers + encoders + 
                               [assembler.setOutputCol("features"),
                                scaler.setInputCol("features").setOutputCol("scaledFeatures"), 
                                linear_reg_model])

# Define parameter grid for Linear Regression
param_grid_lr = ParamGridBuilder() \
    .addGrid(linear_reg_model.regParam, [0.1, 0.01, 0.001]) \
    .addGrid(linear_reg_model.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Cross-validation setup
cv_lr = CrossValidator(
    estimator=linear_reg_pipeline,
    estimatorParamMaps=param_grid_lr,
    evaluator=r2_evaluator,
    numFolds=3,
    seed=42
)

# Fit the cross-validation model
cv_lr_model = cv_lr.fit(train_data)

# Get the best model
lr_best_model = cv_lr_model.bestModel

# Make predictions on the test data
lr_predictions = lr_best_model.transform(test_data)

# Evaluate the model
lr_r2 = r2_evaluator.evaluate(lr_predictions)
lr_mae = mae_evaluator.evaluate(lr_predictions)

# Append results
results.append(("Linear Regression", lr_r2, lr_mae))


# COMMAND ----------

# Access the best Linear Regression model from the pipeline
best_lr_model = lr_best_model.stages[-1]

# Retrieve the best parameters for Linear Regression
best_reg_param_lr = best_lr_model.getRegParam()
best_elastic_net_param_lr = best_lr_model.getElasticNetParam()

# Print the best Linear Regression parameters
print(f"Best Linear Regression regParam: {best_reg_param_lr}")
print(f"Best Linear Regression elasticNetParam: {best_elastic_net_param_lr}")


# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

# Define the Decision Tree model
decision_tree_model = DecisionTreeRegressor(featuresCol="scaledFeatures", labelCol=target_col)

# For Decision Tree
print("Processing Decision Tree...")

# Create a pipeline specifically for Decision Tree without PCA
decision_tree_pipeline = Pipeline(stages=indexers + encoders + [
    assembler.setOutputCol("features"),
    scaler.setInputCol("features").setOutputCol("scaledFeatures"),
    decision_tree_model
])

# Define parameter grid for Decision Tree
param_grid_dt = ParamGridBuilder() \
    .addGrid(decision_tree_model.maxDepth, [5, 10, 20]) \
    .addGrid(decision_tree_model.maxBins, [32, 64]) \
    .build()

# Set up CrossValidator for Decision Tree
cv_dt = CrossValidator(
    estimator=decision_tree_pipeline,
    estimatorParamMaps=param_grid_dt,
    evaluator=r2_evaluator,
    numFolds=2,
    seed=42
)

# Train the model using CrossValidator
cv_dt_model = cv_dt.fit(train_data)

# Retrieve the best model from cross-validation
dt_best_model = cv_dt_model.bestModel

# Make predictions on the test data
dt_predictions = dt_best_model.transform(test_data)

# Evaluate the model using R² and MAE
dt_r2 = r2_evaluator.evaluate(dt_predictions)
dt_mae = mae_evaluator.evaluate(dt_predictions)

# Append results to the list
results.append(("Decision Tree", dt_r2, dt_mae))

# Print the results for the Decision Tree model
print(f"Decision Tree - R²: {dt_r2:.4f}, MAE: {dt_mae:.4f}")

# Unpersist the data after use
train_data.unpersist()
test_data.unpersist()

# COMMAND ----------

# Access the best hyperparameters from the cross-validation
best_max_depth = dt_best_model.stages[-1]._java_obj.getMaxDepth()
best_max_bins = dt_best_model.stages[-1]._java_obj.getMaxBins()

# Print the best hyperparameters
print(f"Best maxDepth: {best_max_depth}")
print(f"Best maxBins: {best_max_bins}")

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml import Pipeline

# Define the Random Forest Regressor model
random_forest_model = RandomForestRegressor(featuresCol="scaledFeatures", labelCol=target_col, seed=42)

print("Processing Random Forest Regressor...")

# Create a pipeline specifically for Random Forest without PCA
random_forest_pipeline = Pipeline(stages=indexers + encoders + [
    assembler.setOutputCol("features"),
    scaler.setInputCol("features").setOutputCol("scaledFeatures"),
    random_forest_model 
])

# Define a simplified parameter grid for Random Forest
param_grid_rf = ParamGridBuilder() \
    .addGrid(random_forest_model.maxDepth, [5, 10]) \
    .addGrid(random_forest_model.numTrees, [10, 25]) \
    .build()

# Set up TrainValidationSplit
train_val_rf = TrainValidationSplit(
    estimator=random_forest_pipeline,
    estimatorParamMaps=param_grid_rf,
    evaluator=r2_evaluator,
    trainRatio=0.8,
    seed=42
)

# Train the model using TrainValidationSplit
cv_rf_model = train_val_rf.fit(train_data)

# Retrieve the best model from TrainValidationSplit
rf_best_model = cv_rf_model.bestModel

# Make predictions on the test data
rf_predictions = rf_best_model.transform(test_data)

# Evaluate the model using R² and MAE
rf_r2 = r2_evaluator.evaluate(rf_predictions)
rf_mae = mae_evaluator.evaluate(rf_predictions)

# Print the results for the Random Forest Regressor model
print(f"Random Forest Regressor - R²: {rf_r2:.4f}, MAE: {rf_mae:.4f}")

# Append results to the list
results.append(("Random Forest Regressor", rf_r2, rf_mae))

# Unpersist the data after use
train_data.unpersist()
test_data.unpersist()

# COMMAND ----------

import os

# Define the model save path in Databricks' file system
local_path = "/dbfs/tmp/random_forest_model"

# Save the best Random Forest model
rf_best_model.write().overwrite().save(local_path)

print(f"Model saved locally at {local_path}")

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Example single query input data
single_query = [
    (1, 1, 20.37, 2, 8, 10, "Manhattan", "Brooklyn", 0, "Medium", "Morning", 0)
]

# Define the schema of your input data based on the model's feature columns
columns = [
    "passenger_count", "payment_type", "trip_duration", 
    "pickup_day_of_week", "pickup_hour", "pickup_month", "pickup_borough", 
    "dropoff_borough", "is_holiday", "distance_bin", "time_of_day_bin", "near_airport"
]

# Initialize SparkSession
spark = SparkSession.builder.appName("ModelPrediction").getOrCreate()

# Convert the input query to a DataFrame
test_data_df = spark.createDataFrame(single_query, columns)

# Show the test data to verify the schema
print("Test Data:")
test_data_df.show()

# COMMAND ----------

# Access the best Random Forest model from the pipeline
best_rf_model = rf_best_model.stages[-1]

# Retrieve the best parameters for Random Forest
best_max_depth_rf = best_rf_model.getMaxDepth()  # Correct usage: No parentheses
best_num_trees_rf = best_rf_model.getNumTrees

# Print the best Random Forest parameters
print(f"Best Random Forest maxDepth: {best_max_depth_rf}")
print(f"Best Random Forest numTrees: {best_num_trees_rf}")
print(f"Best Random Forest: {best_rf_model}")

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# Define the Gradient Boosted Tree Regressor model
gradient_boosting_model = GBTRegressor(featuresCol="scaledFeatures", labelCol=target_col, seed=42)

print("Processing Gradient Boosted Tree Regressor...")

# Cache train and test data
train_data.cache()
test_data.cache()

# Create a pipeline specifically for GBT
gradient_boosting_pipeline = Pipeline(stages=indexers + encoders + [
    assembler.setOutputCol("features"),
    scaler.setInputCol("features").setOutputCol("scaledFeatures"),
    gradient_boosting_model
])

# Define a simplified parameter grid for GBT
param_grid_gbt = ParamGridBuilder() \
    .addGrid(gradient_boosting_model.maxIter, [10, 25]) \
    .addGrid(gradient_boosting_model.maxDepth, [3, 5]) \
    .build()

# Set up TrainValidationSplit
train_val_gbt = TrainValidationSplit(
    estimator=gradient_boosting_pipeline,
    estimatorParamMaps=param_grid_gbt,
    evaluator=r2_evaluator,  # R² evaluator
    trainRatio=0.8,
    seed=42
)

# Train the model using TrainValidationSplit
cv_gbt_model = train_val_gbt.fit(train_data)

# Retrieve the best model from TrainValidationSplit
gbt_best_model = cv_gbt_model.bestModel

# Make predictions on the test data
gbt_predictions = gbt_best_model.transform(test_data)

# Evaluate the model using R² and MAE
gbt_r2 = r2_evaluator.evaluate(gbt_predictions)
gbt_mae = mae_evaluator.evaluate(gbt_predictions)

# Print the results for the Gradient Boosted Tree Regressor model
print(f"Gradient Boosted Tree Regressor - R²: {gbt_r2:.4f}, MAE: {gbt_mae:.4f}")

# Append results to the list (optional, if you need to store the results)
results.append(("Gradient Boosted Tree Regressor", gbt_r2, gbt_mae))

# Unpersist the data after use
train_data.unpersist()
test_data.unpersist()

# COMMAND ----------

# Access the best Gradient Boosted Tree model from the pipeline
best_gbt_model = cv_gbt_model.bestModel.stages[-1]

# Retrieve the best parameters for Gradient Boosted Tree
best_max_iter_gbt = best_gbt_model.getMaxIter()
best_max_depth_gbt = best_gbt_model.getMaxDepth()

# Print the best Gradient Boosted Tree parameters
print(f"Best Gradient Boosted Tree maxIter: {best_max_iter_gbt}")
print(f"Best Gradient Boosted Tree maxDepth: {best_max_depth_gbt}")

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Create a dictionary to store the best parameters for each model
best_params = {
    "Linear Regression": {
        "regParam": best_reg_param_lr,
        "elasticNetParam": best_elastic_net_param_lr
    },
    "Decision Tree": {
        "maxDepth": best_max_depth,
        "maxBins": best_max_bins
    },
    "Random Forest": {
        "maxDepth": best_max_depth_rf,
        "numTrees": best_num_trees_rf
    },
    "Gradient Boosted Tree": {
        "maxIter": best_max_iter_gbt,
        "maxDepth": best_max_depth_gbt
    }
}

# Store the evaluation results in a structured format
results = [
    Row(model_name="Linear Regression", hyperparameters=best_params["Linear Regression"], r2_score=lr_r2, mae=lr_mae),
    Row(model_name="Decision Tree", hyperparameters=best_params["Decision Tree"], r2_score=dt_r2, mae=dt_mae),
    Row(model_name="Random Forest", hyperparameters=best_params["Random Forest"], r2_score=rf_r2, mae=rf_mae),
    Row(model_name="Gradient Boosted Tree", hyperparameters=best_params["Gradient Boosted Tree"], r2_score=gbt_r2, mae=gbt_mae)
]

# Define the schema for the DataFrame
schema = StructType([
    StructField("model_name", StringType(), True),
    StructField("hyperparameters", StringType(), True),
    StructField("r2_score", FloatType(), True),
    StructField("mae", FloatType(), True)
])

# Convert the results list into a DataFrame
results_df = spark.createDataFrame(results, schema=schema)

# Show the results DataFrame
results_df.show(truncate=False)

# Unpersist cached data
train_data.unpersist()
test_data.unpersist()

# COMMAND ----------

import matplotlib.pyplot as plt

# Example dictionary of models and their evaluation metrics
models_results = {
    "Linear Regression": {"r2": lr_r2, "mae": lr_mae},
    "Decision Tree": {"r2": dt_r2, "mae": dt_mae},
    "Random Forest": {"r2": rf_r2, "mae": rf_mae},
    "Gradient Boosted Tree": {"r2": gbt_r2, "mae": gbt_mae}
}

# Extract model names, R² scores, and MAE scores from the dictionary
models = list(models_results.keys())
r2_scores = [models_results[model]["r2"] for model in models]
mae_scores = [models_results[model]["mae"] for model in models]

# Create a figure with subplots (side-by-side layout)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot R² scores
ax[0].bar(models, r2_scores, color='green')
ax[0].set_title("R² Scores")
ax[0].set_ylabel("R²")
ax[0].grid(True, axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability

# Plot MAE scores
ax[1].bar(models, mae_scores, color='red')
ax[1].set_title("MAE Scores")
ax[1].set_ylabel("MAE")
ax[1].grid(True, axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability

# Adjust layout to prevent overlap and improve spacing
plt.tight_layout()

# Show the plots
plt.show()
