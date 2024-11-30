# NYC Taxi Fare Price Prediction  

## Capstone Project Overview  

This capstone project focuses on applying advanced data science techniques to predict taxi fare prices in New York City. By leveraging large-scale datasets, this project generates accurate fare predictions based on various factors such as pickup/dropoff locations, time, passenger count, and external variables like weather conditions. The project follows a structured approach, from data ingestion and preprocessing to model development and deployment. The final output is an interactive, user-friendly web application that provides real-time fare predictions, offering insights into how different factors influence taxi fares in NYC.  

---

## Data Preparation and Cleaning  

The project began with a thorough cleaning and consolidation of the NYC taxi fare dataset, integrating trip data with external sources such as weather conditions and public holidays.  

Key steps included:  
- Handling missing values.  
- Standardizing location coordinates (latitude and longitude) for both pickup and dropoff points.  
- Resolving data inconsistencies.  
- Detecting outliers.  
- Merging taxi zone data using `LocationID` for accurate mapping of locations.  

These processes were crucial for ensuring data quality and preparing the dataset for reliable and accurate analysis in the subsequent phases.  

---

## Feature Engineering  

The dataset was enriched by engineering new features that captured intricate details about NYC taxi trips:  

- **Trip Distance Bins**:  
   - The `trip_distance` feature was categorized into Short, Medium, and Long trips, helping to identify fare patterns across varying trip lengths.  

- **Temperature Ranges**:  
   - The `temp` feature was binned into Cold, Cool, Warm, and Hot categories, capturing the impact of weather on taxi demand and fares.  

- **Time of Day Categories**:  
   - The `pickup_hour` feature was segmented into Late Night, Morning, Afternoon, and Evening bins to understand fare variations by time of day.  

- **Seasonal Classifications**:  
   - Trips were classified into Winter, Spring, Summer, and Fall based on the pickup date, providing insights into seasonal demand fluctuations.  

- **Proximity to Airports**:  
   - A binary `near_airport` feature was introduced to indicate whether the pickup or dropoff occurred near an airport zone, reflecting the influence of proximity to major transportation hubs on fare prices.  

---

## Feature Selection and Importance Analysis  

To ensure the most impactful variables were used for fare prediction, a comprehensive feature importance analysis was conducted.  

Techniques included:  
- Random Forest.  
- Gradient Boosted Trees.  
- Lasso Regression.  
- Recursive Feature Elimination (RFE).  
- SHAP values.  

### Results:  
- A unified DataFrame was created by merging the results, and the average importance of each feature was computed.  
- Key features identified included passenger count, payment type, trip duration, pickup day of the week, pickup hour, pickup month, pickup and dropoff boroughs, holiday indicators, distance bins, time-of-day classifications, and proximity to airports.  
- The dataset was reduced to 13 essential features to streamline modeling and enhance computational efficiency.  
- The final dataset consisted of **35,864,766 rows and 13 columns**, ensuring all necessary details for accurate fare predictions.

  ![IMG_20241130_125220](https://github.com/user-attachments/assets/16f15422-91cc-431f-982d-611409b6db6c) , ![IMG_20241130_123405](https://github.com/user-attachments/assets/e608ae83-181a-4de0-a98a-eb15548be379)
---

## Exploratory Data Analysis (EDA)  

### Univariate Analysis  
- Analyzed distributions of `fare_amount`, `total_amount`, `passenger_count`, and `trip_distance` for skewness and outliers.  
- Binned `trip_duration` into very low, low, and average categories for better interpretability.  
- Addressed missing values, duplicates, and outliers to ensure data integrity.  

### Geospatial Analysis  
- Explored pickup and dropoff boroughs to identify high-demand areas.  
- Analyzed latitude and longitude with respect to `total_amount`, removing geographical outliers.  

### Temporal Patterns  
- Studied average trip distances by day of the week and weekend vs weekday distributions.  
- Examined monthly trip distributions, hourly pickup/dropoff trends, and zone-based insights.  
- Visualized fare trends over different times and locations.  

### Multivariate Analysis  
- Investigated the relationship between trip distance, trip duration, and fare amount.  
- Analyzed the effect of payment types on fares and passenger counts.  

### Weather and Holiday Impacts  
- Explored the influence of temperature on fares.  
- Analyzed holiday trip patterns to identify unique travel behaviors.  

This structured EDA phase revealed critical insights into the dataset’s structure, trends, and relationships, paving the way for robust feature engineering and model development.  

---

## Algorithms for NYC Taxi Fare Prediction  

Several regression models were evaluated for their ability to predict NYC taxi fares, focusing on accuracy, generalization capability, and computational efficiency.  

### Models Evaluated  

1. **Linear Regression**:  
   - A simple yet effective baseline model that captures linear relationships between input features and taxi fares.  

2. **Decision Tree Regressor**:  
   - A tree-based model that splits data into decision nodes based on feature thresholds to predict fares.  

3. **Random Forest Regressor**:  
   - An ensemble method combining multiple decision trees to improve prediction accuracy and reduce overfitting.  

4. **Gradient Boosted Trees (GBT) Regressor**:  
   - A sequential ensemble technique that builds decision trees iteratively, optimizing for reduced errors.  

### Performance Metrics  
- Root Mean Squared Error (RMSE).  
- Mean Absolute Error (MAE).  
- R² Score.  

### Hyperparameter Tuning  
- Linear Regression: Regularization parameters (e.g., L1/L2).  
- Decision Tree: Max depth, minimum samples split.  
- Random Forest: Number of trees, max depth, and feature sampling.  
- Gradient Boosted Trees: Learning rate, number of iterations, tree depth.  

---

## Pipeline Development and Deployment  

### Full Pipeline Steps  

1. **Data Ingestion and Preprocessing**:  
   - Processed large-scale NYC Taxi trip datasets using Databricks and PySpark.  
   - Integrated weather data (via Visual Crossing API) and public holiday calendars.  

2. **Feature Engineering**:  
   - Extracted and added spatial features using GeoPandas.  

3. **Model Training and Evaluation**:  
   - Scaled training using Databricks clusters for distributed computation.  

4. **Deployment**:  
   - Integrated the trained model into a Streamlit-based web application.  
   - Hosted the application on Azure for global accessibility.  

---

## Analytics Module  

An interactive analytics dashboard was built to provide insights into NYC taxi fare trends.  

Key visualizations included:  
- **Heatmaps**: Visualizing fare distribution across NYC boroughs.  
- **Scatter Plots**: Displaying correlations between trip distances, durations, and fares.  
- **Box Plots**: Identifying fare distributions by passenger counts and trip types.  
- **Line Plots**: Analyzing fare trends over time.  
- **Weather Analysis**: Understanding the impact of weather conditions on fares.  
- **Pie Charts**: Showing trip proportions by payment type and passenger count.  

---

## Technologies Used  

### Big Data Processing  
- PySpark, Databricks, Delta Lake  

### Machine Learning  
- Scikit-learn, XGBoost, LightGBM, Azure ML Studio  

### Visualization  
- Streamlit, Matplotlib, Seaborn, Plotly  

### Data Integration  
- Visual Crossing API, GeoPandas  

### Deployment  
- Azure App Service, Databricks Clusters, Docker  

### Version Control  
- GitHub  
  
