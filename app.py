# Step 2: Prediction
if submitted:
    # Map categorical inputs to numerical values
    payment_type_map = {"Card": 1, "Cash": 2}
    is_holiday_map = {"Yes": 1, "No": 0}
    pickup_day_of_week_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    near_airport_map = {"Yes": 1, "No": 0}  # Map near_airport correctly to 1/0

    # Prepare the single query
    single_query = [(
        passenger_count,
        map_categorical_inputs(payment_type, payment_type_map),
        trip_duration,
        map_categorical_inputs(pickup_day_of_week, pickup_day_of_week_map),
        pickup_hour,
        pickup_month,
        pickup_borough,
        dropoff_borough,
        map_categorical_inputs(is_holiday, is_holiday_map),
        distance_bin,
        time_of_day_bin,
        map_categorical_inputs(near_airport, near_airport_map)  # Ensure near_airport is mapped properly
    )]

    columns = [
        "passenger_count", "payment_type", "trip_duration", "pickup_day_of_week",
        "pickup_hour", "pickup_month", "pickup_borough", "dropoff_borough",
        "is_holiday", "distance_bin", "time_of_day_bin", "near_airport"
    ]

    try:
        # Create a DataFrame with the mapped values, ensure proper types
        test_data_df = spark.createDataFrame(single_query, columns)

        # Ensure that all categorical columns are cast to the correct type if needed
        test_data_df = test_data_df.withColumn("payment_type", test_data_df["payment_type"].cast("int"))
        test_data_df = test_data_df.withColumn("is_holiday", test_data_df["is_holiday"].cast("int"))
        test_data_df = test_data_df.withColumn("pickup_day_of_week", test_data_df["pickup_day_of_week"].cast("int"))
        test_data_df = test_data_df.withColumn("near_airport", test_data_df["near_airport"].cast("int"))

        predictions = model.transform(test_data_df)
        prediction_result = predictions.select("prediction").collect()

        # Display the prediction
        st.write("### Prediction Result:")
        for row in prediction_result:
            st.write(f"Predicted Value: {row['prediction']}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
