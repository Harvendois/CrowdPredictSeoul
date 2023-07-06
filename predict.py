import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from joblib import load
import time
import json

def predict_population(predicted_hours):
    df_dong_codes = pd.read_csv('D:\CJH\CrowdPredict\dong_codes.csv')
    dong_codes = df_dong_codes['dong_code'].tolist()
    feature_names = ['date','DayOfWeek_Friday', 'DayOfWeek_Monday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'IsWeekend']
    current_date = pd.to_datetime(time.time(), unit='s')
    current_date = current_date.tz_localize('UTC').tz_convert('Asia/Seoul')
    current_hour = current_date.hour
    all_predictions = []

    # Iterate over each 'dong' code
    for dong_code in dong_codes:

        model = load(f'D:\CJH\CrowdPredict\joblib\{dong_code}.joblib')

        predictions = []

        for i in range(predicted_hours):
            new_data = pd.Series(0, index=feature_names)  # Initialize with zeros instead of NaNs
            new_data['date'] = (current_date + pd.DateOffset(hours=i)).strftime('%Y%m%d')
           
            # Set all hour features to 0
            for h in range(24):  # Change this to 24
                new_data['hour_' + str(h)] = 0
            new_data['hour_' + str((current_hour + i) % 24)] = 1  # Change this to 24

            # Set all DayOfWeek features to 0
            for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                new_data['DayOfWeek_' + dow] = 0

            # Determine the day of the week
            day_of_week = (current_date + pd.DateOffset(hours=i)).day_name()
            new_data['DayOfWeek_' + day_of_week] = 1

            # Determine whether it's a weekend
            new_data['IsWeekend'] = 1 if day_of_week in ['Saturday', 'Sunday'] else 0

            # Convert the Series to a DataFrame
            new_data_df = new_data.to_frame().transpose()

            # Make a prediction for the new data point
            new_pred = model.predict(new_data_df)
            predictions.append(new_pred.tolist())  # Convert numpy array to list


        all_predictions.append(predictions)

    return all_predictions

# Usage
# predicted_hours = 72
# all_predictions = predict_population(predicted_hours)
# print(f"Predicted population for the next {predicted_hours} hours for each 'dong': {len(all_predictions)}")

# Convert the list of lists to a JSON array
# all_predictions_json = json.dumps(all_predictions)

# Now you can send `all_predictions_json` to another server
