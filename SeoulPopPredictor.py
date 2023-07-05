# this class is based on predictVersion2.


import sys
print(sys.executable)

class SeoulPopPredictor:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    def __init__(self, dong_code):
        self.dong_code = dong_code
        self.model = None

    def prepare_data(self):
        # Read the TSV file
        df = pd.read_csv('dong.csv', delimiter=',', encoding='euc-kr')

        # Read the 2nd TSV file
        df2 = pd.read_csv('dong2.csv', delimiter=',', encoding='utf-8')

        # Append the additional data to the existing data
        df = pd.concat([df, df2], ignore_index=True)

        # Select the required columns
        df = df[['기준일ID', '시간대구분', '총생활인구수', '행정동코드']]

        # Rename the columns for easier understanding
        df.rename(columns={
            '기준일ID': 'date',
            '시간대구분': 'hour',
            '총생활인구수': 'total_population',
            '행정동코드': 'dong_code'
        }, inplace=True)

        # Filter the dataframe based on the 'dong' code
        df = df[df['dong_code'].astype(str) == self.dong_code]

        # Convert the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        # Convert 'date' column to Unix timestamp (int64 first then division)
        df['date'] = df['date'].astype('int64') // 10**9

        # Extract the day of the week and create a new column 'DayOfWeek'
        df['DayOfWeek'] = pd.to_datetime(df['date'], unit='s').dt.day_name()

        # One-hot encode the 'DayOfWeek' column
        df = pd.get_dummies(df, columns=['DayOfWeek'])

        # One-hot encode the 'hour' column
        df = pd.get_dummies(df, columns=['hour'])

        # Create a new column 'IsWeekend'
        df['IsWeekend'] = ((df['DayOfWeek_Saturday'] == 1) | (df['DayOfWeek_Sunday'] == 1)).astype(int)

        # Convert 'date' column back to datetime format
        df['date'] = pd.to_datetime(df['date'], unit='s')

        # Define the specific holidays
        holidays = ['2023-01-01','2023-01-23','2023-01-24','2023-03-01','2023-05-05', '2023-05-29', '2023-06-06']

        # Update 'IsWeekend' column to include the holidays
        df.loc[df['date'].dt.strftime('%Y-%m-%d').isin(holidays), 'IsWeekend'] = 1

        # Convert 'date' column back to Unix timestamp
        df['date'] = df['date'].astype('int64') // 10**9

        return df

    def train(self):
        df = self.prepare_data()

        # -------------- splitting train and test dataset --------------

        from sklearn.model_selection import train_test_split

        # Our target variable is 'total_population'
        X = df.drop('total_population', axis=1)
        y = df['total_population']

        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

        # -------------- hyperparameter tuning --------------

        from sklearn.model_selection import GridSearchCV
        from sklearn.tree import DecisionTreeRegressor

        # Define the parameter grid
        param_grid = {
            'max_depth': [10, 20, 30, 40, 50],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'criterion': ['absolute_error','friedman_mse','squared_error']
        }

        # Create a base model
        dt = DecisionTreeRegressor(random_state=47)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, 
                                cv=3, n_jobs=-1, verbose=2)

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_

        print("Best parameters: ", best_params)

        # -------------- training the model --------------

        from sklearn.model_selection import cross_val_score
        import numpy as np
        from sklearn.metrics import mean_squared_error
        from joblib import dump, load

        # Train the final model with the best parameters found
        model = DecisionTreeRegressor(max_depth=best_params['max_depth'], 
                                    criterion=best_params['criterion'])

        # Train the model using the training data
        trained_model = model.fit(X_train, y_train)

        # -------------- save model --------------

        # At the end of training, assign the trained model to self.model
        self.model = trained_model

        # Save the model
        joblib.dump(self.model, f'model_{self.dong_code}.joblib')

    def predict(self, new_data):
        if self.model is None:
            raise Exception("Model not trained yet. Call the 'train' method before predicting.")

        model = self.model
        import time
        # Get the feature names from the training data
        feature_names = X_train.columns.tolist()

        # Create a list to store the predictions
        predictions = []

        # Get the current date and hour
        current_date = pd.to_datetime(time.time(), unit='s')
        current_date = current_date.tz_localize('UTC').tz_convert('Asia/Seoul')
        current_hour = current_date.hour
        print(current_hour)
        print(current_date)

        predicted_hours = 72
        # Create new data points for the next x hours
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
            predictions.append(new_pred)

        print(f"Predicted population for the next {predicted_hours} hours: {predictions}")


        return predictions

# predictor = SeoulPopPredictor('11545510')  # Replace '11545510' with your dong code
# df = predictor.prepare_data()
# print(df.head())  # Print the first few rows of the dataframe to check if it's correct
