import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


class DataScaler:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    # Creates scaler instances for MinMaxScaler, StandScaler, RobustScaler, QuantileTransformer, and PowerTransformer
    def create_scalers(self):
        # Create a dictionary of scaler instances
        scalers = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "QuantileTransformer": QuantileTransformer(
                n_quantiles=1000, output_distribution="uniform"
            ),
            # "PowerTransformer": PowerTransformer(
            #     method="yeo-johnson", standardize=True
            # ),
        }
        return scalers

    # Takes a scaler instance, a training set, and a test set as input,
    # fits the scaler on the training set, and transforms both the training and test sets using the fitted scaler:
    def scale_data(self, scaler):
        # Fit the scaler on the training set
        scaler.fit(self.x_train)

        # Transform the training set
        X_train_scaled = scaler.transform(self.x_train)

        # Transform the test set
        X_test_scaled = scaler.transform(self.x_test)

        # Create DataFrames using the scaled datasets
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=self.x_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=self.x_test.columns)

        # print(X_train_scaled_df)
        # print(X_test_scaled_df)

        return X_train_scaled_df, X_test_scaled_df

    def scale_datasets(self):
        # Create scaler instances
        scalers = self.create_scalers()

        # Initialize empty lists to store scaled training and test variable pairs
        scaled_train_pairs = []
        scaled_test_pairs = []
        scaler_names = []

        # Define the number of workers for parallel processing
        num_workers = multiprocessing.cpu_count()

        # Parallelize the scaling process using joblib
        results = Parallel(n_jobs=num_workers)(
            delayed(self.scale_data)(scaler_instance)
            for scaler_name, scaler_instance in scalers.items()
        )

        # Process the results
        for scaler_name, (X_train_scaled, X_test_scaled) in zip(
            scalers.keys(), results
        ):
            # Append scaled variable pairs to the respective lists
            scaled_train_pairs.append((X_train_scaled, self.y_train))
            scaled_test_pairs.append((X_test_scaled, self.y_test))

            # Append corresponding scaler names to a list
            scaler_names.append(scaler_name)

            print(f"Training set for {scaler_name}: {scaled_train_pairs}")
            print(f"Test set for {scaler_name}: {scaled_test_pairs}")

        # Return the lists of scaled variable pairs and the scaler names used
        return scaler_names, scaled_train_pairs, scaled_test_pairs
