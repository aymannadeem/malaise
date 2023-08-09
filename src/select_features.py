from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd


class FeatureSelector:
    def __init__(self, scaler_names, scaled_train_pairs, scaled_test_pairs):
        self.scaler_names = scaler_names
        self.scaled_train_pairs = scaled_train_pairs
        self.scaled_test_pairs = scaled_test_pairs

    def select_features(self, method):
        scaler_names = []
        train_scaled = []
        test_scaled = []

        for scaler_name, scaled_train_pair, scaled_test_pair in zip(
            self.scaler_names, self.scaled_train_pairs, self.scaled_test_pairs
        ):
            X_train_scaled, Y_train = scaled_train_pair

            # Convert X_train_scaled to a DataFrame
            X_train_scaled = pd.DataFrame(X_train_scaled)

            # Apply absolute value transformation to handle negative inputs
            X_train_scaled = np.abs(X_train_scaled)

            # Apply VarianceThreshold to remove constant features
            constant_filter = VarianceThreshold()
            X_train_filtered = constant_filter.fit_transform(X_train_scaled)

            # Get the indices of selected features
            selected_feature_indices = constant_filter.get_support(indices=True)

            # Retrieve the selected feature names
            selected_feature_names = X_train_scaled.columns[
                selected_feature_indices
            ].tolist()
            # print(f"Selected features for scaler {scaler_name}:")
            # print(selected_feature_names)

            # Perform feature selection
            if method == "SelectKBest":
                selector = SelectKBest(score_func=f_classif, k=10)
            elif method == "Chi-Squared":
                selector = SelectKBest(score_func=chi2, k=10)
            elif method == "Mutual Information":
                selector = SelectKBest(score_func=mutual_info_classif, k=10)
            else:
                raise ValueError(f"Invalid feature selection method: {method}")

            X_train_selected = selector.fit_transform(X_train_filtered, Y_train)

            # Apply the same feature selection to the test set
            X_test_scaled, Y_test = scaled_test_pair
            X_test_scaled = np.abs(X_test_scaled)
            X_test_scaled = pd.DataFrame(X_test_scaled)
            X_test_filtered = constant_filter.transform(X_test_scaled)
            X_test_selected = selector.transform(X_test_filtered)

            # Store the selected features, scaler names, and scaled data
            scaler_names.append(scaler_name)
            train_scaled.append((X_train_selected, Y_train))
            test_scaled.append((X_test_selected, Y_test))

        return scaler_names, train_scaled, test_scaled
