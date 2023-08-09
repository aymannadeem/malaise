import random
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


class ModelEvaluator:
    def __init__(
        self,
        vectorizer_name,
        feature_selection_method,
        scaler_names,
        train_datasets,
        test_datasets,
    ):
        self.vectorizer_name = vectorizer_name
        self.feature_selection_method = feature_selection_method  # Feature selection
        self.scaler_names = scaler_names
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.confusion_matrices = []
        self.metrics_dataframes = []

    # Define the classifiers
    def create_classifiers(self):
        # Create a dictionary of classifier instances
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Gaussian Naive Bayes": GaussianNB(),
            "Perceptron": Perceptron(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(random_state=1),
        }
        return classifiers

    # Takes a classifier object and datasets as arguments and performs the necessary steps to fit the classifier to each dataset.
    def evaluate_classifiers(self):
        vectorizer_name = self.vectorizer_name
        feature_selection_method = self.feature_selection_method
        classifiers = self.create_classifiers()
        results = {}

        combined_metrics_df = pd.DataFrame(
            columns=[
                "Training Accuracy",
                "Accuracy",
                "Training Error",
                "Test Error",
                "Training Precision",
                "Precision",
                "Training Recall",
                "Recall",
                "Training Specificity",
                "Specificity",
                "Training F1-Score",
                "F1-Score",
                "Training ROC-AUC",
                "ROC-AUC",
                "classifier_name",
                "scaler_name",
                "vectorizer_name",
                "feature_selection_method",
            ]
        )

        def evaluate_classifier(
            classifier_name, classifier, train_dataset, test_dataset, scaler_name
        ):
            X_train, y_train = train_dataset
            X_test, y_test = test_dataset

            # Fit the classifier
            classifier.fit(X_train, y_train)

            # Plot learning curve
            self.plot_learning_curve(
                classifier,
                X_train,
                y_train,
                classifier_name,
                scaler_name,
                vectorizer_name,
                feature_selection_method,
            )

            # Make predictions
            y_train_pred = classifier.predict(X_train)
            y_pred = classifier.predict(X_test)

            # Calculate evaluation metrics
            train_confusion_mat = confusion_matrix(y_train, y_train_pred)
            confusion_mat = confusion_matrix(y_test, y_pred)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            accuracy = accuracy_score(y_test, y_pred)
            train_error = 1 - train_accuracy
            test_error = 1 - accuracy
            train_precision = precision_score(y_train, y_train_pred)
            precision = precision_score(y_test, y_pred)
            train_recall = recall_score(y_train, y_train_pred)
            recall = recall_score(y_test, y_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            f1 = f1_score(y_test, y_pred)
            train_roc_auc = roc_auc_score(y_train, y_train_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            # Extract true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP) from the confusion matrix
            train_tn, train_fp, train_fn, train_tp = train_confusion_mat.ravel()
            tn, fp, fn, tp = confusion_mat.ravel()
            # Calculate specificity
            train_specificity = train_tn / (train_tn + train_fp)
            specificity = tn / (tn + fp)

            if feature_selection_method == "none":
                self.compute_shap_values(
                    classifier,
                    X_train,
                    y_train,
                    classifier_name,
                    scaler_name,
                    vectorizer_name,
                    feature_selection_method,
                )

            # Return the results
            return (
                classifier_name,
                scaler_name,
                vectorizer_name,
                feature_selection_method,
                train_accuracy,
                accuracy,
                train_error,
                test_error,
                train_precision,
                precision,
                train_recall,
                recall,
                train_specificity,
                specificity,
                train_f1,
                f1,
                train_roc_auc,
                roc_auc,
                confusion_mat,
            )

        for classifier_name, classifier in classifiers.items():
            results[classifier_name] = {}

            evaluated_results = Parallel(n_jobs=-1)(
                delayed(evaluate_classifier)(
                    classifier_name,
                    classifier,
                    self.train_datasets[i],
                    self.test_datasets[i],
                    self.scaler_names[i],
                )
                for i in range(len(self.train_datasets))
            )

            for result in evaluated_results:
                (
                    classifier_name,
                    scaler_name,
                    vectorizer_name,
                    feature_selection_method,
                    train_accuracy,
                    accuracy,
                    train_error,
                    test_error,
                    train_precision,
                    precision,
                    train_recall,
                    recall,
                    train_specificity,
                    specificity,
                    train_f1,
                    f1,
                    train_roc_auc,
                    roc_auc,
                    confusion_mat,
                ) = result

                # Store the results
                results[classifier_name][scaler_name] = {
                    "Training Accuracy": train_accuracy,
                    "Accuracy": accuracy,
                    "Training Error": train_error,
                    "Test Error": test_error,
                    "Training Precision": train_precision,
                    "Precision": precision,
                    "Training Recall": train_recall,
                    "Recall": recall,
                    "Training Specificity": train_specificity,
                    "Specificity": specificity,
                    "Training F1-Score": train_f1,
                    "F1-Score": f1,
                    "Training ROC-AUC": train_roc_auc,
                    "ROC-AUC": roc_auc,
                    "Confusion Matrix": confusion_mat,
                }

                self.store_confusion_mat(
                    vectorizer_name,
                    confusion_mat,
                    classifier_name,
                    scaler_name,
                    feature_selection_method,
                )

                metrics_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Training Accuracy",
                            "Accuracy",
                            "Training Error",
                            "Test Error",
                            "Training Precision",
                            "Precision",
                            "Training Recall",
                            "Recall",
                            "Training Specificity",
                            "Specificity",
                            "Training F1-Score",
                            "F1-Score",
                            "Training ROC-AUC",
                            "ROC-AUC",
                        ],
                        "Score": [
                            train_accuracy,
                            accuracy,
                            train_error,
                            test_error,
                            train_precision,
                            precision,
                            train_recall,
                            recall,
                            train_specificity,
                            specificity,
                            train_f1,
                            f1,
                            train_roc_auc,
                            roc_auc,
                        ],
                    }
                )
                metrics_df = metrics_df.sort_values(by="Score", ascending=False)
                metrics_df = metrics_df.set_index("Score")
                metrics = self.format_metrics(
                    metrics_df,
                    vectorizer_name,
                    classifier_name,
                    scaler_name,
                    feature_selection_method,
                )
                self.metrics_dataframes.append(metrics)

                combined_metrics_df = pd.concat(
                    [
                        combined_metrics_df,
                        pd.DataFrame(
                            {
                                "Training Accuracy": train_accuracy,
                                "Accuracy": accuracy,
                                "Training Error": train_error,
                                "Test Error": test_error,
                                "Training Precision": train_precision,
                                "Precision": precision,
                                "Training Recall": train_recall,
                                "Recall": recall,
                                "Training Specificity": train_specificity,
                                "Specificity": specificity,
                                "Training F1-Score": train_f1,
                                "F1-Score": f1,
                                "Training ROC-AUC": train_roc_auc,
                                "ROC-AUC": roc_auc,
                                "classifier_name": classifier_name,
                                "scaler_name": scaler_name,
                                "vectorizer_name": vectorizer_name,
                                "feature_selection_method": feature_selection_method,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

        self.display_results()

        # Save the DataFrames as CSV tables
        combined_metrics_df.to_csv(
            f"/Users/aymannadeem/code/malware-detection/results/combined_metrics_{vectorizer_name}_{feature_selection_method}.csv",
            index=False,
        )

        return results, combined_metrics_df

    def store_confusion_mat(
        self,
        vectorizer_name,
        confusion_mat,
        classifier_name,
        scaler_name,
        feature_selection_method,
    ):
        self.confusion_matrices.append(
            (
                vectorizer_name,
                confusion_mat,
                classifier_name,
                scaler_name,
                feature_selection_method,
            )
        )

    def format_metrics(
        self,
        metrics_df,
        vectorizer_name,
        classifier_name,
        scaler_name,
        feature_selection_method,
    ):
        metrics_df_styled = metrics_df.style.set_table_attributes(
            "style='display:inline'"
        ).set_caption(
            f"{classifier_name} - {scaler_name} - {vectorizer_name} - {feature_selection_method}"
        )

        # Add CSS styling to align the "Metric" and "Score" columns
        metrics_df_styled = metrics_df_styled.set_table_styles(
            [
                {
                    "selector": "th.col_heading",
                    "props": [("padding", "5px 15px 5px 15px")],
                }
            ]
        )

        return metrics_df_styled

    # Generates a learning curve plot for a given classifier and dataset.
    # Uses the learning_curve function to calculate the training and cross-validation
    # scores for different training set sizes.
    # The learning curve plot provides insights into how the classifier's performance varies with the number of training examples. It helps to identify if the model is overfitting (high training accuracy but low cross-validation accuracy) or underfitting (low training accuracy and low cross-validation accuracy). It also provides an estimate of the model's generalization performance as the training set size increases.
    def plot_learning_curve(
        self,
        classifier,
        X,
        y,
        classifier_name,
        scaler_name,
        vectorizer_name,
        feature_selection_method,
    ):
        # The learning_curve function performs cross-validation by splitting the dataset
        # into multiple train-test splits and returns the training and test scores for each split.
        train_sizes, train_scores, test_scores = learning_curve(
            classifier,
            X,
            y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring="accuracy",
        )

        # Compute mean and standard deviation of training
        # and test scores across different train-test splits.
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.title(
            f"Learning Curve - {classifier_name} - {scaler_name} - {vectorizer_name} - {feature_selection_method}"
        )
        plt.xlabel("Training Examples")
        plt.ylabel("Accuracy")
        plt.grid()

        # Create two shaded regions to represent variances
        # in the training and test scores, respectively.
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(
            train_sizes,
            train_scores_mean,
            "o-",
            color="r",
            label="Training Accuracy",
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-Validation Accuracy",
        )
        plt.legend(loc="best")
        plt.tight_layout()

        # Save the figure as a JPEG image
        output_dir = "/Users/aymannadeem/code/malware-detection/results/fit"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"learning_curve_{vectorizer_name}_{classifier_name}_{scaler_name}_{feature_selection_method}.jpeg"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(
            f"Learning curve for {classifier_name} - {scaler_name} - {vectorizer_name} - {feature_selection_method} saved as a JPEG image."
        )

    def compute_shap_values(
        self,
        classifier,
        X_train,
        y_train,
        classifier_name,
        scaler_name,
        vectorizer_name,
        feature_selection_method,
    ):
        try:
            # Fit the classifier
            classifier.fit(X_train, y_train)

            # Create a callable function based on the classifier name
            predict_function = (
                lambda input_data: classifier.predict_proba(input_data)[:, 1]
                if classifier_name != "Perceptron"
                else classifier.predict(input_data)
            )

            # Determine the SHAP explainer based on the classifier name
            explainer = shap.Explainer(
                predict_function, X_train, algorithm="auto", n_jobs=2
            )

            # Disable the additivity check
            explainer.check_additivity = False

            # Calculate SHAP values for the sampled data
            shap_values = explainer.shap_values(X_train)

            # Create a DataFrame to store the SHAP values for the sampled data
            shap_df = pd.DataFrame(data=shap_values, columns=X_train.columns)

            # Sort the DataFrame by the absolute sum of SHAP values across instances
            shap_df["importance"] = shap_df.abs().sum(axis=0)
            shap_df = shap_df.sort_values(by="importance", ascending=False).drop(
                "importance", axis=1
            )

            # Save the SHAP values to a CSV file
            output_dir = "/Users/aymannadeem/code/malware-detection/results/shap"  # Specify the directory where you want to save the CSV file
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"shap_values_{vectorizer_name}_{classifier_name}_{scaler_name}_{feature_selection_method}.csv"
            output_path = os.path.join(output_dir, output_filename)
            shap_df.to_csv(output_path, index=False)

            # Plot SHAP summary plot
            title = f"SHAP Summary Plot - {classifier_name} - {scaler_name} - {vectorizer_name} - {feature_selection_method}"
            shap.summary_plot(
                shap_values,
                X_train,
                feature_names=X_train.columns.tolist(),
                class_names=["No Malware", "Malware"],
                show=False,
            )
            plt.title(title)

            # Save the figure as a JPEG image
            output_dir = "/Users/aymannadeem/code/malware-detection/results/shap"  # Specify the directory where you want to save the JPEG image
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"shap_summary_{vectorizer_name}_{classifier_name}_{scaler_name}_{feature_selection_method}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            print(
                f"SHAP summary plot and SHAP values for {classifier_name} - {scaler_name} - {vectorizer_name} - {feature_selection_method} saved as PNG image and CSV file."
            )

        except Exception as e:
            print(f"Error occurred while computing SHAP values: {str(e)}")

    def display_results(self):
        # Save results to jpeg
        output_dir = "/Users/aymannadeem/code/malware-detection/results"  # Specify the directory where you want to save the JPEG images
        os.makedirs(
            output_dir, exist_ok=True
        )  # Create the output directory if it doesn't exist

        for i, (
            vectorizer_name,
            confusion_mat,
            classifier_name,
            scaler_name,
            feature_selection_method,
        ) in enumerate(self.confusion_matrices):
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Plot the confusion matrix
            ax = axes[0]
            ax.matshow(confusion_mat, cmap=plt.cm.Oranges, alpha=0.3)
            for j in range(confusion_mat.shape[0]):
                for k in range(confusion_mat.shape[1]):
                    ax.text(
                        x=k,
                        y=j,
                        s=confusion_mat[j, k],
                        va="center",
                        ha="center",
                        size="xx-large",
                    )

            ax.set_xlabel("Predictions", fontsize=18)
            ax.set_ylabel("Actuals", fontsize=18)
            ax.set_title(
                f"Confusion Matrix for {classifier_name} - {scaler_name} - {vectorizer_name} - {feature_selection_method}",
                # f"Confusion Matrix for {classifier_name} - {scaler_name} - {vectorizer_name}",
                fontsize=18,
            )

            # Plot the metrics table
            ax = axes[1]
            ax.axis("off")
            metrics_html = self.metrics_dataframes[
                i
            ].to_html()  # Render the styled DataFrame as HTML
            metrics_df = pd.read_html(metrics_html)[
                0
            ]  # Convert the HTML table back to a DataFrame
            metrics_table = (
                metrics_df.values.tolist()
            )  # Convert the DataFrame to a list of lists

            ax.table(
                cellText=metrics_table,
                colLabels=metrics_df.columns,
                cellLoc="center",
                loc="center",
            )

            # Save the figure as a JPEG image
            output_filename = f"{vectorizer_name}_{classifier_name}_{scaler_name}_{feature_selection_method}.jpeg"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

        print("Results saved as JPEG images.")
