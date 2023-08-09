import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class OverUnderFitEvaluator:
    def __init__(self, metrics_df):
        self.metrics_df = metrics_df

    def evaluate(self):
        # Compute the differences between test and training metrics for each class
        class_labels = self.metrics_df["classifier_name"].unique()
        evaluation_results = []

        for class_label in class_labels:
            class_metrics_df = self.metrics_df[
                self.metrics_df["classifier_name"] == class_label
            ]

            diff_error = (
                class_metrics_df["Test Error"] - class_metrics_df["Training Error"]
            )
            diff_accuracy = (
                class_metrics_df["Accuracy"] - class_metrics_df["Training Accuracy"]
            )
            diff_precision = (
                class_metrics_df["Precision"] - class_metrics_df["Training Precision"]
            )
            diff_recall = (
                class_metrics_df["Recall"] - class_metrics_df["Training Recall"]
            )
            diff_f1_score = (
                class_metrics_df["F1-Score"] - class_metrics_df["Training F1-Score"]
            )
            diff_f1_score = np.nan_to_num(diff_f1_score, nan=0.0)
            diff_roc_auc = (
                class_metrics_df["ROC-AUC"] - class_metrics_df["Training ROC-AUC"]
            )
            diff_specificity = (
                class_metrics_df["Specificity"]
                - class_metrics_df["Training Specificity"]
            )

            evaluation_df = pd.DataFrame()
            evaluation_df["Test-Train Error Difference"] = diff_error
            evaluation_df["Test-Train Accuracy Difference"] = diff_accuracy
            evaluation_df["Test-Train Precision Difference"] = diff_precision
            evaluation_df["Test-Train Recall Difference"] = diff_recall
            evaluation_df["Test-Train F1-Score Difference"] = diff_f1_score
            evaluation_df["Test-Train ROC-AUC Difference"] = diff_roc_auc
            evaluation_df["Test-Train Specificity Difference"] = diff_specificity
            evaluation_df["Evaluation"] = np.where(
                (diff_error > 0)
                & (diff_accuracy > 0)
                & (diff_precision > 0)
                & (diff_recall > 0)
                & (diff_f1_score > 0)
                & (diff_roc_auc > 0)
                & (diff_specificity > 0),
                "Overfitting",
                np.where(
                    (diff_error > 0)
                    & (diff_accuracy < 0)
                    & (diff_precision < 0)
                    & (diff_recall < 0)
                    & (diff_f1_score < 0)
                    & (diff_roc_auc < 0)
                    & (diff_specificity < 0),
                    "Underfitting",
                    "Neither",
                ),
            )

            evaluation_results.append((class_label, evaluation_df))

        return evaluation_results

    def plot_evaluation(self, save_path):
        evaluation_results = self.evaluate()

        for i, (class_label, evaluation_df) in enumerate(evaluation_results):
            fig, ax = plt.subplots(figsize=(8, 6))

            ax.scatter(
                evaluation_df["Test-Train Accuracy Difference"],
                evaluation_df["Test-Train Error Difference"],
                c="blue",
                alpha=0.6,
                label="Accuracy",
            )

            # Scatter plot of Test-Train Error Difference vs Test-Train Precision Difference
            ax.scatter(
                evaluation_df["Test-Train Precision Difference"],
                evaluation_df["Test-Train Error Difference"],
                c="red",
                alpha=0.6,
                label="Precision",
            )

            # Scatter plot of Test-Train Error Difference vs Test-Train Recall Difference
            ax.scatter(
                evaluation_df["Test-Train Recall Difference"],
                evaluation_df["Test-Train Error Difference"],
                c="green",
                alpha=0.6,
                label="Recall",
            )

            # Scatter plot of Test-Train Error Difference vs Test-Train F1-Score Difference
            ax.scatter(
                evaluation_df["Test-Train F1-Score Difference"],
                evaluation_df["Test-Train Error Difference"],
                c="orange",
                alpha=0.6,
                label="F1-Score",
            )

            # Scatter plot of Test-Train Error Difference vs Test-Train ROC-AUC Difference
            ax.scatter(
                evaluation_df["Test-Train ROC-AUC Difference"],
                evaluation_df["Test-Train Error Difference"],
                c="purple",
                alpha=0.6,
                label="ROC-AUC",
            )

            # Scatter plot of Test-Train Error Difference vs Test-Train Specificity Difference
            ax.scatter(
                evaluation_df["Test-Train Specificity Difference"],
                evaluation_df["Test-Train Error Difference"],
                c="pink",
                alpha=0.6,
                label="Specificity",
            )

            ax.set_xlabel("Test-Train Difference")
            ax.set_ylabel("Test Error")
            ax.set_title(
                f"Overfitting and Underfitting Evaluation for Class: {class_label}"
            )
            ax.legend()
            ax.grid(True)

            # Save the plot as a JPEG file
            # Convert the relative path to an absolute path
            absolute_save_path = os.path.abspath(save_path)
            save_file_path = f"{absolute_save_path}_{class_label}.jpg"
            plt.savefig(save_file_path)

            plt.close(fig)
