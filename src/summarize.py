import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import precision_recall_curve


# Summarizes the performance metrics of different models
class PerformanceSummarizer:
    def __init__(self, metrics_df):
        self.metrics_df = metrics_df
        # TODO: The code assumes that the input metrics_df DataFrame is correctly formatted
        # and contains the necessary columns. It would be beneficial to add some error handling
        # or validation code to check the DataFrame's structure and column names before performing
        # calculations to avoid potential errors or unexpected behavior.

    # Computes summary statistics (mean, median, max, min, variance, and standard deviation)
    # for each metric in the metrics_df DataFrame.
    # Returns a new DataFrame with stats for each metric.
    def aggregate_metrics(self):
        average_scores = self.metrics_df.mean(axis=0, skipna=True, numeric_only=True)
        median_scores = self.metrics_df.median(axis=0, skipna=True, numeric_only=True)
        max_scores = self.metrics_df.max(numeric_only=True)
        min_scores = self.metrics_df.min(numeric_only=True)
        var_scores = self.metrics_df.var(numeric_only=True)
        std_scores = self.metrics_df.std(numeric_only=True)

        mean_df = pd.DataFrame(average_scores, columns=["mean"])
        median_df = pd.DataFrame(median_scores, columns=["median"])
        max_df = pd.DataFrame(max_scores, columns=["max"])
        min_df = pd.DataFrame(min_scores, columns=["min"])
        var_df = pd.DataFrame(var_scores, columns=["var"])
        std_df = pd.DataFrame(std_scores, columns=["std"])

        summary_stats = pd.concat(
            [mean_df, median_df, max_df, min_df, var_df, std_df], axis=1
        )
        return summary_stats

    # Rank models based on a specific metric and low test error.
    # Groups metrics_df by classifier_name, vectorizer_name, and scaler_name,
    # and computes the mean, median, and maximum ranks of the specified metric and error metric for each group.
    # Models are ranked in descending order of metric and ascending order of error.
    def rank_models(self, metric, error_metric):
        ranked_models_mean = (
            self.metrics_df.groupby(
                [
                    "classifier_name",
                    "vectorizer_name",
                    "scaler_name",
                    "feature_selection_method",
                ]
            )
            .apply(
                lambda x: (
                    x[metric].mean(numeric_only=True),
                    x[error_metric].mean(numeric_only=True),
                )
            )
            .rank(ascending=[False, True])
        )
        ranked_models_mean = pd.DataFrame(ranked_models_mean)

        ranked_models_median = (
            self.metrics_df.groupby(
                [
                    "classifier_name",
                    "vectorizer_name",
                    "scaler_name",
                    "feature_selection_method",
                ]
            )
            .apply(
                lambda x: (
                    x[metric].median(numeric_only=True),
                    x[error_metric].median(numeric_only=True),
                )
            )
            .rank(ascending=[False, True])
        )
        ranked_models_median = pd.DataFrame(ranked_models_median)

        ranked_models_max = (
            self.metrics_df.groupby(
                [
                    "classifier_name",
                    "vectorizer_name",
                    "scaler_name",
                    "feature_selection_method",
                ]
            )
            .apply(
                lambda x: (
                    x[metric].max(numeric_only=True),
                    x[error_metric].max(numeric_only=True),
                )
            )
            .rank(ascending=[False, True])
        )
        ranked_models_max = pd.DataFrame(ranked_models_max)

        return ranked_models_mean, ranked_models_median, ranked_models_max

    #     def rank_models_performance(self, metric):
    #         grouped = self.metrics_df.groupby(["classifier_name", "vectorizer_name", "scaler_name"])
    #         ranked_models_perf_mean = grouped[metric].mean().rank(ascending=False)
    #         ranked_models_perf_median = grouped[metric].median().rank(ascending=False)
    #         ranked_models_perf_max = grouped[metric].max().rank(ascending=False)

    #         return ranked_models_perf_mean, ranked_models_perf_median, ranked_models_perf_max

    #     def rank_models_error(self, error_metric):
    #         grouped = self.metrics_df.groupby(["classifier_name", "vectorizer_name", "scaler_name"])
    #         ranked_models_error_mean = grouped[error_metric].mean().rank(ascending=False)
    #         ranked_models_error_median = grouped[error_metric].median().rank(ascending=False)
    #         ranked_models_error_max = grouped[error_metric].max().rank(ascending=False)

    #         return ranked_models_error_mean, ranked_models_error_median, ranked_models_error_max

    # Aggregate ranks across multiple metrics.
    # Takes ranked metrics and calculates the mean, median, and max rank
    # for each model across metrics.
    def aggregate_ranks(self, ranking_metrics):
        aggregated_ranks_mean = ranking_metrics.mean(axis=1, numeric_only=True)
        aggregated_ranks_median = ranking_metrics.median(axis=1, numeric_only=True)
        aggregated_ranks_max = ranking_metrics.max(axis=1, numeric_only=True)

        aggregated_ranks_mean = pd.DataFrame(
            {"Aggregated Rank Mean": aggregated_ranks_mean}
        )
        aggregated_ranks_median = pd.DataFrame(
            {"Aggregated Rank Median": aggregated_ranks_median}
        )
        aggregated_ranks_max = pd.DataFrame(
            {"Aggregated Rank Max": aggregated_ranks_max}
        )

        return aggregated_ranks_mean, aggregated_ranks_median, aggregated_ranks_max

    # Finds model with lowest aggregated rank.
    def find_best_model(self, aggregated_ranks):
        best_model_index = aggregated_ranks.astype(float).idxmin()
        best_model = self.metrics_df.loc[best_model_index, :]
        best_model = pd.DataFrame(best_model)
        return best_model

    # Finds model with highest aggregated rank.
    def find_worst_model(self, aggregated_ranks):
        worst_model_index = aggregated_ranks.astype(float).idxmax()
        worst_model = self.metrics_df.loc[worst_model_index, :]
        worst_model = pd.DataFrame(worst_model)
        return worst_model

    def save_result_as_csv(self, result, filename):
        df = pd.DataFrame({f"{filename}": [result]})
        df.to_csv(
            f"/Users/aymannadeem/code/malware-detection/results/performance_summaries/{filename}.csv",
            index=False,
        )
        return df

    def plot_summary_per_classifier(self):
        # Specify the directory to save the JPEG files
        output_directory = "/Users/aymannadeem/code/malware-detection/results/performance_summaries/plots/"

        # Define colors for mean, median, max, and min values
        colors = ["blue", "orange", "green", "red"]

        # Group the metrics dataframe by 'classifier_name'
        grouped_df = self.metrics_df.groupby("classifier_name")

        # Iterate over each group and plot the statistics
        for classifier_name, group in grouped_df:
            # Compute the mean, median, max, and min of the metrics for the current classifier_name
            metric_stats = pd.concat(
                [
                    group.mean(numeric_only=True).loc[
                        [
                            "Test Error",
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "Specificity",
                            "F1-Score",
                            "ROC-AUC",
                        ]
                    ],
                    group.median(numeric_only=True).loc[
                        [
                            "Test Error",
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "Specificity",
                            "F1-Score",
                            "ROC-AUC",
                        ]
                    ],
                    group.max(numeric_only=True).loc[
                        [
                            "Test Error",
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "Specificity",
                            "F1-Score",
                            "ROC-AUC",
                        ]
                    ],
                    group.min(numeric_only=True).loc[
                        [
                            "Test Error",
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "Specificity",
                            "F1-Score",
                            "ROC-AUC",
                        ]
                    ],
                ]
            )

            # Create a new figure and axes
            fig, ax = plt.subplots()

            # Plot the statistics as a bar chart with different colors
            metric_stats.plot(kind="bar", ax=ax, color=colors)

            # Set the labels and title
            ax.set_xlabel("Metric")
            ax.set_ylabel("Value")
            ax.set_title(f"Statistics for {classifier_name}")

            # Create a legend for the colors
            # legend_labels = ["Mean", "Median", "Max", "Min"]
            # legend_patches = [
            #     mpatches.Patch(color=color, label=label)
            #     for color, label in zip(colors, legend_labels)
            # ]
            # ax.legend(
            #     handles=legend_patches, loc="upper right", bbox_to_anchor=(1.2, 1)
            # )

            # plt.show()

            # Save the figure as a JPEG file
            output_file = f"{output_directory}_{classifier_name}_statistics.jpg"
            plt.savefig(output_file, format="jpeg", bbox_inches="tight")

            # Close the figure to release memory
            plt.close(fig)

    def plot_summary_per_metric(self):
        # Specify the directory to save the JPEG files
        output_directory = "/Users/aymannadeem/code/malware-detection/results/performance_summaries/plots/"

        # Define colors for each metric
        colors = ["blue", "red", "green", "purple", "orange", "brown", "cyan"]

        # Group the metrics dataframe by 'classifier_name'
        grouped_df = self.metrics_df.groupby("classifier_name")

        # Compute the mean, median, max, and min of the metrics for each classifier
        mean_metrics = grouped_df.mean(numeric_only=True)
        median_metrics = grouped_df.median(numeric_only=True)
        max_metrics = grouped_df.max(numeric_only=True)
        min_metrics = grouped_df.min(numeric_only=True)

        # Plot and save the mean metrics
        self.plot_metrics(mean_metrics, "Mean Metrics", output_directory, colors)

        # Plot and save the median metrics
        self.plot_metrics(median_metrics, "Median Metrics", output_directory, colors)

        # Plot and save the max metrics
        self.plot_metrics(max_metrics, "Max Metrics", output_directory, colors)

        # Plot and save the min metrics
        self.plot_metrics(min_metrics, "Min Metrics", output_directory, colors)

    def plot_metrics(self, metrics, title, output_directory, colors):
        # Create a new figure and axes
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot the metrics as a bar chart with different colors
        metrics.plot(kind="bar", ax=ax, color=colors)

        # Set the labels and title
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Value")
        ax.set_title(title)

        # Create a legend for the colors and metrics
        legend_labels = metrics.columns
        legend_patches = [
            mpatches.Patch(color=color, label=label)
            for color, label in zip(colors, legend_labels)
        ]
        ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.2, 1))

        # plt.show()

        # Save the figure as a JPEG file
        output_file = f"{output_directory}{title.lower().replace(' ', '_')}.jpg"
        plt.savefig(output_file, format="jpeg", bbox_inches="tight")

        # Close the figure to release memory
        plt.close(fig)

    def plot_precision_recall_curves(self):
        # Specify the directory to save the JPEG files
        output_directory = "/Users/aymannadeem/code/malware-detection/results/performance_summaries/plots/"

        # Group the metrics dataframe by 'classifier_name'
        grouped_df = self.metrics_df.groupby("classifier_name")

        # Iterate over each group and plot the precision-recall curve
        for classifier_name, group in grouped_df:
            # Compute precision and recall for the current classifier_name
            precision, recall, _ = precision_recall_curve(
                group["Actual"], group["Predicted"]
            )

            # Create a new figure and axes
            fig, ax = plt.subplots()

            # Plot the precision-recall curve
            ax.plot(recall, precision)

            # Set the labels and title
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curve for {classifier_name}")

            plt.show()

            # Save the figure as a JPEG file

    #             output_file = f"{output_directory}_{classifier_name}_precision_recall_curve.jpg"
    #             plt.savefig(output_file, format="jpeg", bbox_inches="tight")

    #             # Close the figure to release memory
    #             plt.close(fig)

    def evaluate_metrics(self):
        # Calculate mean, median, max, min, var, and std for each metric
        stats = self.aggregate_metrics()

        # Rank models based on a specific metric and low test error
        (
            ranked_by_accuracy_mean,
            ranked_by_accuracy_median,
            ranked_by_accuracy_max,
        ) = self.rank_models("Accuracy", "Test Error")
        (
            ranked_by_precision_mean,
            ranked_by_precision_median,
            ranked_by_precision_max,
        ) = self.rank_models("Precision", "Test Error")
        (
            ranked_by_recall_mean,
            ranked_by_recall_median,
            ranked_by_recall_max,
        ) = self.rank_models("Recall", "Test Error")
        (
            ranked_by_specificity_mean,
            ranked_by_specificity_median,
            ranked_by_specificity_max,
        ) = self.rank_models("Specificity", "Test Error")
        ranked_by_f1_mean, ranked_by_f1_median, ranked_by_f1_max = self.rank_models(
            "F1-Score", "Test Error"
        )
        (
            ranked_by_roc_auc_mean,
            ranked_by_roc_auc_median,
            ranked_by_roc_auc_max,
        ) = self.rank_models("ROC-AUC", "Test Error")

        metrics_ranked_by_mean = [
            ranked_by_accuracy_mean,
            ranked_by_precision_mean,
            ranked_by_recall_mean,
            ranked_by_specificity_mean,
            ranked_by_f1_mean,
            ranked_by_roc_auc_mean,
        ]

        metrics_ranked_by_median = [
            ranked_by_accuracy_median,
            ranked_by_precision_median,
            ranked_by_recall_median,
            ranked_by_specificity_median,
            ranked_by_f1_median,
            ranked_by_roc_auc_median,
        ]

        metrics_ranked_by_max = [
            ranked_by_accuracy_max,
            ranked_by_precision_max,
            ranked_by_recall_max,
            ranked_by_specificity_max,
            ranked_by_f1_max,
            ranked_by_roc_auc_max,
        ]

        # Aggregate ranks across multiple metrics
        ranking_metrics = self.metrics_df[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1-Score",
                "ROC-AUC",
                "Test Error",
            ]
        ]
        (
            aggregated_ranks_mean,
            aggregated_ranks_median,
            aggregated_ranks_max,
        ) = self.aggregate_ranks(ranking_metrics)

        # Determine the best model with the lowest aggregated rank
        best_model_mean = self.find_best_model(aggregated_ranks_mean)
        best_model_median = self.find_best_model(aggregated_ranks_median)
        best_model_max = self.find_best_model(aggregated_ranks_max)

        # Determine the worst model with the highest aggregated rank
        worst_model_mean = self.find_worst_model(aggregated_ranks_mean)
        worst_model_median = self.find_worst_model(aggregated_ranks_median)
        worst_model_max = self.find_worst_model(aggregated_ranks_max)

        # print(f"Stats: {stats}")
        # print(f"Metrics ranked by mean: {metrics_ranked_by_mean}")
        # print(f"Metrics ranked by median: {metrics_ranked_by_median}")
        # print(f"Metrics ranked by max: {metrics_ranked_by_max}")
        # print(f"Aggregated ranks by mean: {aggregated_ranks_mean}")
        # print(f"Aggregated ranks by median: {aggregated_ranks_median}")
        # print(f"Aggregated ranks by max: {aggregated_ranks_max}")
        print(f"Best model (mean): {best_model_mean}")
        print(f"Best model (median): {best_model_median}")
        print(f"Best model (max): {best_model_max}")
        print(f"Worst model (mean): {worst_model_mean}")
        print(f"Worst model (median): {worst_model_median}")
        print(f"Worst model (max): {worst_model_max}")

        # Save the evaluation results to separate CSV files
        #         stats = self.save_result_as_csv(stats, "stats")
        #         metrics_ranked_by_mean = self.save_result_as_csv(
        #             metrics_ranked_by_mean, "metrics_ranked_by_mean"
        #         )
        #         metrics_ranked_by_median = self.save_result_as_csv(
        #             metrics_ranked_by_median, "metrics_ranked_by_median"
        #         )
        #         metrics_ranked_by_max = self.save_result_as_csv(
        #             metrics_ranked_by_max, "metrics_ranked_by_max"
        #         )
        #         aggregated_ranks_mean = self.save_result_as_csv(
        #             aggregated_ranks_mean, "aggregated_ranks_mean"
        #         )
        #         aggregated_ranks_median = self.save_result_as_csv(
        #             aggregated_ranks_median, "aggregated_ranks_median"
        #         )
        #         aggregated_ranks_max = self.save_result_as_csv(
        #             aggregated_ranks_max, "aggregated_ranks_max"
        #         )

        #         best_model_mean = self.save_result_as_csv(best_model_mean, "best_model_mean")
        #         best_model_median = self.save_result_as_csv(
        #             best_model_median, "best_model_median"
        #         )
        #         best_model_max = self.save_result_as_csv(best_model_max, "best_model_max")
        #         worst_model_mean = self.save_result_as_csv(worst_model_mean, "worst_model_mean")
        #         worst_model_median = self.save_result_as_csv(
        #             worst_model_median, "worst_model_median"
        #         )
        #         worst_model_max = self.save_result_as_csv(worst_model_max, "worst_model_max")

        self.plot_summary_per_classifier()
        self.plot_summary_per_metric()
        # self.plot_precision_recall_curves()

        return (
            stats,
            metrics_ranked_by_mean,
            metrics_ranked_by_median,
            metrics_ranked_by_max,
            aggregated_ranks_mean,
            aggregated_ranks_median,
            aggregated_ranks_max,
            best_model_mean,
            best_model_median,
            best_model_max,
            worst_model_mean,
            worst_model_median,
            worst_model_max,
        )
