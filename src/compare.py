import pandas as pd


class SummaryComparator:
    def __init__(self):
        self.summary_dataframes = {}

    def add_summary(self, filename):
        # Read the summary CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Store the DataFrame in the summary_dataframes dictionary
        self.summary_dataframes[filename] = df

    def compare_summaries(self):
        # Perform comparison based on a selected metric (e.g., Accuracy)
        selected_metric = "Accuracy"
        best_summary = None
        best_metric_value = None

        for filename, df in self.summary_dataframes.items():
            # Get the maximum metric value for the selected metric
            max_metric_value = df[selected_metric].max()

            # Check if this summary has the best metric value so far
            if best_metric_value is None or max_metric_value > best_metric_value:
                best_summary = filename
                best_metric_value = max_metric_value

        return best_summary, best_metric_value
