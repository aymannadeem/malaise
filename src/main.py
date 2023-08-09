#!/usr/bin/env python3
import sys
import os
import pandas as pd
from enrich import Enricher
from preprocess import Preprocessor
from vectorize import TextVectorizer
from split import DataSplitter
from scale import DataScaler

from select_features import FeatureSelector
from classify import ModelEvaluator
from overunder import OverUnderFitEvaluator
from summarize import PerformanceSummarizer

# Get the absolute path of the 'src' directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

# Add the 'src' directory to the module search path
sys.path.append(src_path)


def process():
    data = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "../datasets/extend/parsed_full_dataset.csv"
        )
    )

    # --- ENRICHMENT ---

    # Create an instance of the Enricher class and pass in the scraped dataset
    enricher = Enricher(data)

    # Create new features or transform existing features to enhance the representation of the data
    enriched_data = enricher.enrich()

    # --- PREPROCESSING AND FEATURE ENGINEERING ---

    # Create an instance of the preprocessor class
    preprocessor = Preprocessor(enriched_data)

    # Clean the dataset by handle missing values, outliers, and other data quality issues.
    preprocessed_data = preprocessor.preprocess()

    # --- TEXT VECTORIZATION ---

    # Create an instance of the DataProcessor class and pass the dataframe
    vectorizer = TextVectorizer(preprocessed_data)

    # Call the method to process the data into two vectorized dataframes
    cv, tf = vectorizer.process_data()

    # --- TRAIN TEST SPLIT ---

    # Create an instance of DataScale class and pass in each vectorized dataframe
    split_cv = DataSplitter(cv)
    split_tf = DataSplitter(tf)

    # Shuffle and split
    x_train_cv, x_test_cv, y_train_cv, y_test_cv = split_cv.shuffle_and_split()
    x_train_tf, x_test_tf, y_train_tf, y_test_tf = split_tf.shuffle_and_split()

    # --- FEATURE SCALING ---

    # Create an instance of DataScale class and pass in each vectorized dataframe
    datascaler_cv = DataScaler(x_train_cv, x_test_cv, y_train_cv, y_test_cv)
    datascaler_tf = DataScaler(x_train_tf, x_test_tf, y_train_tf, y_test_tf)

    # Scale
    scaler_names_cv, train_scaled_cv, test_scaled_cv = datascaler_cv.scale_datasets()
    scaler_names_tf, train_scaled_tf, test_scaled_tf = datascaler_tf.scale_datasets()

    # --- FEATURE SELECTION ---

    # Create instances of FeatureSelector
    feature_selector_cv = FeatureSelector(
        scaler_names_cv, train_scaled_cv, test_scaled_cv
    )
    feature_selector_tf = FeatureSelector(
        scaler_names_tf, train_scaled_tf, test_scaled_tf
    )

    # Perform feature selection using different methods
    (
        scaler_names_cv_kbest,
        train_scaled_cv_kbest,
        test_scaled_cv_kbest,
    ) = feature_selector_cv.select_features("SelectKBest")
    (
        scaler_names_cv_chi,
        train_scaled_cv_chi,
        test_scaled_cv_chi,
    ) = feature_selector_cv.select_features("Chi-Squared")
    (
        scaler_names_cv_mi,
        train_scaled_cv_mi,
        test_scaled_cv_mi,
    ) = feature_selector_cv.select_features("Mutual Information")

    (
        scaler_names_tf_kbest,
        train_scaled_tf_kbest,
        test_scaled_tf_kbest,
    ) = feature_selector_tf.select_features("SelectKBest")
    (
        scaler_names_tf_chi,
        train_scaled_tf_chi,
        test_scaled_tf_chi,
    ) = feature_selector_tf.select_features("Chi-Squared")
    (
        scaler_names_tf_mi,
        train_scaled_tf_mi,
        test_scaled_tf_mi,
    ) = feature_selector_tf.select_features("Mutual Information")

    # --- MACHINE LEARNING ---

    # All features -- no selection
    #     evaluator_cv = ModelEvaluator(
    #         "CV", "none", scaler_names_cv, train_scaled_cv, test_scaled_cv
    #     )
    #     evaluator_tf = ModelEvaluator(
    #         "TF-IDF", "none", scaler_names_tf, train_scaled_tf, test_scaled_tf
    #     )

    #     # Model results -- no selection
    #     results_cv, metrics_df_cv = evaluator_cv.evaluate_classifiers()
    #     results_tf, metrics_df_tf = evaluator_tf.evaluate_classifiers()

    # Features selected via K-best
    evaluator_cv_kbest = ModelEvaluator(
        "CV",
        "k-best",
        scaler_names_cv_kbest,
        train_scaled_cv_kbest,
        test_scaled_cv_kbest,
    )
    evaluator_tf_kbest = ModelEvaluator(
        "TF-IDF",
        "k-best",
        scaler_names_tf_kbest,
        train_scaled_tf_kbest,
        test_scaled_tf_kbest,
    )

    # Model results via K-best
    results_cv_kbest, metrics_df_cv_kbest = evaluator_cv_kbest.evaluate_classifiers()
    results_tf_kbest, metrics_df_tf_kbest = evaluator_tf_kbest.evaluate_classifiers()

    # Features selected via Chi-squared
    evaluator_cv_chi = ModelEvaluator(
        "CV",
        "chi-squared",
        scaler_names_cv_chi,
        train_scaled_cv_chi,
        test_scaled_cv_chi,
    )
    evaluator_tf_chi = ModelEvaluator(
        "TF-IDF",
        "chi-squared",
        scaler_names_tf_chi,
        train_scaled_tf_chi,
        test_scaled_tf_chi,
    )

    # Model results via chi-squared
    results_cv_chi, metrics_df_cv_chi = evaluator_cv_chi.evaluate_classifiers()
    results_tf_chi, metrics_df_tf_chi = evaluator_tf_chi.evaluate_classifiers()

    # Features selected via Mutual information
    evaluator_cv_mi = ModelEvaluator(
        "CV",
        "mutual information",
        scaler_names_cv_mi,
        train_scaled_cv_mi,
        test_scaled_cv_mi,
    )
    evaluator_tf_mi = ModelEvaluator(
        "TF-IDF",
        "mutual information",
        scaler_names_tf_mi,
        train_scaled_tf_mi,
        test_scaled_tf_mi,
    )

    # Model results via Mutual information
    results_cv_mi, metrics_df_cv_mi = evaluator_cv_mi.evaluate_classifiers()
    results_tf_mi, metrics_df_tf_mi = evaluator_tf_mi.evaluate_classifiers()

    # --- SECOND ITERATION WITH SHAP ---

    # Get all files in the output directory
    output_dir = os.path.join(os.path.dirname(__file__), "../results/shap")
    # output_dir = "/Users/aymannadeem/code/malware-detection/results/shap"
    all_files = os.listdir(output_dir)

    # Filter out non-CSV files
    csv_files = [file for file in all_files if file.endswith(".csv")]

    # Load all CSV files into a dictionary of DataFrames
    dfs = {file[:-4]: pd.read_csv(os.path.join(output_dir, file)) for file in csv_files}

    # Compute the absolute values before concatenation
    abs_dfs = {name: df.abs() for name, df in dfs.items()}

    # Concatenate all dataframes, adding a 'source' column to know where each row came from
    concat_df = pd.concat([df.assign(source=name) for name, df in abs_dfs.items()])

    # Exclude 'source' from the features for the importance computation
    features = concat_df.columns.drop("source")

    # Compute the total importance of each feature
    total_importance = concat_df[features].sum()

    # Sort by the total importance
    sorted_importance = total_importance.sort_values(ascending=False)

    # Get the top contributing features
    n = 20  # change this to the number of top features desired
    top_features = sorted_importance.index[:n]
    print(top_features)

    # Update train_scaled_tf and test_scaled_tf to include only top_features
    for i in range(len(train_scaled_tf)):
        train_scaled_tf[i] = (
            train_scaled_tf[i][0][top_features],
            train_scaled_tf[i][1],
        )

    for i in range(len(test_scaled_tf)):
        test_scaled_tf[i] = (test_scaled_tf[i][0][top_features], test_scaled_tf[i][1])

    # No need to update scaler_names_tf, it just contains the names of the scalers used.

    # Now you can use these updated inputs in your ModelEvaluator
    evaluator_cv_shap = ModelEvaluator(
        "CV",
        "shap",
        scaler_names_tf,
        train_scaled_tf,
        test_scaled_tf,
    )

    evaluator_tf_shap = ModelEvaluator(
        "TF-IDF",
        "shap",
        scaler_names_tf,
        train_scaled_tf,
        test_scaled_tf,
    )

    results_shap, metrics_cv_shap = evaluator_cv_shap.evaluate_classifiers()
    results_shap, metrics_tf_shap = evaluator_tf_shap.evaluate_classifiers()

    # --- PERFORMANCE EVALUATION ---

    # Concatenate the results of all vectorization and feature selection methods
    metrics_df = pd.concat(
        [
            # metrics_df_cv,
            # metrics_df_tf,
            metrics_df_cv_kbest,
            metrics_df_tf_kbest,
            metrics_df_cv_chi,
            metrics_df_tf_chi,
            metrics_df_cv_mi,
            metrics_df_tf_mi,
            metrics_cv_shap,
            metrics_tf_shap,
        ]
    )
    metrics_df["index"] = range(0, len(metrics_df))
    metrics_df = metrics_df.set_index("index")
    metrics_df.to_csv(
        os.path.join(os.path.dirname(__file__), "../results/all_metrics.csv"),
        index=False,
    )

    # Evaluate overfitting and/or underfitting
    fit_evaluator = OverUnderFitEvaluator(metrics_df)
    fit_evaluator.plot_evaluation(
        os.path.join(os.path.dirname(__file__), "../results/fit/evaluation_plot")
    )

    # Evaluate performance summaries
    evaluator = PerformanceSummarizer(metrics_df)
    (
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
    ) = evaluator.evaluate_metrics()


if __name__ == "__main__":
    process()
