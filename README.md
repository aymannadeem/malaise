# malaise
### _Malware AI Search Engine_

#### What does this project do?

Malaise (Malware AI Search Engine) is a malware detection engine that uses AI to learn user activity patterns that indicate which repositories host malware. This prototype, developed in service of a CS dissertation at the University of Oxford, scrapes repository data, preprocesses it, and trains ML models to predict which repositories have malware. 

#### Why is this project useful?

This project enables the security and malware research communities by publishing a new dataset, which augments the [SourceFinder dataset]([url](https://www.usenix.org/system/files/raid20-rokon.pdf)) used to establish groundtruth labels, in addition to proving out a new technique that uses behavioural data instead of source file data to detect malware. 

#### How do I get started?

First, run `script/setup`, which recreates `virtualenv` from script.

There are two parts to this library: (1) data curation and (2) machine learning.

### Data Curation

1. Run `./script/get_data.py` which ingests two files, `datasets/raw/data.txt` and `datasets/raw/source_finder.csv`, makes calls to the GitHub REST API to get the desired fields, adds a `has_malware` label to indicate which repositories have malware, and writes the results to two files: `datasets/inputs/benign_repos.csv` and `datasets/inputs/benign_malware_repos.csv`. 
2. Merge benign and malicious data using `/concat-parsed-repos.ipynb` from `datasets/inputs/benign_repos.csv` and `datasets/inputs/benign_malware_repos.csv` into `/malware-detection/datasets/extend/parsed_full_dataset.csv`

### Machine Learning

1. Run `./src/main.py` over the generated `/malware-detection/datasets/extend/parsed_full_dataset.csv` dataset, which will do the following:
  1. Enrich the data with additional features using existing attributes. 
  2. Preprocess the data.
  3. Apply text vectorization via CountVectorizer and TF-IDF methods.
  4. Scale the data using MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, and PowerTransformer.
  5. Perform binary classification using 5 models: Logistic Regression, Gaussian Naive Bayes, Perceptron, Decision Tree, and Random Forest, and produce confusion matrices for each permutation of vectorization method, scaling technique, and classifier, in addition to train and test accuracy, error, precision, recall, f1-score, and specificify metrics. 
  6. Compute SHAP values, precision-recall curves, fit analysis, and summarize results.
  7. Perform feature selection using methods such as K-best selection, Chi-Squared, Mutual Information, as well as by selecting the top features as determined by SHAP summary plots.
  8. Re-run model fitting.
  9. Tune and adjust parameters as necessary. 


### Where can I get more help, if I need it?

Contact creator [@aymannadeem]([url](https://github.com/aymannadeem)https://github.com/aymannadeem) / `aymannadeem@gmail.com`
