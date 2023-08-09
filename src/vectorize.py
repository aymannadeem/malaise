import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextVectorizer:
    def __init__(self, x):
        self.x = x

    def vectorize_text_data(self, dataframe, column, vectorizer):
        sparse_matrix = vectorizer.fit_transform(dataframe[column])
        return sparse_matrix

    def process_data(self):
        cv_vectorizer = CountVectorizer()
        tfidf_vectorizer = TfidfVectorizer()

        cv_repo = self.vectorize_text_data(self.x, "Repo", cv_vectorizer)
        cv_author = self.vectorize_text_data(self.x, "Author", cv_vectorizer)
        cv_topic = self.vectorize_text_data(self.x, "Topic", cv_vectorizer)

        tfidf_repo = self.vectorize_text_data(self.x, "Repo", tfidf_vectorizer)
        tfidf_author = self.vectorize_text_data(self.x, "Author", tfidf_vectorizer)
        tfidf_topic = self.vectorize_text_data(self.x, "Topic", tfidf_vectorizer)

        cv_df = self.x.drop(columns=["Repo", "Author", "Topic"])
        tfidf_df = self.x.drop(columns=["Repo", "Author", "Topic"])

        return cv_df, tfidf_df
