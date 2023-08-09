import numpy as np
import pandas as pd
import string
import ast


class Preprocessor:
    def __init__(self, x):
        self.x = x

    def preprocess(self):
        # Drop columns that have all NaN values for every row
        self.x = self.drop_columns_with_all_nan()

        # Remove punctuation from repo, author, topic
        self.x["Repo"] = self.preprocess_text(self.x, "Repo")
        self.x["Author"] = self.preprocess_text(self.x, "Author")
        self.x["Topic"] = self.preprocess_text(self.x, "Topic")

        # Unpack language dictionary
        self.x = self.unpack_languages(self.x)

        # Drop datetime columns
        self.x = self.x.drop(
            columns=[
                "Creation",
                "Last Update",
                "author_account_created_at",
                "author_account_last_update",
            ],
            axis=1,
        )

        return self.x

    # Remove columns with all NaN values
    def drop_columns_with_all_nan(self):
        # Drop columns with all NaN values
        df = self.x.dropna(axis=1, how="all")

        return df

    # Preprocess text to make it amenable to text vectorization
    def preprocess_text(self, df, column_name):
        # Remove punctuation
        punct = string.punctuation

        for c in punct:
            df.loc[:, column_name] = df.loc[:, column_name].map(
                lambda y: y.replace(c, " ")
            )

        # Convert to lowercase
        df.loc[:, column_name] = df.loc[:, column_name].str.lower()

        return df.loc[:, column_name]

    # Process language column into multiple columns representing
    # each language and its corresponding coverage for a given repo
    def unpack_languages(self, df):
        # Convert the object data into dictionary data
        df["Language"] = df["Language"].apply(ast.literal_eval)

        # Use the 'apply' function to create new columns from the dictionary data
        new_columns = df["Language"].apply(pd.Series, dtype="float64")

        # Replace all NaN values with 0
        new_columns = new_columns.fillna(0)

        # Concatenate the original DataFrame with the new columns
        df = pd.concat([df, new_columns], axis=1)

        # Drop the original language column
        df = df.drop(columns=["Language"], axis=1)

        return df
