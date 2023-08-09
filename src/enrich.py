import numpy as np
import pandas as pd
import json


class Enricher:
    def __init__(self, x):
        self.x = x

    def enrich(self):
        # Length of author name
        self.x["author_name_length"] = self.x["Author"].str.len()

        # Length of repo name
        self.x["repo_name_length"] = self.x["Repo"].str.len()

        # Total Issues
        self.x["total_issues"] = self.x["open_issues"] + self.x["closed_issues"]

        # Total pulls
        self.x["total_pulls"] = self.x["open_pulls"] + self.x["closed_pulls"]

        # Number of languages
        a = self.x["Language"].map(lambda a: str(a).replace("'", '"'))
        self.x["num_languages"] = a.map(lambda b: len(json.loads(b).keys()))

        # Number of topics
        y = self.x["Topic"].map(lambda a: str(a).replace("'", '"'))
        self.x["num_topics"] = y.map(lambda b: len(json.loads(b)))

        # Author followers:following
        self.x["author_followers_nonzero"] = self.x["author_followers"].map(
            lambda a: a + 1
        )
        self.x["author_following_nonzero"] = self.x["author_following"].map(
            lambda a: a + 1
        )
        self.x["author_followers_following"] = (
            self.x["author_followers_nonzero"] / self.x["author_following_nonzero"]
        )

        # Drop nonzero follower and following columns
        self.x = self.x.drop(
            columns=["author_followers_nonzero", "author_following_nonzero"], axis=1
        )

        # Author email is unique (0 or 1)
        unique_emails = self.x["author_email"].unique()

        repo_count_per_email = {}
        for unique_email in unique_emails:
            repo_count_per_email[unique_email] = len(
                self.x[self.x["author_email"] == unique_email]
            )

        self.x["author_email_repo_count"] = self.x["author_email"].map(
            lambda y: repo_count_per_email[y]
        )

        # Temporal metrics

        # Repository activity duration
        diff = pd.to_datetime(self.x["Last Update"]) - pd.to_datetime(
            self.x["Creation"]
        )
        self.x["Repo Activity Duration (Days)"] = diff / np.timedelta64(1, "D")

        # Age of account when repo created
        diff = pd.to_datetime(self.x["Creation"]) - pd.to_datetime(
            self.x["author_account_created_at"]
        )
        self.x[
            "Age of account at time of repo creation (Days)"
        ] = diff / np.timedelta64(1, "D")

        # author account activity duration
        diff = pd.to_datetime(self.x["author_account_last_update"]) - pd.to_datetime(
            self.x["author_account_created_at"]
        )
        self.x["Author Account Activity Duration (Days)"] = diff / np.timedelta64(
            1, "D"
        )

        self.x = self.extract_datetime_features(self.x, "Creation")
        self.x = self.extract_datetime_features(self.x, "Last Update")
        self.x = self.extract_datetime_features(self.x, "author_account_created_at")
        self.x = self.extract_datetime_features(self.x, "author_account_last_update")

        return self.x

    # Decompose datetime values into categorical variables
    def extract_datetime_features(self, df, datetime_column):
        # Convert datetime column to pandas datetime format
        df[datetime_column] = pd.to_datetime(df[datetime_column])

        # Extract year, month, day, day of the week, and hour
        df[f"{datetime_column}_year"] = df[datetime_column].dt.year
        df[f"{datetime_column}_month"] = df[datetime_column].dt.month
        df[f"{datetime_column}_day"] = df[datetime_column].dt.day
        df[f"{datetime_column}_day_of_week"] = df[datetime_column].dt.dayofweek
        df[f"{datetime_column}_hour_of_day"] = df[datetime_column].dt.ceil("H").dt.hour

        return df

    # @staticmethod
