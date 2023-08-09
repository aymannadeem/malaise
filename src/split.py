import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, x):
        self.x = x

    # Takes a dataset and returns four values consisting of a training and test variable for each of X and Y:
    def shuffle_and_split(self):
        # Shuffle data to ensure randomization
        shuffled = self.x.sample(frac=1, random_state=42)

        # Define variables for models
        X = shuffled.drop("has_malware", axis=1)
        Y = shuffled["has_malware"]

        # Split the given data into separate training and test datasets
        (X_train, X_test, Y_train, Y_test) = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        return X_train, X_test, Y_train, Y_test


#################### Dylan's Addition ####################
# # Takes a dataset and splits into train/test sets via condition provided as input
# def conditional_split(self, cond_col:str, split_perc:float, shuffle:bool=True):
#     # Shuffle dataset by default
#     if(shuffle):
#         idx = [i for i in range(self.x.shape[0])]
#         random.shuffle(idx)

#     # Given a column name, split dataset such that no common values are shared between train/test
#     cond_set = self.x[cond_col].unique()
#     limiter = 0

#     while(True):
#         #shuffle here
#         train_idx = cond_set[0:(len(cond_set)*split_perc)]
#         train = self.x[self.x[cond].isin(train_idx)]
#         test = self.x[~self.x[cond].isin(train_idx)]

#         if((len(train)/len(self.x) < split_perc+0.05 or len(train)/len(self.x) > split_perc-0.05)):
#             break
#         elif(limiter > 20):
#             break

#         limiter = limiter+1

#     return train, test
