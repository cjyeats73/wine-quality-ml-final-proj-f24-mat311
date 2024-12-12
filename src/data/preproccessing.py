import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def binary_classification_enabler(df, new_col, old_col, thresh):
    df[new_col] = np.where(df[old_col] > thresh, 1, 0)
    clean_df = df.drop(columns=old_col)
    return clean_df

def train_val_test_split(df, col, random_seed=73):
    X = df.drop(columns=[col])
    y = df[col]

    # train and validation split 80: 20
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # train and test split 70: 10
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.125, random_state=random_seed)

    print(X_train)
    print(X_val)
    print(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_new_dataset(df, file_location):
    df.to_csv(file_location)