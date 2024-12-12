import pandas as pd

def read_dataset(file_name):
    return pd.read_csv(file_name, delimiter=';')