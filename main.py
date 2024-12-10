import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn import set_config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

sys

# output pandas DataFrames rather than numpy arrays
set_config(transform_output="pandas")

#initialize data
wine = src.data.read_dataset("data/raw/winequlaity-white.csv")

print(wine)