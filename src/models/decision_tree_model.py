from sklearn.tree import DecisionTreeClassifier
import random

random.seed(73)

def initialize_dt(X_train, y_train,random_seed=73, max_dep=None):
    dt = DecisionTreeClassifier(max_depth=max_dep, random_state=random_seed)
    dt_fit = dt.fit(X_train, y_train)
    return dt_fit