from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import random

from src.models.evaluation_metrics import fpr_calculator

random.seed(73)

def initialize_knn(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_fit = knn.fit(X_train, y_train)
    return knn_fit

def find_optimal_k(start, stop, X_train, y_train, X_test, y_test):
    optimal = 0
    optimal_fpr = 1

    for k in range(start,stop, 2):
        knn_search = KNeighborsClassifier(n_neighbors=k)
        knn_search_fit = knn_search.fit(X_train, y_train)
        y_pred = knn_search_fit.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fpr = fpr_calculator(cm)
        if fpr < optimal_fpr:
            optimal = k
            optimal_fpr = fpr

    return optimal