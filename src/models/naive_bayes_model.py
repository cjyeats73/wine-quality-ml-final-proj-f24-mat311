from sklearn.naive_bayes import GaussianNB
import random

random.seed(73)

def initialize_nb(X_train, y_train):
    nb = GaussianNB()
    nb_fit = nb.fit(X_train, y_train)
    return nb_fit