#importing external libraries
import random
from sklearn import set_config

#importing modules from this repo
from src.data.read_dataset import read_dataset
import src.data.preproccessing as prep
import src.visualizations.eda as eda
import src.models.knn_model as knn
import src.models.naive_bayes_model as nb
import src.models.decision_tree_model as dt
from src.models.predict_model import model_prediction
from src.visualizations.metric_comparison import single_model_comparison, multi_model_comparison
from src.visualizations.roc_auc_curve import roc_auc_curve

random.seed(73)

# output pandas DataFrames rather than numpy arrays
set_config(transform_output="pandas")

#initialize data
wine = read_dataset("data/raw/winequality-white.csv")

#wrangle data
# "1" means good quality, "0" means bad quality
clean_wine = prep.binary_classification_enabler(wine, "good_or_bad_quality", "quality", 5)
prep.save_new_dataset(clean_wine, 'data/processed/clean_winequality-white.csv')

print(clean_wine.dtypes)

#initial eda
eda.mass_histplot_features(clean_wine, "good_or_bad_quality")

# split data for machine learning models (70/20/10)
X_train, y_train, X_val, y_val, X_test, y_test = prep.train_val_test_split(clean_wine, "good_or_bad_quality")

#knn models (different sizes of K)
knn_three = knn.initialize_knn(X_train, y_train, 3)
model_prediction(X_val, y_val, knn_three, "Oranges", "3-Nearest Neightbors")

knn_seven = knn.initialize_knn(X_train, y_train, 7)
model_prediction(X_val, y_val, knn_seven, "Purples", "7-Nearest Neightbors")

knn_fifteen = knn.initialize_knn(X_train, y_train, 15)
model_prediction(X_val, y_val, knn_seven, "Grays", "15-Nearest Neightbors")

#knn model (optimal k)
k = knn.find_optimal_k(3, 30, X_train, y_train, X_val, y_val)
knn_optimal = knn.initialize_knn(X_train, y_train, k)
knn_val_pred = model_prediction(X_val, y_val, knn_optimal, "Reds", f"{k}-Nearest Neightbors")

#naive bayes model
nb_model = nb.initialize_nb(X_train, y_train)
nb_val_pred = model_prediction(X_val, y_val, nb_model, "Blues", "Gaussian Naïve Bayes")

#decision tree model
dt_model = dt.initialize_dt(X_train, y_train)
dt_val_pred = model_prediction(X_val, y_val, dt_model, "Greens", "Decision Tree")

#compare evaluation metrics against each other
single_model_comparison(f"{k}-NN Model", y_val, knn_val_pred)
single_model_comparison(f"Naïve Bayes", y_val, nb_val_pred)
single_model_comparison(f"Decision Tree", y_val, dt_val_pred)

multi_model_comparison(y_val, f"{k}-NN Model", knn_val_pred, f"Naïve Bayes", nb_val_pred, f"Decision Tree", dt_val_pred)

#roc-auc curve
roc_auc_curve(X_val, y_val, knn_optimal, f"{k}-NN Model", nb_model, f"Naïve Bayes", dt_model, f"Decision Tree")

#run best model with test data
dt_test_pred = model_prediction(X_test, y_test, dt_model, "Greens", "Decision Tree")
single_model_comparison(f"Decision Tree", y_test, dt_test_pred)

