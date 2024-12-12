# Machine Learning Project: Wine Quality

## Description
This project uses machine learning to predict the tasting quality of white wine samples using a variety of features, like acidity and alcohol volume. This is an experiemtn to see which of three machine learning models is best for making sure bad quality wine doesn't make it to store shelves.

## Documentation
Brief overview on what each in-use department does.

### Data
    - Raw: the unchanged, original dataset.
    - Processed: the dataset after being cleaned.

### Notebooks
    Any Jupyter notebooks we used will be in here.

### Src
    Holds all the standalone scripts
    **Data:** 
        - preprocessing.py - has a number of pre-EDA functions within
        - read_dataset.py - reads the dataset and stores it in a pandas DataFrame
    **Models:**
        - decision_tree_model.py - initializes the DecisionTreeClassifier
        - evaluation_metrics.py - contains several functions that computer the evaluation metrics of a model
        - knn_model.py - initializes the K-NearestNeightborsClassifier. Also has a function that finds the optimal k value from specificity.
        - naive_bayes_model.py - initializes the GaussianNaiveBayesClassifier
        - predict_model.py - a universal function that makes models predict the y value of the dataset they were fitted for. Also creates a confusion matrix and lists evaluation metrics.
    **Visualizations:**
        - confusion_matrix_heatmap.py: Calculates the confusion matrix and plots a heat map using it.
        - eda.py - Hosts simple EDA visualization functions
        - metric_comparison.py - Creates bar plots that show the relation of multiple metrics and models.
        - roc_auc_curve.py - Used to create a ROC-AUC Curve

### main.py
    Main file; when run, recreates the whole project start to finish.

### README.md
    Used to document the contents of the GitHub repository and give an overview of the project.

