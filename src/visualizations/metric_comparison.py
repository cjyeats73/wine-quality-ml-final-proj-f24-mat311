import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.models.evaluation_metrics import stored_evaluation_metrics

def single_model_comparison(model_name, y_test, y_pred):
    # Combine the metrics into a DataFrame
    metric_names = ["Accuracy", "Precision", "Recall", "FPR", "Specificity", "F1 Score"]

    cm = confusion_matrix(y_test, y_pred)

    data = {
        "Metric": metric_names,
        model_name: list(stored_evaluation_metrics(y_test, y_pred, cm))
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Metric", y=model_name, data=df)
    plt.title(f"Metric Scores for {model_name}")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # Metrics typically range from 0 to 1
    plt.tight_layout()
    plt.show()

def multi_model_comparison(y_test, model1, y_pred1, model2, y_pred2, model3, y_pred3):
    metric_names = ["Accuracy", "Precision", "Recall", "FPR", "Specificity", "F1 Score"]

    cm1 = confusion_matrix(y_test, y_pred1)
    cm2 = confusion_matrix(y_test, y_pred2)
    cm3 = confusion_matrix(y_test, y_pred3)

    # This organizes the metrics, their scores, and the model names in a format suitable for creating a grouped bar plot
    data = {
        "Metric": metric_names * 3,  # Repeat metric names for each model
        "Score": list(stored_evaluation_metrics(y_test, y_pred1, cm1)) \
                    + list(stored_evaluation_metrics(y_test, y_pred2, cm2)) \
                    + list(stored_evaluation_metrics(y_test, y_pred3, cm3)),  # Combine all metrics
        "Model": [model1] * len(metric_names) + [model2] * len(metric_names) + [model3] * len(metric_names)  # Model labels
    }

    df = pd.DataFrame(data)

    # Plot the metrics using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Metric", y="Score", hue="Model", data=df, palette="viridis")

    # Customize the plot
    plt.title("Comparison of Metrics Across Models")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # Assuming metrics range between 0 and 1
    plt.legend(title="Model")
    plt.tight_layout()

    # Show the plot
    plt.show()