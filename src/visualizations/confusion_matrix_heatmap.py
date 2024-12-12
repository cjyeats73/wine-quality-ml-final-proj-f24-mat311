import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def confusion_matrix_heatmap(y_true, y_pred, cmap, model):
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt=".0f", cmap=cmap, cbar=True,
                    xticklabels=["Predicted False", "Predicted True"], yticklabels=["Actual False", "Actual True"])
    ax.set_title(f"Confusion Matrix for {model}")
    plt.show()
    return cm