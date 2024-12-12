import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curve(y_true, y_proba, label):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

def roc_auc_curve(X_val, y_val, model1, model_name1, model2, model_name2, model3, model_name3):   
    # ROC Curves
    plt.figure(figsize=(8, 6))

    plot_roc_curve(y_val, model1.predict_proba(X_val)[:, 1], model_name1)
    plot_roc_curve(y_val, model2.predict_proba(X_val)[:, 1], model_name2)
    plot_roc_curve(y_val, model3.predict_proba(X_val)[:, 1], model_name3)

    # Customize plot
    plt.title("ROC Curves for Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(title="Model", loc="lower right", fontsize=15, title_fontsize= 'xx-large')
    plt.tight_layout()

    # Show plot
    plt.show()