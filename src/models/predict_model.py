from src.visualizations.confusion_matrix_heatmap import confusion_matrix_heatmap
from src.models.evaluation_metrics import evaluation_metrics

def model_prediction(X_test, y_test, model, cmap, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix_heatmap(y_test, y_pred, cmap, model_name)
    evaluation_metrics(y_test, y_pred, model_name, cm)
    return y_pred