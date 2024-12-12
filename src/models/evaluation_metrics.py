from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluation_metrics(y_test, y_pred, model, cm):
    accur = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    spec = 1 - fpr_calculator(cm)
    fpr = fpr_calculator(cm)

    print(f"Statistics for {model} Model:")
    print(f"Accuracy Score: {accur}")
    print(f"Precision Score: {prec}")
    print(f"Recall Score: {recall}")
    print(f"FPR Score: {fpr}")
    print(f"Specificity Score: {spec}")
    print(f"F1 Score: {f1}")
    print()

def stored_evaluation_metrics(y_test, y_pred, cm):
    accur = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    spec = 1 - fpr_calculator(cm)
    fpr = fpr_calculator(cm)

    return accur, prec, recall, fpr, spec, f1

def fpr_calculator(cm):
    FP = cm[0][1]
    TN = cm[0][0]

    return FP / (FP+TN)