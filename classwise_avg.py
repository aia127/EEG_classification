from sklearn.metrics import accuracy_score
import numpy as np

def classwise_avg_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    accuracies = []

    for cls in classes:
        cls_indices = (y_true == cls)
        if np.sum(cls_indices) == 0:
            continue
        acc = accuracy_score(y_true[cls_indices], y_pred[cls_indices])
        accuracies.append(acc)

    return np.mean(accuracies)
