import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, make_scorer, classification_report
from classwise_avg import classwise_avg_accuracy



classwise_scorer = make_scorer(classwise_avg_accuracy, greater_is_better=True)

# --------------------
# Load and preprocess data
df = pd.read_csv("eeg_with_row_statistics_feature_uplifting.csv")  # Adjust path if needed

# Encode target labels
le = LabelEncoder()
df['task_type_encoded'] = le.fit_transform(df['task_type'])

# Drop non-feature columns: subject_id and original target
df = df.drop(columns=['subject_id', 'task_type'])

# Feature list from prior stepwise selection
selected_features = ['max_value', 'Cz', 'Fz', 'CP6', 'FC6', 'CP5']
X = df[selected_features]
y = df['task_type_encoded']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# --------------------
# Hyperparameter grid
param_grid = {
    'n_estimators': [10, 20, 50,100],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Grid SearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=classwise_scorer,
    cv=3,
    n_jobs=-4,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# --------------------
# Results
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best CV Classwise Accuracy:", grid_search.best_score_)

# Evaluate on test set
y_test_pred = best_rf.predict(X_test)
test_classwise_acc = classwise_avg_accuracy(y_test, y_test_pred)
print("Test Set Classwise Averaged Accuracy:", test_classwise_acc)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Predict probabilities for the positive class
y_test_proba = best_rf.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line (random guess)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


