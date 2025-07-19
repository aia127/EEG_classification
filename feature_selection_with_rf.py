from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from classwise_avg import classwise_avg_accuracy
df=pd.read_csv("eeg_with_row_statistics_feature_uplifting.csv")
# Drop non-feature columns
X = df.drop(columns=["task_type", "subject_id"])
y = df["task_type"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train (60%), validation (20%), test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
def stepwise_selection_classwise(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, max_features=None):
    selected_features = []
    remaining_features = list(X_train.columns)
    best_score = 0.0
    improved = True

    if max_features:
        remaining_features = remaining_features[:max_features]

    while improved:
        improved = False

        # --- Forward Step ---
        forward_scores = []
        for feature in remaining_features:
            current_features = selected_features + [feature]
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            clf.fit(X_train[current_features], y_train)
            preds = clf.predict(X_val[current_features])
            score = classwise_avg_accuracy(y_val, preds)
            forward_scores.append((feature, score))

        if forward_scores:
            best_feature, best_forward_score = max(forward_scores, key=lambda x: x[1])
            if best_forward_score > best_score:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                best_score = best_forward_score
                improved = True

        # --- Backward Step ---
        if len(selected_features) > 1:
            backward_scores = []
            for feature in selected_features:
                current_features = [f for f in selected_features if f != feature]
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                clf.fit(X_train[current_features], y_train)
                preds = clf.predict(X_val[current_features])
                score = classwise_avg_accuracy(y_val, preds)
                backward_scores.append((feature, score))

            worst_feature, best_backward_score = max(backward_scores, key=lambda x: x[1])
            if best_backward_score > best_score:
                selected_features.remove(worst_feature)
                remaining_features.append(worst_feature)
                best_score = best_backward_score
                improved = True

    print (selected_features, best_score)

stepwise_selection_classwise()