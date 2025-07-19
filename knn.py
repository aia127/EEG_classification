import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

#Load data
df = pd.read_csv("eeg_iqr_filtered_rows.csv")

#Features and labels
X = df.iloc[:, :16]  # first 16 columns = EEG electrodes
y = df["task_type"]

#Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(f"KNN Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
