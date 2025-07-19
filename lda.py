import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import roc_curve, roc_auc_score


#Load EEG data
df = pd.read_csv("eeg_sorted_by_task.csv")

#Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#Features and labels
X = df.iloc[:, :16]  # EEG channels
y = df["task_type"]

#Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

#LDA classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

#Predict and evaluate
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#LDA accuracy difference with changes in test and training data size
with open('output.txt', 'a') as file:
    file.write(f"LDA Accuracy for training 0.9 and test size 0.1--{accuracy:.4f}\n")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
#Plot confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

#Calculate precision
precision = precision_score(y_test, y_pred, average='binary')
print(f"Precision {precision:.2f}")

y_probs = lda.predict_proba(X_test)[:, 1]

#Compute ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

#Compute AUC score
auc = roc_auc_score(y_test, y_probs)
print(le.classes_)

#Plot the ROC curve
# plt.figure()
# plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # diagonal line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.show()
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line (random guess)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
print(classification_report(y_test, y_pred, target_names=le.classes_))
