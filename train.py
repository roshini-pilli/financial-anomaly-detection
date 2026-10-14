
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("data/creditcard.csv")

print("Original dataset:", df.shape)

df = df.drop_duplicates().copy()

print("After removing duplicates:", df.shape)
print("\nClass distribution:")
print(df["Class"].value_counts())

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]

print("\n========== LOGISTIC REGRESSION ==========")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred, zero_division=0))

print("PR-AUC:", round(average_precision_score(y_test, lr_prob), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, lr_prob), 4))

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_fit, y_fit)

val_prob = rf.predict_proba(X_val)[:, 1]

print("\n========== THRESHOLD TUNING ==========")
print("Threshold | Precision | Recall | F1-score")

best_threshold = 0.5
best_f1 = -1

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
    val_pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, val_pred, zero_division=0
    )

    print(
        f"{threshold:.1f}       | "
        f"{precision:.3f}     | "
        f"{recall:.3f}  | "
        f"{f1:.3f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nSelected threshold:", best_threshold)
print("Validation F1-score:", round(best_f1, 4))

rf.fit(X_train, y_train)

rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob >= best_threshold).astype(int)

print("\n========== RANDOM FOREST FINAL TEST RESULTS ==========")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred, zero_division=0))

print("PR-AUC:", round(average_precision_score(y_test, rf_prob), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, rf_prob), 4))

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": rf,
        "threshold": best_threshold,
        "features": list(X.columns)
    },
    "models/fraud_model.pkl"
)

print("\nModel saved to models/fraud_model.pkl")
