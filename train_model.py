import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load Dataset
df = pd.read_csv("data/heart.csv")

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Base Models
base_models = [
    ("rf", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )),
    ("gb", GradientBoostingClassifier(
        random_state=42
    ))
]

# Meta Model
meta_model = LogisticRegression(max_iter=1000)

# Stacking Classifier
model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# Save Model
joblib.dump(
    model,
    "models/stacking_classifier.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/columns.pkl"
)

print("Model Saved Successfully")