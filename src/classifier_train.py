import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "creditcard.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

print("Loading dataset...")

data = pd.read_csv(
    DATA_PATH
)

X = data.drop(
    "Class",
    axis=1
)

y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "fraud_classifier.pkl"
    )
)

print(
    "\nModel saved successfully"
)