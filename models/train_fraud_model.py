from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import pandas as pd
import joblib


print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)

print("\nDATASET LOADED SUCCESSFULLY\n")

print("Shape:", df.shape)

features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X = df[features]

y = df["isFraud"]

print("\nFeatures Selected:")
print(features)

print("\nTarget Variable:")
print("isFraud")

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("\nFraud Rate:")
print(round((y.sum() / len(y)) * 100, 4), "%")

print(y.value_counts())

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
print("Training Complete")

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print(feature_importance)


joblib.dump(
    model,
    "models/fraud_model.pkl"
)

print("Model Saved Successfully")


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

