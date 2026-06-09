import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest

df = pd.read_csv(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)

features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X = df[features]

model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)

model.fit(X)

joblib.dump(
    model,
    "models/anomaly_model.pkl"
)

print(
    "Anomaly model saved successfully."
)