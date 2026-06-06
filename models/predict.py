import joblib
import pandas as pd

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = joblib.load(
    "models/fraud_model.pkl"
)

print("MODEL LOADED SUCCESSFULLY")

sample_transaction = pd.DataFrame(
    [[
        50000,
        100000,
        50000,
        0,
        50000
    ]],
    columns=[
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]
)

prediction = model.predict(
    sample_transaction
)

print("\nPrediction:")

if prediction[0] == 1:
    print("FRAUD")
else:
    print("NOT FRAUD")