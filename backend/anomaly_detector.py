import joblib

model = joblib.load(
    "models/anomaly_model.pkl"
)

def detect_anomaly(input_data):

    prediction = model.predict(
        input_data
    )

    score = model.decision_function(
        input_data
    )

    return (
        prediction[0],
        score[0]
    )