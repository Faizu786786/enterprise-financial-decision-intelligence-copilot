import os
import joblib

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, ".."))
MODEL_PATH = os.path.join(project_root, "models", "anomaly_model.pkl")

model = joblib.load(MODEL_PATH)

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