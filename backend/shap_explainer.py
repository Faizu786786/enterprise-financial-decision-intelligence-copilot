import os
import joblib
import shap
import pandas as pd

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, ".."))
MODEL_PATH = os.path.join(project_root, "models", "fraud_model.pkl")

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

def get_shap_importance(input_data):

    shap_values = explainer.shap_values(
        input_data
    )

    values = shap_values[0, :, 1]

    return pd.DataFrame(
        {
            "Feature": input_data.columns,
            "Importance": abs(values)
        }
    ).sort_values(
        by="Importance",
        ascending=False
    )