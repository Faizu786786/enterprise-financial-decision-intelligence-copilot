import joblib
import shap
import pandas as pd

model = joblib.load(
    "models/fraud_model.pkl"
)

explainer = shap.TreeExplainer(
    model
)

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