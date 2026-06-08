import sys
import os
import json

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(project_root)

from backend.data_loader import load_data
from backend.kpi_calculator import calculate_basic_kpis

print("Loading dataset...")

df = load_data()

print("Calculating KPIs...")

kpis = calculate_basic_kpis(df)

os.makedirs(
    "data/processed",
    exist_ok=True
)

kpis = json.loads(
    json.dumps(
        kpis,
        default=lambda x: (
            int(x)
            if hasattr(x, "__int__")
            else float(x)
        )
    )
)

with open(
    "data/processed/kpis.json",
    "w"
) as file:

    json.dump(
        kpis,
        file,
        indent=4
    )

print("Dashboard data saved successfully.")