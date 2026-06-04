from data_loader import load_data
from kpi_calculator import calculate_basic_kpis

print("=" * 60)
print("FINANCIAL DECISION INTELLIGENCE COPILOT")
print("=" * 60)

df = load_data()

kpis = calculate_basic_kpis(df)

print("\nKPI SUMMARY\n")

for key, value in kpis.items():

    print(f"{key}: {value}")