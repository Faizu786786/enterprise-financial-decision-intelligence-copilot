import pandas as pd

DATASET_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"

def load_data():
    """
    Load PaySim dataset
    """

    df = pd.read_csv(DATASET_PATH)

    return df