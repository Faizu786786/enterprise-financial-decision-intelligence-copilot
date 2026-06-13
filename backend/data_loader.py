import os
import pandas as pd
import streamlit as st

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, ".."))
DATASET_PATH = os.path.join(project_root, "data", "raw", "PS_20174392719_1491204439457_log.csv")

@st.cache_data
def load_data():

    df = pd.read_csv(DATASET_PATH)

    return df