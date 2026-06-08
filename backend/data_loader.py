import pandas as pd
import streamlit as st

DATASET_PATH = (
    "data/raw/PS_20174392719_1491204439457_log.csv"
)

@st.cache_data
def load_data():

    df = pd.read_csv(DATASET_PATH)

    return df