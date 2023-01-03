import streamlit as st
import pandas as pd

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_diabetes_data():
    """
    Function which loads the Diabetes Dataset as a DataFrame
    """

    df = pd.read_csv("outputs/datasets/collection/diabetes.csv")
    
    return df