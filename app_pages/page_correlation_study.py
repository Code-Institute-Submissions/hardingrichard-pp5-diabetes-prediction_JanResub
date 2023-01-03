import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Loads the function in data_management for dataset
from src.data_management import load_diabetes_data


def page_correlation_study_body():
    """
    Function that loads the elements within the correlation study page.
    To display the correlated features
    """

    st.write("## Diabetes Correlation Study:")
    st.info(
        "Placeholder text"
    )