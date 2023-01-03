import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Loads the function in data_management for dataset
from src.data_management.diabetes_data import load_diabetes_data


def page_correlation_study_body():
    """
    Function that loads the elements within the correlation study page.
    To display the correlated features
    """

    df = load_diabetes_data()

    st.write("## Diabetes Correlation Study: ")
    st.info(
        f"Business Requirement 1 - The client is interested in discovering how "
        f"various biomarkers in female patients correlate between those with "
        f"diabetes and without diabetes. The client expects to better "
        f"understand this by reviewing data visualizations of the biomarker "
        f"variables. "
    )
    
    # Inspecting the dataset
    if st.checkbox("Inspect Diabetes Dataset"):
        st.write(
            f"The diabetes dataset contains {df.shape[0]} rows and "
            f"{df.shape[1]} columns. 'Outcome' is the target variable, which "
            f"we will be correlating feature sets against this variable. \n"
            f"\nThe first 10 rows will be shown below."
            )

        st.write(df.head(10))
