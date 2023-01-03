import streamlit as st
import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
import pickle
from src.data_management.diabetes_data import load_pkl_file

def page_machine_learning_model_body():
    """
    Function that displays the ML pipeline and elements on page
    """

    # Loads the files and trained model
    load_svm_model = pickle.load(open('outputs/svm_pipeline/predict_diabetes/v1.0/trained_svm.sav', 'rb'))
    x_train = pd.read_csv(
        f"outputs/svm_pipeline/predict_diabetes/v1.0/x_train.csv"
        )
    x_test = pd.read_csv(
        f"outputs/svm_pipeline/predict_diabetes/v1.0/x_test.csv"
        )
    y_train = pd.read_csv(
        f"outputs/svm_pipeline/predict_diabetes/v1.0/y_train.csv"
        )
    y_test = pd.read_csv(
        f"outputs/svm_pipeline/predict_diabetes/v1.0/y_test.csv"
        )

    st.write("## SVM Machine Learning Model: Diabetes Predictor")
    st.info(
        f"* A Machine Learning pipeline was developed to answer the Business "
        f"Requirement 2 - The client requires a machine learning tool that "
        f"their healthcare practitioners can use to identify whether a patient "
        f"has diabetes. \n"
        f"* It was decided that a Support Vector Machine model would be used "
        f"to predict the diabetes outcome in order to meet the Business "
        f"Requirement 3 - The client expects an accuracy score of 75% or "
        f"higher in predicting the outcome of diabetes. \n"
        f"* The SVM model pipeline delivered an f1 score greater than "
        f"0.75 on both the train and test sets coming in at **0.79** for the "
        f"train set and **0.77** on the test set. This therefor meets the "
        f"business requirement."
    )