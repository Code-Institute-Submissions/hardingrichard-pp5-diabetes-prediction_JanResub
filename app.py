import streamlit as st
from app_pages.multipage import MultiPage

# Loads scripts for the dashboard pages
from app_pages.page_introduction import page_introduction_body
# from app_pages.page_correlation_study import page_correlation_study_body
# from app_pages.page_hypothesis import page_hypothesis_body
# from app_pages.page_machine_learning_model import page_machine_learning_model_body
# from app_pages.page_diabetes_prediction import page_diabetes_prediction_body

# Creates an instance of the app
app = MultiPage(app_name = "Diabetes Prediction")

# App pages for the dashboard in the sidebar
app.add_page("Introduction", page_introduction_body)
# app.add_page("Diabetes Correlation Study", page_correlation_study_body)
# app.add_page("Project Hypothesis", page_hypothesis_body)
# app.add_page("Machine Learning Model", page_machine_learning_model_body)
# app.add_page("Diabetes Predictor", page_diabetes_prediction_body)

app.run()