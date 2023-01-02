import streamlist as st
from app_pages.multipage import MultiPage

# Loads scripts for the dashboard pages
from app_pages.page_introduction import page_introduction_body
from app_pages.page_correlation_study import page_correlation_study_body
from app_pages.page_hypothesis import page_hypothesis_body
from app_pages.page_machine_learning_model import page_machine_learning_model_body
from app_pages.page_diabetes_prediction import page_diabetes_prediction_body

app = MultiPage(app_name = "Diabetes Prediction")