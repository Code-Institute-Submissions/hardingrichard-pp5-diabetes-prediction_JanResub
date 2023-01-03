import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
from src.machine_learning.svm_model import SVM_pipeline


def page_diabetes_prediction_body():
    """
    Function that loads the variable and runs the elements
    within the Diabetes Prediction dashboard page.
    """

    st.write("## Diabetes Prediction:")
    st.info(
        "The client expresses that the machine learning tool can be easily accessed via a dashboard with the ability to input patient data to produce a prediction for diagnosis which can be used to better support the patient."
    )

    # Load the dataset
    df = pd.read_csv(f"outputs/datasets/collection/diabetes.csv")
    x = df.drop(columns = 'Outcome', axis=1)
    y = df['Outcome']

    # Scale the features
    df_scaler = StandardScaler()
    df_scaler.fit(x)
    stnd_data = df_scaler.transform(x)

    x = stnd_data
    y = df['Outcome']

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=2)

    # Build SVM model
    model = SVM_pipeline(tuning_parameter=0.001, iteration_no=1000, lambda_parameter=0.01)
    model.fit(x_train, y_train)

    # Testing the model
    x_train_predict = model.diabetes_prediction(x_train)
    x_test_predict = model.diabetes_prediction(x_test)

    train_accuracy = accuracy_score(x_train_predict, y_train)
    test_accuracy = accuracy_score(x_test_predict, y_test)
    st.write(
        "The training accuracy is:", train_accuracy, 
        "and the test accuracy is:", test_accuracy)

    st.info(
        f"With the below dashboard interface, the client is able to manually "
        f"enter data into the number input fields in order to run predictive "
        f"analytics and predict the outcome of Diabetic or Non-Diabetic"
    )
    # Testing the predictive outcome
    # Random row of data from source dataset
    manual_input = (11,143,94,33,146,36.6,0.254,51)

    st.write("Input Values")

    # Give each input field a label
    labels = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    for value, label in zip(manual_input, labels):
        st.number_input(label, value=value)

    manual_input_nparray = np.asarray(manual_input)
    manual_input_shaped = manual_input_nparray.reshape(1, -1)

    stnd_manual_input = df_scaler.transform(manual_input_shaped)
    predict = model.diabetes_prediction(stnd_manual_input)

    if st.button("Predict outcome"):
        if predict[0] == 0:
            st.write('Based on the data entered. This person does not show signs of being diabetic.')
        else: 
            st.write('Based on the data entered. This person shows signs of being diabetic.')

    