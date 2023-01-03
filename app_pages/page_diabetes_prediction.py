import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm


# Support Vector Machine pipeline for predicting output of Diabetes
class SVM_pipeline():

    # Initiates the hyperparameters
    def __init__(self, tuning_parameter, iteration_no, lambda_parameter):
        self.tuning_parameter = tuning_parameter
        self.iteration_no = iteration_no
        self.lambda_parameter = lambda_parameter

    # Fits the diabetes dataset to the SVM classifier model
    def fit(self, x, y):
        # M refers to the No. of data points (rows) and Y refers to No. of input features (columns)
        self.m, self.n = x.shape

        # Initiate weight and bias values
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

        # Optimisation Algorithm
        for i in range(self.iteration_no):
            self.update_weight_value()
    
    # Encoding the label
    def update_weight_value(self):
        label_y = np.where(self.y <= 0, -1, 1)

        # Conditions for the gradients (dw, db)
        for index, x_i in enumerate(self.x):
            constraint = label_y[index] * (np.dot(x_i, self.w) - self.b) >= 1

            if (constraint == True):
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            
            else:
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, label_y[index])
                db = label_y[index]
                
            # Formula used for updating weight and bias values
            self.w = self.w - self.tuning_parameter * dw
            self.b = self.b - self.tuning_parameter * db
    
    # Predicts the Outcome label when using an input value
    def diabetes_prediction(self, x):
        result = np.dot(x, self.w) - self.b
        label_prediction = np.sign(result)
        predicted_outcome = np.where(label_prediction <= -1, 0, 1)

        return predicted_outcome

def page_diabetes_prediction_body():
    """
    Function that loads the variable and runs the elements
    within the Diabetes Prediction dashboard page.
    """
    df = pd.read_csv(f"outputs/datasets/collection/diabetes.csv")
    df.head(15)
    df.shape

    x = df.drop(columns = 'Outcome', axis=1)
    y = df['Outcome']

    df_scaler = StandardScaler()
    df_scaler.fit(x)
    stnd_data = df_scaler.transform(x)

    x = stnd_data
    y = df['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=2)

    classifier_SVM = SVM_pipeline(tuning_parameter=0.001, iteration_no=1000, lambda_parameter=0.01)

    classifier_SVM.fit(x_train, y_train)

    x_train_predict = classifier_SVM.diabetes_prediction(x_train)
    x_test_predict = classifier_SVM.diabetes_prediction(x_test)

    # Random row of data from source dataset
    manual_input = (1,97,66,15,140,23.2,0.487,22)

    manual_input_nparray = np.asarray(manual_input)
    manual_input_shaped = manual_input_nparray.reshape(1, -1)

    stnd_manual_input = df_scaler.transform(manual_input_shaped)
    predict = classifier_SVM.diabetes_prediction(stnd_manual_input)

    if (predict[0] == 0):
        print('Based on the data entered. This person does not show signs of being diabetic.')
    else: 
        print('Based on the data entered. This person shows signs of being diabetic.')

    st.write("## Diabetes Prediction:")
    st.info(
        "The client expresses that the machine learning tool can be easily accessed via a dashboard with the ability to input patient data to produce a prediction for diagnosis which can be used to better support the patient."
    )

    