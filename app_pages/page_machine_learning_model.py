import streamlit as st
import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
import pickle
# from src.data_management.diabetes_data import load_pkl_file

def page_machine_learning_model_body():
    """
    Function that displays the ML pipeline and elements on page
    """

    # Loads the files and trained model
    # load_svm_model = pickle.load(
    #     open('outputs/svm_pipeline/predict_diabetes/v1/trained_svm.sav', 'rb')
    #     )
    
    # x_train = pd.read_csv(
    #     f"outputs/svm_pipeline/predict_diabetes/v1/x_train.csv"
    #     )
    # x_test = pd.read_csv(
    #     f"outputs/svm_pipeline/predict_diabetes/v1/x_test.csv"
    #     )
    # y_train = pd.read_csv(
    #     f"outputs/svm_pipeline/predict_diabetes/v1/y_train.csv"
    #     )
    # y_test = pd.read_csv(
    #     f"outputs/svm_pipeline/predict_diabetes/v1/y_test.csv"
    #     )

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

    st.write("## Support Vector Machine Model Pipeline to Predict Diabetes")
    st.code(
        """
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

        """
    )

    st.write("## Pipeline Performance: ")
    st.write(
        f"### Model Evaluation \n"
        f"Train dataset Accuracy Score:  0.7866449511400652 \n"
        f"\nTest dataset Accuracy Score:  0.7727272727272727 \n"
    )
    st.info(
        """
        * As we can see the Accuracy score is showing at 0.79 rounded up which is above the Business Requirement 3 of needing a score of at least 0.75. If the score was below this then it would be deemed a fail. However, as we are above the 0.75 minimum requirement this can be considered a success.

* As we can see the Accuracy score is showing at 0.77 rounded which is just above the Business Requirement 3 of needing a score of at least 0.75. If the score was below this then it would be deemed a fail. However, as we have met the 0.75 minimum requirement this can be considered a success.

* As the training and test data has output similar Accuracy Scores it is a good indication that the model is not overtrained. If the accuracy score was high in the training data and the test data was low then this would be a signal that the model is overfitted.

* Unfortunately one of the limitations we have is due to the low size of the dataset, it is difficult to get a high accuracy as there isn't a lot of training data for the model to then use with the test data.
    """
    )

    st.write(
        f"### Confusion Matrix Train Output: \n"
        f"\nTrue Positive Predictions: 359 \n"
        f"\nFalse Negative Predictions: 41 \n"
        f"\nFalse Positive Predictions: 90 \n"
        f"\nTrue Negative Predictions: 124 \n"
    )
    st.write(
        f"### Confusion Matrix Test Output: \n"
        f"\nTrue Positive Predictions: 91 \n"
        f"\nFalse Negative Predictions: 9 \n"
        f"\nFalse Positive Predictions: 26 \n"
        f"\nTrue Negative Predictions: 28 \n"
    )

    st.info(
        """
        * The true positive value represents the number of times the model correctly predicted the Diabetic outcome (1).

* The false negative value represents the number of times the model incorrectly predicted a Non-Diabetic outcome(0) for the dataset values that was actually Diabetic (1).

* The false positive value represents the number of times the model incorrectly predicted the Diabetic outcome (1) for the dataset values that was actually Non-Diabetic (0). 

* The true negative value represents the number of times the model correctly predicted the Non-Diabetic outcome (0).

* As we can see above, the confusion matrix shows us that the model is predicting a higher number of true positive and true negative predictions compared to that of the false positive and false negative predictions. This indicates to us that the model is performing well on both the train and test datasets.
"""
    )

    st.write(
        f"## Predictive Power Score: \n"
        f"### Train Dataset: \n"
        f"\n**Precision:** \n" 
        f"\n 0.80 (0) \n"
        f"\n 0.75 (1) \n"
        f"\n**Recall:** \n" 
        f"\n 0.90 (0) \n"
        f"\n 0.58 (1) \n"
        f"\n**F1-Score:** \n" 
        f"\n 0.85 (0) \n"
        f"\n 0.65 (1) \n"
        f"\n**Support:** \n" 
        f"\n 400 (0) \n"
        f"\n 214 (1) \n"
        f"\n**Accuracy:** 0.79 \n"
        f"\n**Macro avg:** 0.75 \n"
        f"\n**Weighted avg:** 0.78 \n"
        f"\n"
        f"### Test Dataset: \n"
        f"\n**Precision:** \n" 
        f"\n 0.78 (0) \n"
        f"\n 0.76 (1) \n"
        f"\n**Recall:** \n" 
        f"\n 0.91 (0) \n"
        f"\n 0.52 (1) \n"
        f"\n**F1-Score:** \n" 
        f"\n 0.84 (0) \n"
        f"\n 0.62 (1) \n"
        f"\n**Support:** \n" 
        f"\n 100 (0) \n"
        f"\n 54 (1) \n"
        f"\n**Accuracy:** 0.77 \n"
        f"\n**Macro avg:** 0.73 \n"
        f"\n**Weighted avg:** 0.76 \n"
        )

    st.info(
        """
        * The classification report provides a detailed breakdown of the evaluation metrics for the output of "Diabetic" (1) and "Non-Diabetic" (0) in both the training and test datasets.

* Let's now asses what each score means:
    * Precision: Precision is the number of true Diabetic predictions made by the model, divided by the total number of Diabetic predictions made by the model. It measures the proportion of Diabetic predictions that are actually correct.
    * Recall: Recall is the number of true Diabetic predictions made by the model, divided by the total number of actual Diabetic cases in the data. It measures the proportion of actual Diabetic cases that were correctly predicted by the model.
    * f1-score: The f1-score is the harmonic mean of precision and recall. It is a balance between precision and recall and reaches its best value at 1.
    * Support: Support is the number of samples of the true response that lies in the outcome of Diabetic(1) and Non-Diabetic(0).


* As we can see the report also returns the overall accuracy score  which is the proportion of correct predictions and confirms what we saw further above. From this, we can see that the model is performing well.
"""
        )