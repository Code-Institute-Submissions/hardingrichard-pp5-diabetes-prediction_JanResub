import numpy as np

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