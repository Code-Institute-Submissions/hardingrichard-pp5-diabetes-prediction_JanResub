import streamlit as st

# Function to display the content of the Introduction page
def page_introduction_body():
    """
    Function which will display the body of the project introduction page
    """

    st.write('## Project Introduction:')
    st.success(
        f"\n Diabetes is a growing disease which affects a large portion "
        f"of the population around the world. Accurate and opportune "
        f"predictions of diabetes is important due to the complications that "
        f"it can pose on other life-threatening diseases. \n"
        f"A medical center in Phoenix, Arizona is looking to improve their "
        f"diagnoses of diabetes using the assistance of machine learning and "
        f"investigating how best to use previous patient data in order to "
        f"better predict whether or not future patients are likely to have "
        f"diabetes. \n"
        f"\n This project will consist of 4 business requirements seen below: \n"
        f"\n 1. The client is interested in discovering how various "
        f"biomarkers in female patients correlate between those with diabetes "
        f"and without diabetes. The client expects to better understand this "
        f"by reviewing data visualizations of the biomarker variables. \n"
        f"\n 2. The client requires a machine learning tool that their "
        f"healthcare practitioners can use to identify whether a patient has "
        f"diabetes. \n"
        f"\n 3. The client expects an accuracy score of 75% or higher in "
        f"predicting the outcome of diabetes. \n"
        f"\n 4. The client expresses that the machine learning tool can be "
        f"easily accessed via a dashboard with the ability to input patient "
        f"data to produce a prediction for diagnosis which can be used to "
        f"better support the patient."
    )

    st.warning(
        f"For further information regarding the above and for the rest of the "
        f"project, please read the [Diabetes Prediction README file]" + 
        f"(https://github.com/hardingrichard/pp5-diabetes-prediction) found on "
        f"github."
    )
    
    st.write('## Diabetes Dataset:')
    st.info(
        f"### About the Dataset: \n"
        f"\n * The dataset is sourced from [Kaggle](https://www.kaggle.com/" +
        f"datasets/uciml/pima-indians-diabetes-database). \n"
        f"* The dataset was originally from the National Institute of Diabetes "
        f"and Digestive and Kidney Diseases. \n" 
        f"* The dataset relates to female subjects based on how many "
        f"pregnancies they have been through and how various biomarkers "
        f"correlate to an outcome of being diabetic. From this dataset "
        f"predictive analytics can be used to assist healthcare professionals "
        f"in providing a prediction on whether or not a patient has diabetes.\n"
        f"* The dataset consists of **768 rows** and **9 feature sets** which "
        f"represent the following predictor variables: \n"
        f"    * Pregnancies (The number of pregnancies had) \n"
        f"    * Glucose (Plasma glucose concentration) \n" 
        f"    * Blood pressure (Diastolic blood pressure measured in mm Hg) \n"
        f"    * Skin Thickness (Tricep skin fold measured in mm) \n"
        f"    * Insulin levels (2-hour serum level measured in mu U/ml) \n"
        f"    * BMI (Body Mass Index measured in weight Kg/height m^2) \n"
        f"    * Diabetes Pedigree Function (Scores likelihood of diabetes "
        f"based on family history) \n"
        f"    * Age (the age of the subject in years) \n"
        f"\n and lastly also a target variable: \n"
        f"    * Outcome (indicator for whether the patient has diabetes) \n"
    )