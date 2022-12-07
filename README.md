# Diabetes Prediction

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [User Stories](#user-stories)
3. [Business Requirements](#business-requirements)
4. [Hypothesis](#hypothesis)
5. [Rationale](#rationale)
6. [ML Business Case](#ml-business-case)
7. [Dashboard Design](#dashboard-design)
8. [Bug fixes](#bug-fixes)
9. [Deployment](#deployment)
10. [Libraries](#libraries)
11. [Credits](#credits)


## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
* The dataset was originally from the National Institute of Diabetes and Digestive and Kidney Diseases.  
*The dataset relates to female subjects based on how many pregnancies they have been through and how various biomarkers correlate to an outcome of being diabetic. From this dataset predictive analytics can be used to assist healthcare professionals in providing a prediction on whether or not a patient has diabetes.
* The dataset consists of 768 rows and 9 feature sets which represent the following predictor variables: Glucose (Plasma glucose concentration), Blood pressure (Diastolic blood pressure measured in mm Hg), Skin Thickness (Tricep skin fold measured in mm), Insulin levels (2-hour serum level measured in mu U/ml), BMI (Body Mass Index measured in weight Kg/height m^2), Diabetes Pedigree Function (Scores likelihood of diabetes based on family history), Age (the age of the subject) and a target variable of Outcome (indicator for whether the patient has diabetes)


|Variable|Meaning|Units|Data Type|
|:----|:----|:----|:----|
|Pregnancies|Number of times pregnant|0-17|int|
|Glucose|Plasma glucose concentration using a glucose tolerance test|0 - 199|int|
|BloodPressure|Diastolic blood pressure (mm Hg)|0 - 122|int|
|SkinThickness|Tricep skin fold thickness (mm)|0-99|int|
|Insulin|2-hour serum insulin (mu U/ml)|0-846|int|
|BMI|Body Mass Index (kg/m^2)|0-67.1|num|
|DiabetesPedigreeFunction|Scores likeliness of diabetes based on family history| 0.08-2.42|num|
|Age|Age of the subject|21-81|int|
|Outcome|Class variable of 0(500) or 1(268)|0-1|int|


## User Stories
|As a |I want to |Satisfied? (Y/N)|
|:----|:----|:----|
|Client|Be able to get a prediction on whether a patient has diabetes|Y|
|Client|Make use of a dashboard to easily input figures and have a resulting outcome|Y|
|Client|See visualisation of data variables to make it easier to interpret|Y|
|Client|See how biomarkers correlate to the likeliness of diabetes|Y|
|Client|Have a brief summary of what each data column means for understanding data values|Y|
|Client|Have a summary of the analysed data for better understanding|Y|
|Client|Have access to a live website to easily access the dashboard at any time|Y|
|Client|Understand the content of the data and where it is sourced|Y|
|Client|See if the data was prepared in any way for modeling for better insight to the process|Y|
|Client|See what the ratio used for the training of the machine learning model to better understand the tuning of the model|Y|
|Client|Have an accuracy score of 75% or greater with prediction whether or not a patient has diabetes|Y|
|Data Practitioner|Have a relevant dataset in order to conduct analysis and create a ML model|Y|
|Data Practitioner|Be able to make predictions off of the dataset following the creation of a ML model|Y|
|Data Practitioner|Be able to clean the dataset of any unusable or incomplete data for use with analysis and the machine learning model|Y|
|Data Practitioner|Create a machine learning model for predicting the likeliness of diabetes|Y|
|Data Practitioner|Be able to optimise the performance of the machine learning model used for a more accurate score|Y|
|Data Practitioner|Create a dashboard to visualise the data and enable the end user to make use of future predictions|Y|
|Data Practitioner|Have a live website where new dashboard implementations can be developed|Y|


## Business Requirements
Diabetes is a growing disease which affects a large portion of the population around the world. Accurate and opportune predictions of diabetes is important due to the complications that it can pose on other life-threatening diseases. A medical center in Phoenix, Arizona is looking to improve their diagnoses of diabetes using the assistance of machine learning and investigating how best to use previous patient data in order to better predict whether or not future patients are likely to have diabetes.

* 1 - The client is interested in discovering how various biomarkers in female patients correlate to the likeliness of developing diabetes. The client expects to better understand this by reviewing data visualizations of the biomarker variables.
* 2 - The client requires a machine learning tool that their healthcare practitioners can use to identify whether a patient has diabetes.
* 3 - The client expects an accuracy score of 75% or higher in predicting the outcome of diabetes.
* 4 - The client expresses that the machine learning tool can be easily accessed via a dashboard with the ability to input patient data to produce a prediction for diagnosis which can be used to better support the patient.


## Hypothesis
The following hypothesis will help guide the direction of data analysis for the above dataset.
* Based off of my understanding of diabetes, High blood sugar levels are likely to be a primary predictor of the outcome of a diabetes diagnosis.
    * Visualisation to be used and correlation study to validate
* I predict that Age will have an affect on the correlation of diabetes
    * Visualisation to be used and correlation study to validate
* I also predict that BMI may hold some weight in the prediction of diabetes
    * Visualisation to be used and correlation study to validate
* Biomarkers such as blood pressure and skin thickness will likely hold a low level of correlation for whether or not a patient has diabetes.
    * Visualisation to be used and correlation study to validate
* Due to the dataset being fairly small in size and splitting the data between train and test, may pose difficulties in achieving a high accuracy score.
    * Experiment with Train/Test split ratios or potential use of other ML models to achieve a better score.


## Rationale
* Business Requirement 1 - Correlation and Visualisation of Data
    * Inspect the data relating to the biomarkers in the diabetes dataset
    * Make use of a Correlation Study to help understand which variables are strongly linked to the outcome of diabetes
    * Standardisation of data required to analyse due to 0 values on some variables which hold no worth
* Business Requirement 2 - Build a Machine Learning model
    * Split data into two categories, training data and testing data
* Business Requirement 3 - Optimise accuracy of ML tool
    * Having trained the model, testing data will be used to determine the accuracy score
    * Fine tuning of the model with hyperparameters to increase accuracy score.
* Business Requirement 4 - Creation of a Dashboard for user input
    * Creation of a user interface where a summarised description of model can be viewed
    * Allow for manual data input for the model to predict diabetes outcome


## ML Business Case
* A Machine Learning model is required to predict the outcome of a patient having diabetes or not.


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)


## Bug Fixes
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.


## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly in case all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.


## Libraries
* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open Source site
- The images used for the gallery page were taken from this other open-source site



### Acknowledgements (optional)
* In case you would like to thank the people that provided support through this project.

