# Diabetes Prediction

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis](#hypothesis)
4. [Rationale](#rationale)
5. [ML Business Case](#ml-business-case)
6. [Dashboard Design](#dashboard-design)
7. [Bug fixes](#bug-fixes)
8. [Deployment](#deployment)
9. [Libraries](#libraries)
10. [Credits](#credits)

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


## Business Requirements
Diabetes is a growing disease which affects a large portion of the population around the world. Accurate and opportune predictions of diabetes is important due to the complications that it can pose on other life-threatening diseases. A medical center in Phoenix, Arizona is looking to improve their diagnoses of diabetes using the assistance of machine learning and investigating how best to use previous patient data in order to better predict whether or not future patients are likely to have diabetes.

* 1 - The client is interested in discovering how various biomarkers in female patients correlate to the likeliness of developing diabetes. The client expects to better understand this by reviewing data visualizations of the biomarker variables.
* 2 - The client is also interested in how these biomarkers can be used to predict the likeliness of the patient having diabetes.
* 3 - The client requires a machine learning tool that their healthcare practitioners can use to identify whether a patient has diabetes.
* 4 - The client expects an accuracy score of 75% or higher in predicting the outcome of diabetes.
* 5 - The client expresses that the machine learning tool can be easily accessed via a dashboard with the ability to input patient data to produce a prediction for diagnosis which can be used to better support the patient.


## Hypothesis
The following hypothesis will help guide the direction of data analysis for the above dataset.
* Based off of my understanding of diabetes, High blood sugar levels are likely to be a primary predictor of the outcome of a diabetes diagnosis.
    * Visualisation to be used to show glucose levels and correlation to diabetes
* Biomarkers such as blood pressure and skin thickness will likely hold a low level of correlation for whether or not a patient has diabetes.
* Due to the dataset being fairly small in size and splitting the data between train and test, may pose difficulties in achieving a high accuracy score.

## Rationale
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks

* Business Requirement 1 - Correlation and Visualisation of Data
    * Inspect the data relating to the biomarkers in the diabetes dataset
    * Make use of a Correlation Study to help understand which variables are strongly linked to the outcome of diabetes
    * 
* Business Requirement 2 - Build a Machine Learning model
* Business Requirement 3 - Optimise accuracy of ML tool
* Business Requirement 4 - Creation of a Dashboard for user input


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course 


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

