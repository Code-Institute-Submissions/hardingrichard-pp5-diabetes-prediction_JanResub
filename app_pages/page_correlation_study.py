import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Loads the function in data_management for dataset
from src.data_management.diabetes_data import load_diabetes_data



def page_correlation_study_body():
    """
    Function that loads the elements within the correlation study page.
    To display the correlated features
    """

    df = load_diabetes_data()

    st.write("## Diabetes Correlation Study: ")
    st.info(
        f"* Business Requirement 1 - The client is interested in discovering "
        f"how various biomarkers in female patients correlate between those "
        f"with diabetes and without diabetes. The client expects to better "
        f"understand this by reviewing data visualizations of the biomarker "
        f"variables. "
        f"* A sample of the Diabetes dataset can be viewed below "
    )
    
    # Inspecting the dataset
    if st.checkbox("Inspect Diabetes Dataset"):
        st.write(
            f"The diabetes dataset contains {df.shape[0]} rows and "
            f"{df.shape[1]} columns. 'Outcome' is the target variable, which "
            f"we will be correlating feature sets against this variable. \n"
            f"\nThe first 10 rows will be shown below."
            )

        st.write(df.head(10))
    
    st.info(
        f"* A correlation study was carried out in the 02-CorrelationStudy "
        f"notebook. Using a correlation matrix \n"
        f"The visualisations which were analysed in the correlation study "
        f"can be accessed below."
    )

    st.write("---")

        # Conclusions of the Correlation Study
    st.success(
        f"## Conclusions: \n"
        f"* The following variables were found to have the strongest "
        f"correlation with the outcome of having Diabetes. These variables "
        f"were as follows: \n"
        f"'Glucose', 'Age', 'BMI' then having less of an impact were: "
        f"'DiabeticPedigreeFunction', 'Pregnancies' with 'BloodPressure' "
        f"'SkinThickness' and 'Insulin' \n"
        f"* There is a strong correlation between "
        f"Glucose and the Outcome variables suggesting that Glucose "
        f"levels are an important variable for identifying diabetics "
        f"and non-diabetics. \n"
        f"* There is also a significant correlation present between Age "
        f"and Pregnancies as well as Insulin and SkinThickness"
        )

    st.write("---")

    # Visualisations for the Diabetes Correlation Study
    st.write("## Visualisations: ")


    correlation_matrix()
    feature_importance()
    pair_plot()

def correlation_matrix():
    df = load_diabetes_data()

    if st.checkbox("Correlation Matrix"):
        corr = df.corr()
        # Create a figure and a set of subplots
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a heatmap of the correlations
        sns.heatmap(
            corr, 
            annot=True, 
            cmap="YlGnBu", 
            fmt=".2f", 
            xticklabels=corr.columns.values, 
            yticklabels=corr.columns.values
            )

        # Show the plot
        st.pyplot(fig)
    
        st.info(
            f"* From this Correlation Matrix heatmap we can see that the "
            f"darker colours indicate there is a stronger correlation "
            f"between variables than those which are lighter. \n"
            f"* As we can see above there is a strong correlation between "
            f"Glucose and the Outcome variables suggesting that Glucose "
            f"levels are an important variable for identifying diabetics "
            f"and non-diabetics. \n"
            f"* There is also a significant correlation present between Age "
            f"and Pregnancies as well as Insulin and SkinThickness"
        )

def feature_importance():
    df = load_diabetes_data()

    if st.checkbox("Feature Importance"):
        x = df[['Glucose', 'BMI', 'Age', 'Pregnancies', 'SkinThickness',
                'Insulin', 'DiabetesPedigreeFunction', 'BloodPressure',]]
        y = df.iloc[:,8]
        model = ExtraTreesClassifier()
        model.fit(x,y)
        feat_importances = pd.Series(model.feature_importances_, index=x.columns)
        fig, ax = plt.subplots()
        feat_importances.nlargest(20).plot(kind='bar', ax=ax)

        st.pyplot(fig)

        st.info(
            f"* The above graph shows us the order of importance between the "
            f"features and the targert variable 'Outcome' \n"
            f"* Plotting the above graph allows us to clearly see that Glucose "
            f"has the largest importance on the outcome of the subject being "
            f"diabetic followed by BMI and Age. This confirms what we see in "
            f"the correlation matrix."
        )

def pair_plot():
    df = load_diabetes_data()

    if st.checkbox("Pair Plot"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Create a heatmap of the correlations
        sns.pairplot(df)

        # Show the plot
        st.pyplot()

        st.info(
            f"* The above plot charts show us the distribution of each "
            f"variable in relation to one another."
        )