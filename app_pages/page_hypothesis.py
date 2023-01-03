import streamlit as st
import pandas as pd

def page_hypothesis_body():
    """
    Function that loads the elements of the hypothesis page.
    """

    st.write("## Project Hypotheses:")
    st.info(
        """
        * At the start of project inception, a set of hypotheses were proposed 
        relating to the diabetes dataset and the variables within and their 
        link to the outcome of Diebetes.
        * After analysing the data in the 02-CorrelationStudy notebook, we have
        gained the following insights:        
        """
    )

    st.success(
        "* Based off of my understanding of diabetes, High blood sugar levels "
        "are likely to be a primary predictor of the outcome of a diabetes "
        "diagnosis. \n"
        "\nThis hypothesis was determined to be **Correct** where Blood sugar "
        "levels had the largest impact on the outcome of a diabetes diagnosis\n"
        "* I predict that Age will have an affect on the correlation of "
        "diabetes \n"
        "\nThis hypothesis was determined to be **Correct**, Age played a large "
        "role in determining whether or not the patient was likely to have "
        "diabetes.\n"
        "* I also predict that BMI may hold some weight in the prediction of "
        "diabetes.\n"
        "\nWhilst not as important as the above to variables BMI had an impactful "
        "outcome on whether a patient was diabetic or not so therefor the "
        "hypothesis was determined to be **Correct**.\n"
        "* Biomarkers such as blood pressure and skin thickness will likely "
        "hold a low level of correlation for whether or not a patient has "
        "diabetes.\n"
        "\nThe hypothesis was determined to be **Correct** based on the data "
        "analysis.\n"
        "* Due to the dataset being fairly small in size and splitting the "
        "data between train and test, may pose difficulties in achieving a high "
        "accuracy score. \n"
        "\nThe outcome of training the model resulted in a well-performing "
        "accuracy score that met the business requirements. Which one could "
        "argue, made the hypothesis false however, due to the small dataset "
        "size it would be difficult to improve this score further."
    )