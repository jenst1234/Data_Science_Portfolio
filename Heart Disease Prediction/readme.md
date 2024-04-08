# **Heart Disease Prediction Project Overview**

<img src="./images/heart_disease.png" alt="Alt-Text" width="100%" height="400px" />

[![Data-Science-Projects](https://img.shields.io/badge/Data_Science_Projects-GitHub_Page-%2300BFFF.svg)](https://jenst1234.github.io)

## **Introduction**

Heart disease is a leading global health concern. Early detection and risk assessment are crucial for preventative measures. This project explores the potential of machine learning to predict an individual's risk of heart disease.

## **Data**

The project utilizes a dataset from the UCI Machine Learning Repository (Cleveland database). This dataset contains information for 303 individuals, with 14 features encompassing:

- **Biometric data:** Age, blood pressure, cholesterol levels, resting heart rate
- **Lifestyle factors:** Chest pain type
- **Medical history:** ST-segment depression, number of major vessels
The target variable is the presence or absence of heart disease.

## **Methodology**

Various machine learning models were employed to analyze the data and build predictive models:

- **Logistic Regression:** A classic approach for classification problems, suitable for understanding the relationship between features and heart disease.
- **Naive Bayes:** A probabilistic model that works well with limited data and can handle features with different scaling.
- **Random Forest:** An ensemble method combining multiple decision trees for improved accuracy and handling complex relationships.
- **Decision Trees:** A model offering interpretability, allowing visualization of the decision-making process for risk prediction.
- **XGBoost:** A powerful machine learning model known for its high accuracy in various tasks, including predicting heart disease.

## **Results**

- **Model Performance:** The project achieved promising results, with the - best models (XGBoost and Random Forest) exceeding 90% accuracy in predicting heart disease.
- **Key Risk Factors:** The analysis identified critical factors influencing heart disease risk, including age, gender, blood pressure, cholesterol levels, resting heart rate, chest pain type, ST-segment depression, and the number of major vessels.

## **Significance**

This project demonstrates the effectiveness of machine learning in predicting heart disease. Early risk assessment can enable healthcare professionals to:

- Implement preventive measures like lifestyle modifications or medication.
- Prioritize patients for further diagnostic testing and treatment.
- Develop personalized healthcare plans to manage risk factors.

## **Next Steps**

- **Model Optimization:** Refining the models by tuning hyperparameters and potentially using feature engineering techniques.
- **Data Integration:** Exploring the inclusion of additional data sources like medical history or genetic information for potentially more comprehensive risk assessment.
- **Web-Based Tool:** Developing a user-friendly web application allowing individuals to input their health data and receive a personalized risk assessment.

This project contributes to the advancement of machine learning in healthcare by providing a promising tool for heart disease prevention. Future developments can enhance the accuracy and clinical utility of this approach, ultimately promoting positive health outcomes.
