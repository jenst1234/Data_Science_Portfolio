# **Titanic: Survival Prediction & Exploratory Data Analysis (EDA)**

<img src="./images/titanic.png" alt="Alt-Text" width="100%" height="400px" />

[![Data-Science-Projects](https://img.shields.io/badge/Data_Science_Projects-GitHub_Page-%2300BFFF.svg)](https://jenst1234.github.io), [![Notebooks](https://img.shields.io/badge/Notebooks-View-Green.svg)](https://github.com/jenst1234/Data_Science_Portfolio/blob/main/2%23%20Product%20Delivery%20Prediction/notebooks/e_commerce.ipynb)


## **Project Background**
The sinking of the Titanic in April 1912 remains one of the most infamous disasters in maritime history. On her maiden voyage, the Titanic collided with an iceberg, leading to the loss of over 1500 lives. This project employs machine learning to analyze and predict which passengers might have survived based on various attributes such as age, gender, ticket class, and more.

## **Objectives**
The main goal of this project is to develop predictive models that can effectively forecast the survival chances of passengers. This involves:
- **Exploratory Data Analysis (EDA)**: Examining data for patterns, anomalies, or correlations between variables that might influence survival outcomes.
- **Predictive Modelling**: Using insights gained from EDA to inform the development and tuning of machine learning models.
- **Feature Utilization**: Analyzing how different features, especially transformed age groups, affect predictions and potentially inform future safety planning and risk assessment.

## **About Data**
The data used comes from the well-known Titanic dataset available on [Kaggle](https://www.kaggle.com/competitions/titanic), which includes passenger information as described in the data dictionary below:

| Variable    | Definition                                     | Key                                            |
|-------------|------------------------------------------------|------------------------------------------------|
| PassengerId | Unique identifier for each passenger           |                                                |
| Survived    | Survival status                                | 0 = No, 1 = Yes                                |
| Pclass      | Ticket class                                   | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| Name        | Full name of the passenger                     |                                                |
| Sex         | Gender of the passenger                        |                                                |
| Age         | Age in years                                   |                                                |
| SibSp       | Number of siblings / spouses aboard the Titanic|                                                |
| Parch       | Number of parents / children aboard the Titanic|                                                |
| Ticket      | Ticket number                                  |                                                |
| Fare        | Passenger fare                                 |                                                |
| Cabin       | Cabin number                                   |                                                |
| Embarked    | Port of embarkation                            | C = Cherbourg, Q = Queenstown, S = Southampton |

## **Data Preparation**
- **Data Cleaning**: Addressing missing values, especially in 'Age', 'Cabin', and 'Embarked'.
- **Feature Engineering**: Transforming 'Age' into categorical age groups prior to EDA, to better analyze its impact on survival.

## **Exploratory Data Analysis (EDA)**
- **Visual Analysis**: Using plots to explore relationships between survival and features such as 'Pclass', 'Sex', 'Age Groups', and 'Embarked'.
- **Statistical Analysis**: Conducting correlation studies to identify significant predictors of survival.

## **Model Development**
1. **Initial Model Setup**: Deployment of baseline models including Decision Trees, Random Forest and XGBoost.
2. **Model Evaluation**: Assessment based on Accuracy, Precision, Recall, F1-Score, ROC-AUC score, and cross-validation scores.
3. **Model Optimization**: Hyperparameter tuning of promising models like Random Forest and XGBoost.
4. **Feature Importance Analysis**: Evaluating the impact of critical features on the predictions, particularly 'Sex', 'Pclass', 'Fare', and 'Age Groups'.

## **Results**
- **XGBoost Tuned Model**: Achieved the highest accuracy of 0.88 and an AUC score of 0.92.
- **Random Forest Tuned Model**: Showcased robust performance with an AUC score of 0.95.
- **Feature Insights**: Identified 'Sex', 'Pclass', 'Fare', and 'Age Groups' as pivotal in determining survival, supported by EDA and historical accounts.

## **Conclusions and Future Work**
The project illustrates the capability of machine learning in interpreting historical data and making crucial predictions. Potential improvements include expanding data sources, refining models with additional features, and exploring more complex algorithms. Insights from this study could enhance safety protocols and risk assessments in analogous scenarios, thereby improving emergency management strategies.
