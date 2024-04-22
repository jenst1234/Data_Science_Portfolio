# **PRODUCT DELIVERY PREDICTION**

<img src="./images/ecommerce.png" alt="Alt-Text" width="100%" height="300px" />

[![Data-Science-Projects](https://img.shields.io/badge/Data_Science_Projects-GitHub_Page-%2300BFFF.svg)](https://jenst1234.github.io) [![Notebooks](https://img.shields.io/badge/Notebooks-View-Green.svg)](https://github.com/jenst1234/Data_Science_Portfolio/blob/main/2%23%20Product%20Delivery%20Prediction/notebooks/e_commerce.ipynb)

## **INTRODUCTION**

### **Objectives**

The primary goal of this project was to develop a predictive model for an international e-commerce company to determine the timeliness of product deliveries. The goal was to unravel the complex factors influencing delivery outcomes and gain a deeper understanding of customer behavior patterns. This project employs advanced machine learning techniques to derive actionable insights from a vast dataset of customer transactions, focusing on enhancing operational efficiency and customer satisfaction.

### **Dataset**

I utilized a dataset from [Kaggle](https://www.kaggle.com/datasets/prachi13/customer-analytics), which includes 10,999 observations across 12 variables, detailing customer purchases and delivery details:

| Column               | Description                                                                           |
|----------------------|---------------------------------------------------------------------------------------|
| ID                   | ID Number of Customers                                                                |
| Warehouse_block      | The company has a large warehouse divided into blocks such as A, B, C, D, E           |
| Mode_of_Shipment     | Modes of product shipment, including Ship, Flight, and Road                           |
| Customer_care_calls  | Number of calls made from enquiry for shipment enquiry                                |
| Customer_rating      | Ratings given by customers, where 1 is the lowest (Worst), 5 is the highest (Best)    |
| Cost_of_the_Product  | Cost of the Product in US Dollars                                                     |
| Prior_purchases      | Number of prior purchases by the customer                                             |
| Product_importance   | Categorization of products based on importance: low, medium, high                     |
| Gender               | Gender of the customer (Male or Female)                                               |
| Discount_offered     | Discount offered on that specific product                                             |
| Weight_in_gms        | Weight of the product in grams                                                        |
| Reached.on.Time_Y.N  | Target variable indicating if the product was delivered on time (0) or not (1)        |

## **MODEL DEVELOPMENT AND EVALUATION**

### **Model Implementation**

We implemented various machine learning models to predict delivery delays, including:

- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- XGBoost
- Logistic Regression
- Gradient Boost
- Support Vector Machine (SVM)
- Naive Bayes

Each model was initially tested in its basic form, followed by hyperparameter tuning to optimize performance. 

### **Feature Importance Analysis**

Feature importance was evaluated to identify the most significant predictors of delivery delays. Our analysis confirmed the following key features:

- **Discount_offered:** Most critical predictor, indicating that higher discounts might correlate with delivery urgency and associated delays.
- **Weight_in_gms and Cost_of_the_Product:** Both crucial in predicting delays, suggesting that heavier and costlier products face more logistical challenges.
- **Less influential features:** Prior purchases and customer care calls also impact delivery timeliness but to a lesser extent.

### **Best Models**

- **Basic and Tuned Random Forest** and **GradientBoost** models achieved an accuracy of 0.75, with the basic Random Forest model showing a high ROC-AUC of 0.86 but exhibiting signs of overfitting.
- **Decision Tree:** Exhibited balanced performance metrics but also displayed significant overfitting, evidenced by its fluctuating training and validation scores.

### **Outlook for Performance Improvement**

To further enhance the models' accuracy, we plan to:

- **Expand Hyperparameter Tuning:** Exploring deeper into model settings to better prevent overfitting.
- **Enrich Feature Set:** Further analysis on less influential features to refine their impact or introduce new data points.
- **Incorporate External Data:** Considering factors like weather or logistic disruptions could enrich the model's predictive capability.
- **Adopt Ensemble Techniques:** Combining multiple models to leverage strengths and mitigate individual weaknesses.
- **Implement Adaptive Learning:** Utilizing online learning to continuously update the model with new data, improving its adaptability to new trends.

These steps will aim to refine our predictive capabilities, ensuring the e-commerce platform can better manage its delivery operations and enhance customer satisfaction.
