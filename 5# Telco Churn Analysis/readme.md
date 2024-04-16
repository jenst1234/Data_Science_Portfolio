# **Teleconfia Customer Churn Prediction - Florida Market Entry**

<img src="./images/churn.png" alt="Alt-Text" width="100%" height="300px" />

[![Data-Science-Projects](https://img.shields.io/badge/Data_Science_Projects-GitHub_Page-%2300BFFF.svg)](https://jenst1234.github.io)

This project aims to support Teleconfia's US market entry, specifically focusing on Florida, by leveraging customer churn prediction to optimize marketing efforts and reduce customer loss.

## **Project Goals:**

- **Identify High-Risk Areas:** Utilize customer data to pinpoint the four Florida city areas with the highest customer churn rates. This will guide targeted billboard marketing campaigns for maximum reach and impact.
- **Predict Individual Churn:** Develop a churn prediction model to identify individual customers at high risk of leaving Teleconfia's network. This information will enable targeted direct marketing actions like personalized calls and special offers.
- **Data-Driven Insights:** Analyze relevant datasets to uncover factors that contribute to customer churn. This will inform strategic decision-making and customer retention efforts.
- **Model Development & Visualization:** Build a robust churn prediction model based on identified variables. Establish a critical probability threshold for defining likely churners. Compelling visualizations will be created to support presentations and facilitate data-driven decision-making.

## **Dataset:**

The project utilizes a comprehensive customer behavior and service usage dataset obtained from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). This data provides insights into customer accounts, geographic details, and churn behavior at Teleconfia.

| Feature                    | Data Type   | Description |
|----------------------------|-------------|-------------|
| `account_length`           | Integer     | Number of days the customer has been with the telecom provider. |
| `international_plan`       | Categorical | Whether the customer has a special pricing plan for international calls (`yes` or `no`). |
| `voice_mail_plan`          | Categorical | Whether the customer has a special pricing plan for voice mail services (`yes` or `no`). |
| `number_vmail_messages`    | Integer     | Number of voice mail messages the customer has. |
| `total_day_minutes`        | Float       | Total duration (in minutes) of the customer's calls during the day. |
| `total_day_calls`          | Integer     | Total number of calls the customer made during the day. |
| `total_day_charge`         | Float       | Total charges for the customer's calls during the day. |
| `total_eve_minutes`        | Float       | Total duration (in minutes) of the customer's calls during the evening. |
| `total_eve_calls`          | Integer     | Total number of calls the customer made during the evening. |
| `total_eve_charge`         | Float       | Total charges for the customer's calls during the evening. |
| `total_night_minutes`      | Float       | Total duration (in minutes) of the customer's calls during the night. |
| `total_night_calls`        | Integer     | Total number of calls the customer made during the night. |
| `total_night_charge`       | Float       | Total charges for the customer's calls during the night. |
| `total_intl_minutes`       | Float       | Total duration (in minutes) of the customer's international calls. |
| `total_intl_calls`         | Integer     | Total number of international calls the customer made. |
| `total_intl_charge`        | Float       | Total charges for the customer's international calls. |
| `customer_service_calls`   | Integer     | Number of calls the customer made to customer service, e.g., for technical issues. |
| `churn`                    | Categorical | Whether the customer has left the service (`True` or `False`). |
| `local_area_code`          | Categorical | The local area code of the customer. |
| `phone_num`                | Categorical | The customer's phone number without the area code. |

## **Key Features:**

- Account tenure (account_length)
- International calling plan subscription (international_plan)
- Voice mail plan subscription (voice_mail_plan)
- Customer service call history (customer_service_calls)
- Call usage patterns (total_day_minutes, total_eve_minutes, total_night_minutes)
- Churn status (churn)

## **Methodology:**

The project employs machine learning techniques, specifically logistic regression, to build a churn prediction model. Seaborn and Matplotlib libraries will be used for data visualization and exploration.

## **Outcomes:**

- Identification of the four Florida city areas with the highest churn rates.
- A churn prediction model to identify individual customers at risk of leaving Teleconfia.
- Actionable insights into factors influencing customer churn.
- Compelling visualizations for clear communication and decision-making.

This project presents valuable insights to guide Teleconfia's Florida market entry strategy, enabling them to prioritize high-risk areas, target at-risk customers, and ultimately minimize customer churn.
