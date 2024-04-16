import streamlit as st 
import os as os
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import itertools
import joblib
import pickle
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform

#Theme and design
header_divider = 'green' 

data_dict = {
    'Feature Name': [
        'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'TAN', 'T', 'Va', 'Ve', 'Mk', 'Cn', 'Ct', 'Cr', 'r',
        'm (kg)', 'Mt', 'Enedc (g/km)', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)',
        'ep (KW)', 'z (Wh/km)', 'Electric range (km)', 'IT', 'Erwltp (g/km)', 'Erwltp (g/km) (WLTP)', 'De', 'Vf',
        'Date of registration', 'Fuel consumption', 'Status', 'Type of data', 'year'
    ],
    'Feature Re-Name': [
        'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'TAN', 'T', 'Va', 'Ve', 'Mk', 'Cn', 'Ct', 'Cr', 'r',
        'm_kg', 'Mt', 'Enedc_g/km', 'Ewltp_g/km', 'W_mm', 'At1_mm', 'At2_mm', 'Ft', 'Fm', 'ec_cm3',
        'ep_KW', 'z_Wh/km', 'Er_km', 'IT', 'Erwltp_g/km', 'Erwltp_g/km)_WLTP', 'De', 'Vf',
        'Dr', 'Fc', 'Status', 'Type of data', 'year'
    ],
    'Meaning': [
        'Country', 'Vehicle family identification number', 'Pool', 'Manufacturer name (EU standard)',
        'Manufacturer name (OEM declaration)', 'Manufacturer name (MS registry denomination)', 'Type approval number',
        'Type', 'Variant', 'Version', 'Make', 'Commercial name', 'Category of the vehicle type approved',
        'Category of the vehicle registered', 'Total new registrations', 'Mass in running order (kg)',
        'WLTP test mass', 'Specific CO2 Emissions in g/km (NEDC)', 'Specific CO2 Emissions in g/km (WLTP)',
        'Wheel base in mm', 'Axle width steering axle in mm', 'Axle width other axle in mm', 'Fuel type',
        'Fuel mode', 'Engine capacity in cm3', 'Engine power in KW', 'Electric energy consumption in Wh/km',
        'Electric range (km)', 'Innovative technology or group of innovative technologies',
        'Emissions reduction through innovative technologies in g/km',
        'Emissions reduction through innovative technologies in g/km (WLTP)', 'Deviation factor', 'Verification factor',
        'Date of registration', 'Fuel consumption', 'Status', 'Type of data', 'Registration year'
    ]
    }

#######################################################################################################################################################
#Introduction
#######################################################################################################################################################

def intro():
    st.header("Reducing Car Emissions with Data Science", divider=header_divider)

    content = """
    **Why it matters:**
    - Cars contribute a lot to climate change by emitting a harmful gas called CO²
    - We urgently need new and smart ways to make cars more environmentally friendly

    **What we're doing:**
    - We're using computer techniques such as ML and DL to study and predict car emissions
    - We're using these techniques to understand cars better and help make them cleaner

    **Where we get our info:**
    - We're using a big set of data from the European Environment Agency (EEA)
    - This data tells us about different countries, types of cars, and how they affect the environment

    
    **Goals:**
    - Sorting cars into different Classes based on CO² Emission
    - Predicting CO² Emission levels for each car 
    - Explore how different car features influence the amount of CO² emissions produced

    """

    st.markdown(content)

def display_dataset_info():
    st.header('General description', divider=header_divider)
    # Add a hyperlink to the first occurrence of "datasets"
    dataset_link = "https://co2cars.apps.eea.europa.eu/?source=%7B%22track_total_hits%22%3Atrue%2C%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22constant_score%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22year%22%3A2022%7D%7D%5D%7D%7D%2C%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22scStatus%22%3A%22Provisional%22%7D%7D%5D%7D%7D%5D%7D%7D%7D%7D%5D%7D%7D%2C%22display_type%22%3A%22tabular%22%7D"

    # Modify the text to include the hyperlink
    text_with_link = (
        "The European Environment Agency (EEA) provides extensive "
        f"[datasets]({dataset_link}) from 2010 to 2022 for studying vehicle carbon dioxide (CO\u00B2) emissions. "
        "To manage computational constraints, we narrowed our focus to the 2021 dataset, initially with 9,920,108 observations and 37 features. "
        "For efficiency, we filtered the data for France, resulting in 1,777,878 observations. "
        "This allows streamlined computations while maintaining a robust sample for a comprehensive analysis of CO\u00B2 emissions in the French context."
    )

    # Display the text with the hyperlink
    st.markdown(text_with_link)

def display_dataframe_info(df):
    st.subheader("DataFrame Info")
    
    # Display basic info
    st.text(f"Number of Rows: {df.shape[0]}")
    st.text(f"Number of Columns: {df.shape[1]}")
    miss_val_flag = df.isna().any().any()
    
    if miss_val_flag:
        st.text(f"The Dataframe includes missing values")
    else :
        st.text(f"The Dataframe doesn't include any missing values")
    
    # Display DataFrame
    with st.expander('Show/Hide Dataframe'):
        st.table(df.head(4)) 
    # Display more detailed info
    st.text("Data Types:")
    st.write(df.dtypes)

    #st.text("Missing Values:")
    #st.write(df.isnull().sum())

    
def feature_meaning():
    st.subheader('Feature description', header_divider)
    # illustration
    df_illustration = pd.DataFrame(data_dict)
    
    # Display the table in Streamlit
    with st.expander("Show/Hide DataFrame Features"):
        st.table(df_illustration)

# Load Dataframe is disabled to spare time for defense  
@st.cache_data
def load_dataframe(df_filepath):
    df = pd.read_csv(df_filepath)
    return df 

@st.cache_data
def filter_country_rename_features(df, country='FR'):
    df = df[df.Country==country]
    renaming_mapping = {col : col.replace(' ', '_')
                    .replace(')', '')
                    .replace('(', '')
                    for col in df.columns}
    renaming_mapping['Electric range (km)'] = 'Er_km'
    renaming_mapping['Fuel consumption '] = 'Fc'
    renaming_mapping['Date of registration'] = 'Rd'
    df.rename(columns = renaming_mapping, inplace = True)
    
    return df 
    
@st.cache_data
def manufacturers_distribution_pie_chart(img_path):
    st.header("Car manufacturers distribution")

    # Calculate the count of each manufacturer
    #manufacturer_counts = df['Mp'].value_counts()

    #plt.figure(figsize=(12, 6))
    #plt.pie(manufacturer_counts, labels=manufacturer_counts.index, autopct='%1.1f%%', startangle=140)
    #plt.title("Manufacturers' distribution")

    # Draw circle to make it a pie chart
    #centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    #fig = plt.gcf()
    #fig.gca().add_artist(centre_circle)
    img = Image.open(img_path)
    st.image(img, width=800)

@st.cache_data  
def countries_pie_chart(img_path):
    st.header("Countries distribution")
    img = Image.open(img_path)
    st.image(img, width=800)
    
   

#######################################################################################################################################################
#Preprocessing
#######################################################################################################################################################

    
# Generate a DataFrame indicating missing values
@st.cache_data
def plot_missing_values(df):
    st.header('Feature with missing values')
    # Calculate the percentage of missing values in each column
    missing_percentage = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_percentage = missing_percentage[missing_percentage > 0]  # Filter columns with missing values

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(x=missing_percentage.index, y=missing_percentage, palette='viridis')

    # Customize the plot
    plt.title('Percentage of Missing Values in Features')
    plt.xlabel('Features')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=90)
    
    # Display the plot using Streamlit
    st.pyplot(plt)
        
        
        
# Generate a DataFrame indicating unique values count
@st.cache_data
def plot_unique_values_count(df):
    #st.header('Unique Values Count per Categorical Feature', divider=header_divider)
    
    #Numerical features are not target fo this study 
    numerical_feat = ['m_kg', 'Ewltp_g/km', 'Enedc_g/km','W_mm','At1_mm', 'At2_mm','ec_cm3','ep_KW','z_Wh/km','Erwltp_g/km','Fc','Er_km', 'MMS', 'Vf', 'De', 'Ernedc_g/km']
    filtered_columns = list(set(df.columns) - set(numerical_feat))

    df = df[filtered_columns]
    # Calculate the count of unique values in each column (excluding 'ID')
    unique_values_count = df.drop(columns=['ID']).nunique().sort_values(ascending=False)

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plot = sns.barplot(x=unique_values_count.index, y=unique_values_count, palette='viridis')
    
    
    threshold = 10
    # Add annotations on top of each bar
    for p in plot.patches:
        if p.get_height() <= threshold :
            plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', xytext=(0, 10), textcoords='offset points', rotation=45)
        elif p.get_height() > threshold :
            plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', xytext=(0, 10), textcoords='offset points', rotation=45, color='red')
            p.set_edgecolor('red')

    # Customize the plot
    plt.title('Count of Unique Values in Features')
    plt.xlabel('Features')
    plt.ylabel('Count of Unique Values')
    plt.xticks(rotation=90)

    # Display the plot using Streamlit
    st.pyplot(plt)
    
    
    
def percentage_count_plot(data, column):
    #st.header(f"Percentage Count Plot for {column}", divider=header_divider)
    
    # Calculate the counts and percentages
    counts = data[column].value_counts()
    percentages = counts / len(data) * 100
    
    # Create a DataFrame for easy plotting
    df_counts = pd.DataFrame({'Count': counts, 'Percentage': percentages})
    
    # Plot the percentage count plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df_counts.index, y=df_counts['Percentage'], palette='viridis')
    plt.title(f"Percentage Count Plot for {column}")
    plt.xlabel(column)
    plt.ylabel('Percentage')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    
    
    
def find_feature_meaning(feature_name, data_dict):
    feature_meaning = None
    
    try:
        pos = data_dict['Feature Name'].index(feature_name)
    except ValueError:
        try:                 
            pos = data_dict['Feature Re-Name'].index(feature_name)
        except ValueError:
            feature_meaning = feature_name
    
    if feature_meaning is None:
        feature_meaning = data_dict['Meaning'][pos]
    
    return feature_meaning
    
    
    
def boxplot_by_category(data, x_column, y_column):
    feature_meaning_x_label = find_feature_meaning(x_column, data_dict)
    feature_meaning_y_label = find_feature_meaning(y_column, data_dict)
    
    st.subheader(f"Boxplot of {feature_meaning_y_label} by {feature_meaning_x_label}")
        
    non_empty_categories = data.groupby(x_column).filter(lambda group: not (group[y_column] == 0).all() and not group[y_column].isna().all())
    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x_column, y=y_column, data=non_empty_categories, palette='viridis')
    plt.xticks(rotation=45)
    plt.title(f"Boxplot of {feature_meaning_y_label} by {feature_meaning_x_label}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    st.pyplot(plt)



def display_normalized_value_counts(df, column_name):
    feature_meaning = find_feature_meaning(column_name, data_dict)
    
    # Calculate the normalized value counts
    normalized_counts = df[column_name].value_counts(normalize=True) * 100

    # Plot the values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=normalized_counts.index, y=normalized_counts.values, palette='viridis')
    plt.title(f'{feature_meaning} distribution')
    plt.xlabel(feature_meaning)
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')

    # Display the plot in Streamlit
    st.pyplot(plt)
    
    
    
def display_feature_cleaning(df):
    st.header("Features Cleaning", divider=header_divider)

    st.subheader("1. Non-Contributory Features:")
    st.markdown(
        "Removed columns 'MMS,' 'Ernedc_g/km,' 'De,' and 'Vf' consisting exclusively of NaN values "
        "as they provide no meaningful information."
    )

    st.subheader("2. Features with Excessive Unique Values:")
    st.markdown(
        "Eliminated columns with an exceptionally high percentage of unique values to prevent noise "
        "and improve model generalization."
        "Exception applies for Features providing valuable informations for our study such as IT"
    )
    # Generate a DataFrame indicating unique values count
    plot_unique_values_count(df)

    

def display_observation_cleaning(df):
    st.header("Observations Cleaning", divider=header_divider)
    
    st.subheader("3. Features with Low Missing Values Percentage:")
    st.markdown(
        "Removed observations with missing values for 'Mp' as they present only 1%. This maintains dataset integrity and "
        "avoids potential inaccuracies from imputing difficult-to-estimate Manufacturer pool data."
    )
    st.subheader("4. Exclusion of Non-Polluting Cars:")
    st.markdown(
        "Excluded environmentally friendly cars ('ELECTRIC' and 'HYDROGEN') and vehicles with an "
        "'UNKNOWN' fuel type to focus on factors associated with CO² pollution."
    )

    st.subheader("5. Rare Categories in Fuel Type Distribution:")
    st.markdown(
        "Excluded 'NG' and 'NG-BIOMETHANE' categories with representation below 0.1% for a more meaningful "
        "exploration of the fuel type feature."
    )
    percentage_count_plot(df, 'Ft')
    
    
    
def display_missing_values_filling(df, img_path):
    st.subheader("Filling Missing Values")

    st.subheader("6. Electric Range and Electric Energy Consumption:")
    st.markdown(
        "Set missing values in 'Er' and 'z_Wh/km' columns to zero for accurate representation, "
        "considering these represent cars with no electric support."
    )

    st.subheader("7. Innovative Technologies and Emission Reduction:")
    st.markdown(
        "Set missing values in 'Erwltp_g/km' and 'IT' columns to zero,   "
        "as NaN values, simply represent the absence of these features for corresponding vehicles."
    )
    
    st.subheader("8. Tailoring Fuel Consumption Imputation by Fuel Type:")
    st.markdown(
        "To handle missing fuel consumption values, it's essential to consider variations across different fuel types."
        "Imputing with median rates within each fuel type group ensures accuracy and alignment with characteristic consumption patterns, enhancing dataset relevance."
    )
    #df_temp = df[df.Ft.isin(['NG-BIOMETHANE', 'ELECTRIC', 'HYDROGEN'])]
    boxplot_by_category(df, 'Ft', 'Fc') 
    img = Image.open(img_path)
    st.image(img, caption='Process of filling missing values in Fc column', use_column_width=True)
   
   
   
def regrouping_fuel_types(df, col_name):
    st.subheader("Regrouping Fuel types")
    # Visualize distribution of Fuel types 
    df_temp = df.drop(df.loc[df.Ft.isin(['ELECTRIC', 'UNKNOWN', 'HYDROGEN'])].index, axis=0)
    df_temp.Ft = df_temp.Ft.replace({'PETROL/ELECTRIC' : 'HYBRID', 'DIESEL/ELECTRIC' : 'HYBRID', 'E85' : 'OTHER', 'LPG' : 'OTHER'})
    df_temp.drop(df_temp.loc[df_temp.Ft.isin(['NG', 'NG-BIOMETHANE'])].index, inplace=True)
    display_normalized_value_counts(df_temp, col_name)
    st.markdown('''
        In order to balance the distribution of fuel types we opted for fuel types regrouping."
        - 'PETROL/ELECTRIC' and 'DIESEL/ELECTRIC' joined in a group labeled 'HYBRID'"
        - 'E85' and 'LPG' are moved to a group called 'OTHER'''
    )
    del df_temp
    
    return

#######################################################################################################################################################
#Exploratory Data Analysis
#######################################################################################################################################################
st.set_option('deprecation.showPyplotGlobalUse', False)
header_divider = 'green'
# Function to create a bar plot for car count by manufacturer
#@st.cache_data 
def plot_car_count(df):
    car_count_by_manufacturer = df['Mp'].value_counts()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=car_count_by_manufacturer.values, y=car_count_by_manufacturer.index, order=car_count_by_manufacturer.index, palette='viridis')
    plt.title('Car Count by Manufacturer', loc="left", pad=30, fontweight="bold", fontsize=20)
    plt.xlabel('Car Count')
    plt.ylabel('Manufacturer')
    sns.despine(top=True, right=True)
    st.pyplot()

# Function to create a bar plot for average CO2 emissions by manufacturer
#@st.cache_data 
def plot_average_emissions(df):
    average_emissions_by_manufacturer = df.groupby('Mp')['Ewltp_g/km'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=average_emissions_by_manufacturer.values, y=average_emissions_by_manufacturer.index, order=average_emissions_by_manufacturer.index, palette='viridis')
    plt.title('Average CO2 Emissions by Manufacturer', loc="left", pad=30, fontweight="bold", fontsize=20)
    plt.xlabel('Average CO2 Emissions (g/km)')
    plt.ylabel('Manufacturer')
    sns.despine(top=True, right=True)
    plt.xticks(rotation=45, ha='right')  # Schräge Beschriftung auf der x-Achse
    st.pyplot()
    
# Function to create a Violinplot for CO2 emissions by manufacturer without 'E'
#@st.cache_data 
def plot_manufacturer_violinplot(df):
    plt.figure(figsize=(14, 6))
    sns.violinplot(x='Mp', y='Ewltp_g/km', data=df, inner='quartile', palette='viridis')
    plt.title('CO2 Emissions by Manufacturer', loc="left", pad=30, fontweight="bold", fontsize=20)
    plt.xlabel('Manufacturer')
    plt.ylabel('CO2 Emissions (g/km)')
    sns.despine(top=True, right=True)
    plt.xticks(rotation=45, ha='right')  # Schräge Beschriftung auf der x-Achse
    st.pyplot()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to create a heatmap of the correlation matrix
@st.cache_data 
def plot_correlation_heatmap(df):
    st.header("Multivariate Analysis Insights", divider=header_divider)
    st.subheader("Correlation")
    st.write(
        """
        - Correlations to our target variable suggest potential predictive power
        - Should be cautious of high correlations between features to avoid multicollinearity.
        - The specific impact depends on the model and the context of the analysis.
        """
    )

    # Zeigen Sie das Bild in Streamlit an
    image_path = os.path.join('plots', 'correlation.png')
    st.image(image_path, caption='', use_column_width=True)
    st.subheader("Insights")
    with st.expander("**Power and Vehicle Characteristics**"):
    
        st.markdown(
            """
            - Strong positive correlations (0.78) between power and vehicle weight, width, and engine capacity.
            - Heavier, wider, and more powerful vehicles tend to have higher power.
            """
        )
        
    with st.expander("**Electric Consumption and CO2 Emissions**"):
        st.markdown(
            """
            - Strong negative correlation (-0.79) between electric consumption and CO₂ emissions.
            - Electric vehicles with lower CO2 emissions tend to have higher electric consumption.
            """
        )

    with st.expander("**Fuel Consumption and Electric Consumption:**"):
        st.markdown(
            """
            - Strong negative correlation (-0.77) between fuel consumption and electric consumption.
            - There is an inverse relationship between fuel consumption and electric consumption.
            """
        )

    with st.expander("**Electric Range and Emissions:**"):
        st.markdown(
            """
            - Negative correlation (-0.83) between electric range and both CO₂ emissions and emissions reduction.
            - Vehicles with higher emissions tend to have a lower electric range.
            """
        )

    with st.expander("**Fuel Consumption and Electric Range:**"):
        st.markdown(
            """
            - Strong negative correlation (-0.81) between fuel consumption and electric range.
            - Vehicles with lower fuel consumption tend to have a higher electric range.
            """
        )

# Function to create a boxenplot for numerical variables
@st.cache_data 
def plot_boxenplot_numerical(df):
    st.header("Univariate Analysis Insights", divider=header_divider)
    st.markdown(
        """ 
        - As first step, we conducted a univariate analysis, focusing on individual variables to understand their distribution, central tendencies, and variations. 
        - This approach provided insights into the structure and characteristics of each numerical variable in isolation.
        """
    )
    # Zeigen Sie das Bild in Streamlit an
    image_path = os.path.join('plots', 'boxenplot.png')
    st.image(image_path, caption='', use_column_width=True)
    
    with st.expander("Boxplot Overview"):
        st.markdown(
            """
            - As we can see, the variables in our dataset reveal diverse scales and ranges. 
            - Notable positive skewness in most variables suggest that the majority of vehicles tend to have lower values, but a subset of vehicles exhibits higher values, impacting the overall average. 
            """
        )

    with st.expander("Variable Patterns"):
        st.markdown(
            """
            - Several variables, including 'Ewltp_g/km', 'ec _cm3', 'ep_KW', 'z_Wh/km', and 'Erwltp_g/km)', share similar distribution patterns marked by positive skewness. 
            - This commonality suggests that a significant portion of vehicles tend to have lower emissions, smaller engine displacements, lower power, reduced energy consumption, and lower emission factors. 
            - Variables such as 'm_kg', 'W_mm', 'At1_mm', and 'Fc' also shows positive skewness but with distinct ranges and scales.
            """
        )

    with st.expander("Outliers"):
        st.markdown(
            """
            - The presence of outliers and extreme values in some variables poses potential challenges for modeling. 
            - Addressing these outliers with careful consideration is imperative to ensure model robustness. 
            - Additionally, acknowledging the varying scales of the variables is vital when selecting algorithms to guarantee their adaptability to the diverse magnitudes present in the dataset. 
            """
        )

from PIL import Image
# Function to plot Violinplot for 'Fm' separated by 'Fuel Mode' ('Fm') without 'E'
#@st.cache_data 
def plot_fm_violin(df):
    st.subheader("CO2 Emission by Fuel Mode")
    st.image("plots\\violin_fm.png", use_column_width=True)
    st.markdown(
        """
        - 'B' and 'F' seem to show specific patterns, demonstrating broader shapes, indicating higher variance
        - 'H' and 'M' exhibit broader variance
        - 'P' consistently displays low emissions."""
    )

# Function to convert absolute values to percentage
#@st.cache_data 
def absolute_to_percentage(absolute_values):
    total = sum(absolute_values)
    return [value / total * 100 for value in absolute_values]

# Function to plot separate Barplots for 'Fm' and 'Ft' in percentage
#@st.cache_data 
def plot_percentage_ft(df):
    # Sort 'Fm' and 'Ft' counts
    fm_counts = df['Fm'].value_counts().sort_values(ascending=False)
    desired_categories_order = ['OTHER', 'DIESEL', 'PETROL', 'HYBRID']
    ft_counts = df['Ft'].value_counts()[desired_categories_order].sort_values(ascending=False)

    # Set style to 'white' to remove grid lines
    sns.set_style("white")

    # Barplot for 'Ft' in percentage
    plt.figure(figsize=(12, 6))
    ft_percentage = absolute_to_percentage(ft_counts)
    sns.barplot(x=ft_counts.index, y=ft_percentage, palette='viridis', order=ft_counts.index)
    plt.title('Fuel Type Distribution', loc="left", pad=30, fontweight="bold", fontsize=20)
    plt.xlabel('Fuel Type')
    plt.ylabel('Percentage (%)')
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Display percentages in the Barplot
    for i, value in enumerate(ft_counts):
        plt.text(i, ft_percentage[i] + 0.5, f'{ft_percentage[i]:.1f}%', ha='center')

    st.pyplot()  
    
# Function to plot Violinplot for 'Ewltp (g/km)' separated by 'Fuel Type'
#st.cache_data
#@st.cache_data  
def plot_ft_violin(df):
    st.subheader("CO2 Emission by Fuel Type")
    st.image("plots\\violin_ft.png", use_column_width=True)
    st.markdown(
    """
    - Slimmer violin shapes, as observed in 'DIESEL' and 'PETROL', suggest relatively lower variance, while broader shapes in 'OTHER' and 'HYBRID' indicate higher variability.
    - DIESEL' exhibits low standard deviation and variance, signifying a relatively consistent emission pattern. The median is at 131.00, reflecting a central tendency.
    - 'PETROL' also displays low standard deviation and variance, along with a sharper histogram peak at a median of 126.73.
    - 'OTHER' and 'HYBRID' present higher variances, implying broader distributions. 'HYBRID' stands out with the lowest median emissions at 37.39.
    - In the Histogram, 'OTHER' consistently maintains a more even and wider spread across emissions. While it does not necessarily indicate lower emissions on average, it suggests a broader and more diverse distribution of CO2 emissions within the 'OTHER' fuel type.

    """
    )

#@st.cache_data 
def count_CO2_category(df):
    st.header("Inbalance of CO2 emission classes", divider=header_divider)
    st.subheader("Feature Engineering")
    st.markdown(
        """
        - In pre-processing of ML classification we transformed the target variable Ewltp (g/km) into specific CO2 emission classes which are widely accepted metrics in the automotive sector
        - Optimizes the model performance and enhance the effectiveness of the training process
    
        """
    )
    
    st.image("plots\\Co2_classes.jpg", use_column_width=False)
    st.subheader("Distribution of CO2 emission classes")
    st.markdown(
        """
        - Balanced class distribution is crucial to ensure that the machine learning model is trained equally on all target classes, preventing biased predictions in favor of the more frequent class.
        - Examined the class distribution to verify and address any imbalances, ensuring a fair representation of all emission classes during the model training process.
        """
    )
    st.image("plots\\count_co2_cat.png", use_column_width=True)
    st.subheader("Insights")
    st.markdown( 
    """
    - As evident from the bar plot, some classes are more prevalent, while others are barely represented.
    - Imbalance has to be managed through oversampling or undersampling techniques to ensure fair representation of all emission classes.

    """
    )

#@st.cache_data 
def main(df):
    
    st.header("Exploratory Data Analysis", divider=header_divider)

    content = """
    - Understand the dataset on vehicle CO2 emissions comprehensively.
    - Identify hidden patterns, significant trends and gain understanding of relationships between different variables
    - Show specific challenges and questions for the upcoming modeling phase.


    """

    st.markdown(content)
  


#######################################################################################################################################################
#Regression - Machine Learning
#######################################################################################################################################################

def get_evaluation_table(model,feat_train, feat_test,target_train, target_test):

  # Evalutaion on Training Set
  y_train_pred = model.predict(feat_train)
  train_rmse = mean_squared_error(target_train, y_train_pred, squared=False)
  train_mse = mean_squared_error(target_train, y_train_pred)
  train_mae = mean_absolute_error(target_train, y_train_pred)
  train_r2 = r2_score(target_train, y_train_pred)

  # Evalutaion on Test Set
  y_test_pred = model.predict(feat_test)
  test_rmse = mean_squared_error(target_test, y_test_pred, squared=False)
  test_mse = mean_squared_error(target_test, y_test_pred)
  test_mae = mean_absolute_error(target_test, y_test_pred)
  test_r2 = r2_score(target_test, y_test_pred)

  # Create table
  results = pd.DataFrame(
    index = ['MSE', 'RMSE', 'MAE', 'R2 Score'],
    data={'Training Set': [train_mse, train_rmse, train_mae, train_r2],
          'Test Set': [test_mse,test_rmse, test_mae, test_r2],
          'Delta': [test_mse-train_mse, test_rmse-train_rmse, test_mae-train_mae, test_r2-train_r2]})
  
  return results

def regression_ml_intro():

    content="""
    - **target**: CO\u00b2 emissions meassured in g/km based on the Worldwide Harmonized Light Vehicles Test Procedure
    - **data**: preprocessed data about car properties (not standardized for ML algorithms). We dropped the features Fuel Consumption and Electric Range because they are highly correlated with CO\u00b2 emissions."""

    st.markdown(content)

def regression_ml_approach():

    content = """
    - We selected 3 ML algorithms and 1 DL model to predict CO2 emissions
    - We **compared** the performance of these **baseline models** and selected the best one
    - We **optimized the best model** using hyperparameter tuning
    - Our **final model** is a used for the **model interpretation** in order to understand how different features influence the amount of CO2 emissions produced
    """

    st.markdown(content)


#######################################################################################################################################################
#Regression - Deep Learning
#######################################################################################################################################################
def regression_dl_summary_comments():

    content = """
    - Standardized data with MinMaxScaler, because DL models are sensitive to the scale of the input data
    - **Functional construction** approach with 'Keras' library. 
    - 2 hidden layers with 64 neurons each and **'relu'** activation function
    - Output layer with 1 neuron and **'linear'** activation function
    """

    st.markdown(content)


def dl_plot_loss(history):
    plt.figure(figsize=(10,5))

    # Get the colors from 'viridis'
    viridis = sns.color_palette("viridis", 3)

    # Use the first color for 'loss' and the second color for 'val_loss'
    sns.lineplot(data=history.history['loss'], label='loss', color=viridis[0])
    sns.lineplot(data=history.history['val_loss'], label='val_loss', color=viridis[2])

    plt.ylim([0, 12])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error [Ewltp_g/km]')
    plt.legend()
    plt.show()

    # Show the plot on streamlit
    st.pyplot(plt)

def dl_show_model_summary(model):
    import sys
    import io

    # Create a StringIO object
    stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call model.summary()
    model.summary()

    # Get the string from StringIO object
    summary_string = sys.stdout.getvalue()

    # Reset the standard output
    sys.stdout = stdout

    # Use st.text() to display the model summary
    st.text(summary_string)

#######################################################################################################################################################
#Regression - Model Selection and Hyperparameter Tuning
#######################################################################################################################################################



#######################################################################################################################################################
#Classification - Machine Learning
#######################################################################################################################################################

def read_comparison_csv():
    # Path
    csv_path = 'plots/comparison.csv'

    # Try to read CSV
    try:
        # Use seperator ';'
        df = pd.read_csv(csv_path, sep=';')
        st.success("")
        st.dataframe(df)  # show table in streamlit
    except FileNotFoundError:
        st.error(f"File '{csv_path}' could not be found.")
    except pd.errors.EmptyDataError:
        st.warning("CSV is empfty.")
    except pd.errors.ParserError:
        st.error("Failure loading CSV")



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_model_evaluation_results():
    # Model evaluation results
    results_data = {
        'Model': ['RandomForest', 'XGBoost (Optimized)', 'DecisionTree', 'XGBoost', 'DecisionTree (Optimized)', 'DNN-Model'],
        'Accuracy': [0.97, 0.97, 0.97, 0.97, 0.97, 0.97],
        'Mean ROC AUC (OvR)': [0.9992, 0.9464, 0.9993, 0.9991, 0.9992, 0.9992],
        'Mean F1-Score (Macro)': [0.9712, 0.8852, 0.9712, 0.9691, 0.9711, 0.9696]
    }

    results_df = pd.DataFrame(results_data)

    # Display the model evaluation results as a table
    st.markdown("### Model Evaluation Results")
    st.table(results_df)

    # Visualize the results
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Accuracy
    sns.barplot(x='Model', y='Accuracy', data=results_df, ax=axes[0], palette='viridis')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0.9, 1)

    # Plot Mean ROC AUC
    sns.barplot(x='Model', y='Mean ROC AUC (OvR)', data=results_df, ax=axes[1], palette='viridis')
    axes[1].set_ylabel('Mean ROC AUC (OvR)')
    axes[1].set_ylim(0.9, 1)

    # Plot Mean F1-Score
    sns.barplot(x='Model', y='Mean F1-Score (Macro)', data=results_df, ax=axes[2], palette='viridis')
    axes[2].set_ylabel('Mean F1-Score (Macro)')
    axes[2].set_ylim(0.9, 1)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()

    # Display the visualizations
    st.pyplot(fig)


def confusion_matrix_heatmap():
    # Provided confusion matrix data
    conf_matrix_data = [
        [206081, 10, 0, 0, 0, 0, 0],
        [145, 200830, 5117, 0, 0, 0, 0],
        [0, 8678, 190290, 7124, 0, 0, 0],
        [4, 0, 4515, 197016, 4556, 0, 0],
        [0, 0, 0, 7217, 198112, 763, 0],
        [0, 0, 0, 0, 145, 203904, 2043],
        [0, 0, 0, 0, 0, 1163, 204929]
    ]

    # Convert the list to a NumPy array
    conf_matrix_array = np.array(conf_matrix_data)

    # Create a DataFrame from the array
    conf_matrix_df = pd.DataFrame(conf_matrix_array, index=range(1, 8), columns=range(1, 8))

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create a heatmap using seaborn
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'Count'})
    
    # Set labels and title
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("")

    # Show the plot
    st.pyplot()
    
def classification_metrics_table():
    # Provided classification metrics data
    metrics_data = {
        'Class': [1, 2, 3, 4, 5, 6, 7],
        'Precision': [1.00, 0.96, 0.95, 0.93, 0.98, 0.99, 0.99],
        'Recall': [1.00, 0.97, 0.92, 0.96, 0.96, 0.99, 0.99],
        'F1-Score': [1.00, 0.97, 0.94, 0.94, 0.97, 0.99, 0.99]
    }

    # Create a DataFrame from the data
    metrics_df = pd.DataFrame(metrics_data)
    
    # Set 'Class' column as the index
    metrics_df.set_index('Class', inplace=True)

    # Display the table without the index column
    st.table(metrics_df)
    
# Function to create a boxenplot for numerical variables
#@st.cache_data 
def results_ml_clf(df):
    st.header("Classification task", divider=header_divider)
    st.markdown(
        """
        - **target:** CO2 emission classes aligning with the standards of the automotive sector
        - **data:** Preprocessed data about car properties. To reduce the impact of outliers, extreme values and different scales and ranges, we also standardised the data. To tackle imbalance we used Oversampling
        """
    )
    
    st.header("Approach", divider=header_divider)
    st.markdown(
        """
        - We selected 3 ML algorithms and 1 DL model to predict CO2 emission classes
        - We compared the performance of these baseline models and selected the best one
        - We performed hyperparameter tuning on several models
        - Our final model is a used for the model interpretation in order to understand how different features influence the predcited CO2 emission class
        """
    )
    
    st.header("Why we have chosen these models?", divider=header_divider)
    with st.expander("**Ensemble Learning**"):
        st.markdown(
            """
            - All three algorithms (Random Forest, Decision Tree, XGBoost) leverage Ensemble Learning for robust model building through the combination of multiple weak models.
            """
        )
    with st.expander("**Robustness to Outliers**"):
        st.markdown(
            """
            - Random Forest and XGBoost exhibit robustness to outliers due to their aggregated decision-making process. Decision Trees are less robust, but pruning techniques can mitigate overfitting.
            """
        )
    with st.expander("**Handling different Variable Ranges**"):
        st.markdown(
            """
            - Random Forest and XGBoost can effectively handle variables with different ranges, while Decision Trees are less independent of scaling. 
            """
        )
    with st.expander("**Efficieny and Spped**"):
        st.markdown(
            """
            - All three algorithms can handle a large number of variables, with Random Forest and XGBoost offering feature importance insights.
            - XGBoost is renowned for its efficiency and speed, making it ideal for large datasets. Random Forest is also efficient, while Decision Trees may take longer, especially with complex models.
            """
    )
    
    st.header("Results", divider=header_divider)
    st.subheader("Overall Metrics")
    read_comparison_csv()
    with st.expander("**Accuracy**"):
        st.markdown(
            """
            - The proportion of correctly classified instances. In this case, 97% of predictions of all models have been correct.
            """
        )
    with st.expander("**Mean ROC AUC (OvR)**"):
        st.markdown(
            """
            - ROC AUC (Area Under the Receiver Operating Characteristic curve) measures the model's ability to distinguish between classes. Higher values (close to 1) indicate better performance.
            - RandomForest and DecisionTree models have outstanding discrimination ability with values close to 1.
            - XGBoost models, while slightly lower, still exhibit strong discrimination.
            """
        )
    with st.expander("**Mean F1-Score (Macro)**"):
        st.markdown(
            """
            - The harmonic mean of precision and recall, providing a balance between the two. It is especially useful in imbalanced datasets. Higher values (closer to 1) indicate better precision and recall balance.
            - All models achieve high F1-Scores, indicating a good balance between precision and recall.
        """
        )

    with st.expander("**Model Comparison:**"):
        st.markdown(
            """
            - Decision tree-based models perform similarly in terms of accuracy, ROC AUC, and F1-Score.
            - XGBoost models have slightly lower ROC AUC and F1-Score but remain at a high level.
            """
        )

    st.subheader("Decision Tree: Confusion Matrix and Predicted True Values")
    # Display the classification metrics table
    classification_metrics_table()
        # Display the heatmap
   
    st.markdown(
        """      
        For the Decision Tree model, the following observations regarding Precision, Recall, and F1-Score can be summarized:
        """
    )
    with st.expander("Well-recognized Classes:"):
        st.markdown(
            """
            - **Class 1**: Outstanding performance with Precision (positive predictive value), Recall (True positive rate), and F1-Score (harmonic mean of precision and recall) all at 1.0000, indicating precise identification of this class.
            - **Classes 6 and 7:** Very good recognition with Precision, Recall, and F1-Score all at 0.9900.
            """
        )
    with st.expander("Not as Well-recognized Classes:"):
        st.markdown(
            """
            - **Class 3:** Although Precision and F1-Score are relatively high at 0.9500 and 0.9400, respectively, the Recall is slightly lower at 0.9200, suggesting some instances of this class might not have been captured.
            - **Class 4:** Similar to Class 3, it exhibits a slightly lower Recall (0.9600) compared to Precision (0.9300) and F1-Score (0.9400), indicating potential missed instances.
            - **Class 2:** Good performance but with a slightly lower Recall of 0.9700 compared to Precision (0.9600) and F1-Score (0.9700).
            """
        )
    confusion_matrix_heatmap()

#######################################################################################################################################################
#Classification - Deep Learning
#######################################################################################################################################################

@st.cache_data    
def classification_intro():

    content="""
    - **target**: CO\u00b2 Class (consists of six classes ranging from 0 to 6)
    - **data**: preprocessed data about car properties """

    st.markdown(content)
    
@st.cache_data
def classification_dl_approach():

    content = """
    - Start with an **arbitrary model** denoted as **First Model** 
    - Conduct a **Random Search** to identify best Hyperparametrs for our configuration
    - Optimize the model by incorporating **callbacks**
    - Enhance the model's performance by **oversampling** the dataset
    """

    st.markdown(content)         
@st.cache_data
def load_saved_model_dnn(model_path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    
    # Define a custom optimizer with the desired parameters
    custom_optimizer = Adam(learning_rate=0.01)

    # Load the model using custom_objects to specify the optimizer
    loaded_model = load_model(model_path, custom_objects={'Adam': custom_optimizer})

    # Compile the loaded model with the same optimizer
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])

    # Capture the summary text in a string
    import io
    summary_buf = io.StringIO()
    loaded_model.summary(print_fn=lambda x: summary_buf.write(x + '\n'))
    summary_text = summary_buf.getvalue()

    # Display the model summary within the Streamlit app
    st.text(summary_text)
    return loaded_model



# Display image    
@st.cache_data
def display_img(img_path):
    img = Image.open(img_path)
    st.image(img)
        
    
# Plot Confusion Matrix
@st.cache_data
def display_confusion_matrix(cm):
  
  # classes = range(0, 6) #delete 
  fig = plt.figure()
    
  plt.imshow(cm, interpolation='nearest', cmap='Blues')
  #plt.title("Confusion Matrix")
  plt.colorbar()

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j],
      horizontalalignment="center",
      color="white" if cm[i, j] > ( cm.max() / 2) else "black")

  plt.ylabel('True labels')
  plt.xlabel('Predictions')
  st.pyplot(fig)
  
  
  
def load_cm_cr_from_pickle_data(data_filepath):
    '''Load data including 
    -training_history
    -confusion Matrix
    -classification report
    '''
    with open(data_filepath, 'rb') as f:
      dnn_data = pickle.load(f) 
      
    th  = dnn_data['training_history']
    cm = dnn_data['conf_matrix']
    cr = dnn_data['classification_report']
    return th, cm, cr
  
  

def display_classification_report(model_name, classification_report_str):
    # Parse the classification report string
    report_lines = classification_report_str.split('\n')
    data = [line.split()[:4] for line in report_lines[2:-5]]  # Exclude the support values
    accuracy_line = [line.split() for line in report_lines[-2:-1]][0]  # Get the accuracy line
    accuracy_line[1] = 'Accuracy average'
    # Add accuracy information to the data list
    data.append(accuracy_line[1:])  # Exclude the "accuracy" label

    # Create a table using plotly
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'table'}]]
    )
    header=dict(values=['Class', 'Precision', 'Recall', 'F1-Score'])
    cells=dict(values=list(zip(*data)))
    # Add metrics table to the plot
    fig.add_trace(go.Table(
        header=header,
        cells=cells,
        columnwidth=[.25, .2, .2, .2],
    ), row=1, col=1)

    # Update layout for better display
    fig.update_layout(
        height=450,
        width=700,
        showlegend=False,
        #title_text="Classification Report",
    )
    #convert it to DataFrame
    cells=dict(values=list(zip(*data[:-1])))
    
    # Convert it to DataFrame
    cr_df = pd.DataFrame(cells['values']).transpose()
    cr_df['Model'] = model_name
    cols = header['values'] + ['Model']  # Corrected concatenation
    cr_df.columns = cols
    cr_df['Class'] = pd.to_numeric(cr_df['Class'], errors='coerce').astype('Int64')
    # Show the Plotly figure directly using st.plotly_chart
    st.plotly_chart(fig)
    
    return cr_df


def plot_acc_loss(model_history):
    loss_valid = model_history['val_loss']
    acc_valid = model_history['val_accuracy']

    loss_train = model_history['loss']
    acc_train = model_history['accuracy']

    plt.figure(figsize=(12, 6))
    viridis = sns.color_palette("viridis", 3)
    plt.subplot(121)
    plt.plot(np.arange(len(loss_valid)), loss_valid, label='Validation', color=viridis[0])
    plt.plot(np.arange(len(loss_train)), loss_train, label='Training', color=viridis[2])
    plt.legend()
    plt.title('loss')
    plt.subplot(122)
    plt.plot(np.arange(len(acc_valid)), acc_valid, label='Validation', color=viridis[0])
    plt.plot(np.arange(len(acc_train)), acc_train, label='Training', color=viridis[2])
    plt.title('accuracy')
    plt.legend()
    st.pyplot(plt)

    return
  

def model_full_report(model_name, data_folder, trained_model, model_summary_filename, tr_cm_cr_pickle_filename, code):
    '''
    Displays a full report about a DNN model including 
    - Model Summary
    - Training History (Plots of both loss & Accuracy)
    - Confusion Matrix 
    - Classification report 
    
    Inputs
        - data_folder: Folder including data 
        - trained_model : Saved pretrained model filename 
        - model_summary_filename : Model summary image filename
        - tr_cm_cr_pickle_filename : Pickle Data filename dict{}.keys = ['training_history', 'conf_matrix', 'classification_report']
    '''
    # Path of saved model file
    model_path = data_folder + '\\' + trained_model

    # Load pretrained model (Best model)
    #dnn = load_saved_model_dnn(model_path)
    
    #
    if code != None:
        with st.expander('Show/Hide Code'):
            st.code(code, language='python')
    
    #Display model summary
    st.subheader('Model Summary')
    final_model_summary = data_folder + '\\' + model_summary_filename
    display_img(final_model_summary)       
    
    # Load confusion_matrix andclassification_report
    pickle_data_filename = data_folder + '\\' + tr_cm_cr_pickle_filename
    training_history, confusion_matrix, classification_report = load_cm_cr_from_pickle_data(pickle_data_filename)  
    
    # Plot Loss & Accuracy Training vs Validation
    st.subheader('Loss & Accuracy Evolution')
    plot_acc_loss(training_history)
    
    #Display confusion_matrix
    st.subheader('Confusion Matrix')
    display_confusion_matrix(confusion_matrix)
    
    # Display_classification_report
    st.subheader('Classification Report')
    class_report_df = display_classification_report(model_name, classification_report)
    
    
    return confusion_matrix, class_report_df, training_history, 



def compare_history_plot(hist_dict):
    plt.figure(figsize=(12, 6))
    viridis = sns.color_palette("viridis", 3)
    n=0
    for model_names, model_history in hist_dict.items():
        loss_valid = model_history['val_loss']
        acc_valid = model_history['val_accuracy']
        loss_train = model_history['loss']
        acc_train = model_history['accuracy']
            
        plt.subplot(121)
        #plt.plot(np.arange(len(loss_valid)), loss_valid, label=f'Validation {model_names}')
        plt.plot(np.arange(len(acc_train)), loss_train, label=f'Training {model_names}', color=viridis[n])

        plt.subplot(122)
        #plt.plot(np.arange(len(acc_valid)), acc_valid, label=f'Validation {model_names}')
        plt.plot(np.arange(len(acc_train)), acc_train, label=f'Training {model_names}', color=viridis[n])
        n += 1
    plt.subplot(121)
    plt.legend()
    plt.title('Loss')

    plt.subplot(122)
    plt.legend()
    plt.title('Accuracy')

    st.pyplot(plt)
    
    
def compare_prec_recall_f1score(classif_report_dataframes, metric):
   
    df = pd.melt(classif_report_dataframes[0], id_vars=['Class', 'Model'], value_vars=['Precision', 'Recall', 'F1-Score'],
                        var_name='Metric', value_name='Score')
    for df_cr in classif_report_dataframes[1:]:
        df_melted = pd.melt(df_cr, id_vars=['Class', 'Model'], value_vars=['Precision', 'Recall', 'F1-Score'],
                        var_name='Metric', value_name='Score')
        df = pd.concat([df, df_melted])
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    
    df = df[df.Metric==metric]
    #st.table(df)
    # Plot using Seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Class', y='Score', hue='Model', data=df, palette='viridis')
    # Add annotations to each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

    #plt.title('Classification Metrics by Class')
    plt.ylim(.8, 1.2)
    plt.yticks(np.arange(.7, 1.01, 0.1))
    st.pyplot(plt)
    return 

def get_class_predictions(cm, true_class):
    predictions = cm[true_class, :]

    normalized_predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
    normalized_predictions = np.asarray(normalized_predictions, dtype=np.float64)
    
    # Plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(range(0, len(normalized_predictions))), y=normalized_predictions, palette='viridis')
    plt.ylim(0, 1.2)
    plt.yticks(np.arange(0, 1.1, 0.1))
    st.pyplot(plt)

    return


#######################################################################################################################################################
#Regression - Machine Learning
#######################################################################################################################################################




#######################################################################################################################################################
#Regression - Deep Learning
#######################################################################################################################################################

#######################################################################################################################################################
#Regression - Model Selection
#######################################################################################################################################################

def plot_ewltp_boxplot(data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 2))
    ax = sns.boxplot(x=data['Ewltp_g/km'], orient='h', palette='viridis')
    ax.set(xlabel=None)  # Hide x label
    plt.title('Boxplot of CO\u00B2 in g/km')
    plt.show()
    st.pyplot(plt)
    sns.set_style("white") # reset to default

#######################################################################################################################################################
#Interpretation
#######################################################################################################################################################
# Load Model
@st.cache_data
def load_regression_ml_model(model_path):
    return joblib.load(model_path)

# Plot Feature Importance with cumulative importance line
def plot_feature_importance(model, X_train):
    # Create dictionary to get top 10 features
    feature_importances = dict(zip(X_train.columns, model.feature_importances_))
    feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_10_features = feature_importances[:10]

    # Extract feature names and importance from the dictionary of selected and sorted features 
    feature_names, importance_values = zip(*top_10_features)

    # Calculate cumulative feature importance
    cumulative_importance = np.cumsum(importance_values)

    # Set style
    sns.set_style("white")

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot feature importance 
    sns.barplot(x=list(feature_names), y=list(importance_values), palette='viridis', ax=ax1)
    ax1.set_xlabel('Feature Names')
    ax1.set_ylabel('Feature Importance')
    ax1.tick_params(axis='y')
    ax1.set_title('Top 10 Feature Importance')
    ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

    # Add the cumulative importance
    ax2 = ax1.twinx()
    sns.lineplot(x=list(feature_names), y=cumulative_importance, marker='o', sort=False, color='red', ax=ax2)
    ax2.set_ylabel('Cumulative Importance', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Show legend
    ax2.legend(["Cumulative Importance"])
    plt.tight_layout()
    plt.show()

    # Show the plot on Streamlit
    st.pyplot(plt)

# Plot Scatterplots between most important numerial features and target
def plot_scatterplots_interpretation(X_train, y_pred_train):
    # Create scatterplot for ep_KW vs. Ewltp_g/km
    plt.figure(figsize=(10, 6))  # Set figure size
    sns.scatterplot(x=X_train['ep_KW'], y=y_pred_train, hue=X_train['Ft_HYBRID'],palette='viridis')
    plt.xlabel('ep_KW', fontsize=14)
    plt.ylabel('Predicted CO2 emission', fontsize=14)
    plt.show()
    st.pyplot(plt)

    # Create a figure and two subplots for scatterplots of m_kg vs. Ewltp_g/km and ep_KW vs. Ewltp_g/km
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))  # 1 row, 2 columns
    sns.scatterplot(x=X_train['ep_KW'], y=y_pred_train, 
                    hue=X_train['Ft_HYBRID'], palette='viridis', ax=axes[0])
    axes[0].set_xlabel('ep_KW', fontsize=14)
    axes[0].set_ylabel('Predicted CO2 emission', fontsize=14)
    sns.scatterplot(x=X_train['m_kg'], y=y_pred_train, 
                    hue=X_train['Ft_HYBRID'], palette='viridis', ax=axes[1])
    axes[1].set_xlabel('m_kg', fontsize=14)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# Plot
def plot_boxplots_interpretation(X_train, y_pred_train):
    sns.set_style("white")
    plt.figure(figsize=(14, 6))  
    # Boxplot for Fm_P on the left side
    plt.subplot(1, 2, 1)
    sns.boxplot(x=X_train['Fm_P'], y=y_pred_train, palette='viridis')
    plt.xlabel('Fuel Mode Petrol ', fontsize=14)
    plt.ylabel('Predicted CO2 emission', fontsize=14)

    # Boxplot for ep_KW on the right side
    plt.subplot(1, 2, 2)
    sns.boxplot(x=X_train['Ft_HYBRID'], y=y_pred_train, palette='viridis')
    plt.xlabel('Fuel Type Hybrid ', fontsize=14)
    plt.ylabel('Predicted CO2 emission', fontsize=14)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plots
    plt.show()
    st.pyplot(plt)

#######################################################################################################################################################
#Conclusion
#######################################################################################################################################################
    
