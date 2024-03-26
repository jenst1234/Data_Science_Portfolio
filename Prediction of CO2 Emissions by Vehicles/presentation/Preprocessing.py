import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import streamlit as st 

@st.cache_data
def apply_preprocessing(df):
    '''
    This function aims to clean the dataframe and transform some features
    '''
    # Remove columns with 100% missing values
    df.dropna(axis = 1, how = 'all', inplace = True)

    # Dropping columns with too many categories
    df = df[['Mp','Cr','m_kg','Ewltp_g/km','W_mm','At1_mm','Ft','Fm','ec_cm3','ep_KW','z_Wh/km','IT','Erwltp_g/km','Fc','Er_km']]

    # Dropping lines if missing values < 1%
    df = df.drop(df.loc[df.Mp.isna()].index, axis=0)

    # Dropping non-polluting cars
    df = df.drop(df.loc[df.Ft.isin(['ELECTRIC', 'UNKNOWN', 'HYDROGEN'])].index, axis=0)

    # Dropping very low observations
    df.drop(df.loc[df.Ft.isin(['NG', 'NG-BIOMETHANE'])].index, inplace=True)

    # Filling missing values for z_Wh/km by 0 if car is not Hybrid
    df.loc[((df['z_Wh/km'].isna()) & (df.Ft.isin(['PETROL', 'DIESEL', 'NG-BIOMETHANE', 'NG', 'E85', 'LPG']))),'z_Wh/km'] = 0

    # Setting It to 0/1 depending if the value has or not an IT tech
    df.IT = np.where(df.IT.isna(), 0, 1)

    # Filling Electric range (km) is 0 for non_hybrid cars
    df.Er_km = np.where(~df.Ft.isin(['PETROL/ELECTRIC', 'DIESEL/ELECTRIC']), 0, df.Er_km)

    # Filling missing values for Erwltp_g/km with 0
    df['Erwltp_g/km'] = df['Erwltp_g/km'].fillna(0)

    #Fill missing values for Fuel consumption by the median of the corresponding class in term of Fuel Type
    fc_med_per_ft_category = df.groupby(['Ft'])['Fc'].median()

    # Create a new column with median values based on 'Ft'
    df['Fc_median'] = df['Ft'].map(fc_med_per_ft_category)

    # Fill missing values in 'Fc' based on the corresponding 'Fc_median' values
    df['Fc'].fillna(df['Fc_median'], inplace=True)

    # Drop the temporary 'Fc_median' column if you no longer need it
    df.drop('Fc_median', axis=1, inplace=True)

    # Grouping cars into new categories Hybrid for 'PETROL/ELECTRIC' and 'DIESEL/ELECTRIC', others for 'E85', 'LPG' to deal with very low observations percentages
    df.Ft = df.Ft.replace({'PETROL/ELECTRIC' : 'HYBRID', 'DIESEL/ELECTRIC' : 'HYBRID', 'E85' : 'OTHER', 'LPG' : 'OTHER'})

    # Encoding 'Mp', 'Ft', 'Fm', 'Cr'
    df = pd.get_dummies(df, columns=['Mp', 'Ft', 'Fm'])
    df.Cr = LabelEncoder().fit_transform(df.Cr)
    
    return df