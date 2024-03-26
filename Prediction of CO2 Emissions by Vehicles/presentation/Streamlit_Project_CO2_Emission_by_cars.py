import streamlit as st 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import pprint
from Streamlit_functions_library import *
from sklearn.model_selection import train_test_split
from Preprocessing import apply_preprocessing
from scipy.stats import uniform, randint

#Streamlit title 
st.title('Project Defense: CO\u00B2 Emission by cars')

#Table of Content 
st.sidebar.title('Tabe of contents')
pages = ['Introduction', 'Overview', 'Preprocessing', 'EDA', 'Regression (ML)', 'Regression (DL)', 'Regression (Model Selection)','Classification (ML)', 'Classification (DL)', 'Interpretation', 'Conclusion']
page = st.sidebar.radio('Go to', pages)

# Data Loading
df = load_dataframe('data_2021_CO2_Project_Fr.csv')
df_preprocessed = load_dataframe('data_2021_CO2_Project_Fr_preprocessed.csv')
df_original = load_dataframe('data_2021_CO2_Project_Fr_not_preproc.csv')# data_2021_CO2_Project_Fr.csv #data_2021_CO2_Project_Fr_preprocessed.csv
df_fr_renamed = filter_country_rename_features(df_original)

# Train Test Split
data = df_preprocessed.drop(['Ewltp_g/km','Fc','Er_km'],axis=1)
target = df_preprocessed['Ewltp_g/km']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2,random_state=123)

#######################################################################################################################################################
#Introduction
#######################################################################################################################################################

if page == 'Introduction':
    intro()
    
#######################################################################################################################################################
#Overview
#######################################################################################################################################################

if page == 'Overview':
      
    st.write('**Data Introduction**')
    
    
    #General description 
    display_dataset_info()
    
    # DataFrame overview
    display_dataframe_info(df_original)
    # Distribution of Countries
    img_path = '.\streamlit_saved_files\countries_distribution.png'
    countries_pie_chart(img_path)
        
    # Explain Features meaning with a table 
    feature_meaning()
    
    # Distribution of Manufacturers 
    img_path = '.\streamlit_saved_files\mp_distribution.png'
    manufacturers_distribution_pie_chart(img_path)
    
    
#######################################################################################################################################################
#Preprocessing
#######################################################################################################################################################

if page == pages[2]:

    # Cleaning Dataset 
    st.title("Dataset Cleaning Summary")
    
    # Generate a DataFrame indicating missing values
    plot_missing_values(df_original)
    
    # Feature cleaning
    display_feature_cleaning(df_fr_renamed)
    
    # Observation cleaning
    display_observation_cleaning(df_fr_renamed)
    img_path = '.\streamlit_saved_files\Filling_miss_val_Fc.PNG'
    
    # Display Missing Values
    display_missing_values_filling(df_fr_renamed, img_path)

    #st.dataframe(df_temp)
    regrouping_fuel_types(df_fr_renamed, 'Ft')

    # Apply preprocessing to clean the DataFrame
    df_fr_renamed = apply_preprocessing(df_fr_renamed)
    
    # Checking New Dataframe
    display_dataframe_info(df_fr_renamed)

#######################################################################################################################################################
# Exploratory Data Analysis
#######################################################################################################################################################
if page == 'EDA':
    main(df)
    # Call the function to generate the boxenplot
    plot_boxenplot_numerical(df)
    # Call the functions to generate the plots
    #plot_car_count(df_eda)
    #plot_average_emissions(df_eda)
    # Call the function to generate the Violinplot
    #plot_manufacturer_violinplot(df_eda )
    # Call the function to generate the heatmap
    plot_correlation_heatmap(df)
    # Call the function to plot Violinplot for 'Fm' separated by 'Fuel Mode' ('Fm') without 'E'
    #plot_fm_violin(df)
    # Call the function to plot Violinplot for 'Ewltp (g/km)' separated by 'Fuel Type'
    #plot_ft_violin(df_eda)
    count_CO2_category(df)

    # Call fm_percentage
    #plot_percentage_fm(df_eda)
   
    #Call ft_percentage    
    #plot_percentage_ft(df_eda)

#######################################################################################################################################################
#Regression - Machine Learning
#######################################################################################################################################################

if page == 'Regression (ML)':
    
    # Load Models with joblib
    dtr_model = joblib.load('.\\streamlit_saved_files\\dt_model_wo_fc.joblib')
    rfr_model = joblib.load('.\\streamlit_saved_files\\rf_model_wo_fc.joblib')
    xgbr_model = joblib.load('.\\streamlit_saved_files\\xgb_model_wo_fc.joblib')

    st.header('Regression task', divider=header_divider)
    regression_ml_intro()

    st.header('Our Approach', divider=header_divider)
    # Introduction
    regression_ml_approach()

    # Evaluation Metrics
    # st.markdown("""## Evalutation Metrics""")
    # st.markdown("""Our main evaluation metric is the $R^2$ score as it meassures the goodness of fit of the models.""")
    # st.latex(r'''R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}''')
    # st.markdown("""To better differentiate the results, we evaluated model errors and differences, focusing on training versus testing error (overfitting) and the difference between MAE and RMSE as indicators of large errors.""")
    # st.latex(r'''\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|''')
    # st.latex(r'''\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2''')
    # st.latex(r'''\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}''')

    st.header('Machine Learning Models', divider=header_divider)
    # Selection box for regression models
    choice = ['Decision Tree Regression', 'Random Forest Regression', 'XGBoost']
    option = st.selectbox('Choose Regression Machine Learning Models', choice)

    # Display the chosen model
    # Decision Tree: caluclate evaluation metrics on the spot
    if option == 'Decision Tree Regression':

        with st.expander('Decision Tree Algorithm'):
            st.markdown("""The Decision Tree algorithm for regression tasks functions by **dividing the data into separate nodes** or groups according to feature values. The goal is to **minimize the difference between the average target value** for the data points within each node **and the actual target values**. 
                        \nWhen making predictions, the algorithm returns the **average target value** for the node to which a new data point belongs.""")
        with st.expander('Why we chose Decision Trees'):
            st.markdown("""
            - **White Box Model**: Decision Trees offer transparent and logical explanations for decisions
            - **Feature Selection**: Implicit feature selection identifies key variables for predicting CO² emissions
            - **Non-linear Patterns**: Captures non-linear relationships which is relevant for the dataset
            - **Computational Efficiency**: Scales efficiently with dataset size, which is important for our dataset with 1.8 million observations.""")
        with st.expander('Results'):
            st.write(get_evaluation_table(dtr_model, X_train, X_test, y_train, y_test).round(3)) 
            st.markdown("""
            - High goodness of fit with $R^2$ score of 0.997
            - Low errors on both training and testing data
            - Overfitting does not seem to be an issue as indicated by the low difference between training and testing error
            - The model seems to make a few large errors, as indicated by the difference between MAE and RMSE""")

    # Random Forest: load evaluation metrics from pickle file (too long to calculate on the spot)
    elif option == 'Random Forest Regression':
        results = joblib.load('./streamlit_saved_files/results_rfr.pkl')
        with st.expander('Random Forest Algorithm'):
            st.write('The Random Forest Regression is a **bagging algorithm** that uses a **collection of decision trees** trained on different subsets of the data (**bootstraps**) and averages their predictions to improve accuracy and control overfitting.')
        with st.expander('Why we chose Random Forest'):
            st.markdown("""
            - **Non-Parametric Algorithm** and therfore suitable for the dataset
            - **Feature Importance Measure** for model interpretation, which is our main goal
            - **High Prediction Scores** by capturing complex relationships in the data through the combination of multiple Decision Trees
            - **Less prone to overfitting** compared to individual Decision Trees due to their ensemble approach and the use of random feature subsets
            """)
        with st.expander('Results'):
            st.write(results.round(3))
            st.markdown("""
            - High goodness of fit with $R^2$ score of 0.997
            - Low errors on both training and testing data
            - Overfitting does not seem to be an issue as indicated by the low difference between training and testing error
            - The model seems to make a few large errors, as indicated by the difference between MAE and RMSE""")
            # st.write(get_evaluation_table(rfr_model, X_train, X_test, y_train, y_test).round(2))
    # XGBoost: calculate evaluation metrics on the spot
    elif option == 'XGBoost':
        with st.expander('XGBoost Algorithm'):
            st.markdown("""XGBoost for regression is a boosting algorithm that uses a collection of decision trees, where **each subsequent tree** is built to **correct the errors made by the previous ones**, and combines their outputs **using a weighted sum** to make **final predictions**, with the weights optimized using gradient descent.""")
        with st.expander('Why we chose XGBoost'):
            st.markdown("""
            - **Superior Performance**: XGBoost excels in ML competitions, offering high-performance compared to other algorithms.
            - **Efficient Memory and Speed**: Optimized for both memory efficiency and computational speed (compared to other boosting algorithms), ideal for large datasets.
            - **Built-in Regularization**: Includes L1 (Lasso) and L2 (Ridge) regularization, preventing overfitting and making it superior to other Boosting Algorithms.""")
        with st.expander('Results'):
            st.write(get_evaluation_table(xgbr_model, X_train, X_test, y_train, y_test).round(3))
            st.markdown("""
            - High goodness of fit with $R^2$ score of 0.995
            - Low errors on both training and testing data
            - Overfitting does not seem to be an issue as indicated by the low difference between training and testing error
            - The model seems to make a few large errors, as indicated by the difference between MAE and RMSE""")

#######################################################################################################################################################
#Regression - Deep Learning
#######################################################################################################################################################

if page == 'Regression (DL)':
    # Load Model with joblib
    dnn_model = joblib.load('.\streamlit_saved_files\dnn_model_2.joblib')   
    
    # Load Training History with joblib
    #training_history = joblib.load('.\streamlit_saved_files\training_history_dnn_model_2')
    with open('.\\streamlit_saved_files\\training_history_dnn_model_2', 'rb') as file:
        training_history = pickle.load(file)


    # Introduction
    st.header('Deep Neural Network Regression', divider=header_divider)
    with st.expander('Deep Neural Network Algorithm'):
        st.write('A Deep Neural Network (DNN) for regression tasks is a type of artificial neural network that has many **layers between the input and output layers**. It learns to connect input features to a continuous output by **reducing the difference between the predicted and actual target values**, using techniques like backpropagation and gradient descent optimization.')
    with st.expander('Deep Neural Network Algorithm'):
        st.markdown("""
        - **Complex Pattern Modeling**: DNNs excel in capturing complex patterns and non-linear relationships
        - **Handling Large Tabular Datasets**: Well-suited for projects with large tabular datasets, ensuring efficient processing
        - **Model interpretability** is not as straightforward but possibble with shap or skater packages""")
    
    # Show Model Summary
    st.header('Constructing the DNN model', divider=header_divider)
    regression_dl_summary_comments()
    with st.expander('Show Model Summary'):
        dl_show_model_summary(dnn_model)

    # Show Learning Curve
    st.header('Learning Curve', divider=header_divider)
    st.write('The model is compiled with **Adam optimizer** and **Mean Squared Error** as loss function and a **learning rate of 0.001**.')
    dl_plot_loss(training_history)
    st.write('Both training and validation loss decrease over time, and stabalize after around 15 epochs. An increase in the **number of epochs** would not seem to improve the model performance.')
    st.write('The model does **not seem to overfit**, as the training and validation loss converge.')
    
    # Show Results
    with st.expander('Results'):
        results = joblib.load('./streamlit_saved_files/results_dnnr.pkl')
        st.write(results)
        st.markdown("""
        - High goodness of fit with $R^2$ score of 0.99
        - Low errors on both training and testing data
        - Overfitting does not seem to be an issue as indicated by the low difference between training and testing error
        - The model seems to make a few large errors, as indicated by the difference between MAE and RMSE""")


#######################################################################################################################################################
#Regression - Model Selection
#######################################################################################################################################################
if page == 'Regression (Model Selection)':
    
    # Load Models with joblib
    gsr = joblib.load('.\\streamlit_saved_files\\grid_search_r.joblib') 
    rsr = joblib.load('.\\streamlit_saved_files\\random_search_r.joblib')
    
    st.header('Model Selection', divider=header_divider)
    
    # Table with results
    regression_model_selection = joblib.load('./streamlit_saved_files/regression_model_selection_results.pkl')
    st.write(regression_model_selection.round(3))
    st.write('All models have a **high goodness of fit**, with a $R^2$ above of 0.99 and **low errors**, considerung the range of the target variable:')
    
    # Boxplot
    with st.expander('Show target range'):
        plot_ewltp_boxplot(df_preprocessed)
    st.write('The Random Forest Regressor and Decision Tree Regressor perform better compared to the other models. We selected the **Random Forest Regressor** as our final model because it offers greater stability compared to the Decision Tree Regressor, resulting in more reliable interpretations of feature importance.')
    
    st.header('Hyperparameter Tuning', divider=header_divider)
    
    with st.expander('Random Search'):
        st.write('We started Hyperparameter Tuning with Random Search, because its more **time efficient**, compared to Grid Search. Random Search randomly selects a combination of hyperparameters from a predefined range or selection.')
        #with st.expander('Random Search'):

        st.write('The following hyperparameters were selected for Random Search:')
        param_grid_code = """
        param_dist = {
            'n_estimators': randint(10, 200),
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'bootstrap': [True, False]
        }
        """
        st.code(param_grid_code, language='python')
        st.write('The best model is:')
        st.code(pprint.pformat(rsr.best_estimator_), language='python')

    with st.expander('Grid Search'):
        st.write('Random Search was not successful, therfore we continued with **Grid Search**. Grid Search evaluates **every possible combination** of the hyperparameters in the predefined hyperparameter grid. selects a combination of hyperparameters from a predefined range or selection. We chose it, because its more time efficient, compared to Grid Search.')
        st.write('We selected the following, simplyfied hyperparameter grid, because of the high computational cost:')
        param_grid_code = """
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None, 10, 20],
            'random_state': [123],
            'criterion': ['squared_error']
        }
        """
        st.code(param_grid_code, language='python')        
        st.write('The best model is:')
        st.code(pprint.pformat(gsr.best_estimator_), language='python')

    hpt_results = joblib.load('./streamlit_saved_files/regression_hyperparameter_tuning_results.pkl')
    st.write(hpt_results)
    st.write('The **Random Search was not successful**, as the best model performs worse than the baseline model.')
    st.write('The **Grid Search sligthly improved the performance** of the baseline model, by increasing the number of estimators from 100 to 200.')
    st.header('Final Regression Model', divider=header_divider)
    st.write('Our final regression model is the **Random Forest Regressor** with the following hyperparameters:')
    st.write(gsr.best_estimator_)

#######################################################################################################################################################
#Classification - Machine Learning
#######################################################################################################################################################
if page == 'Classification (ML)':
    results_ml_clf(df)


#######################################################################################################################################################
#Classification - Deep Learning
#######################################################################################################################################################

if page == 'Classification (DL)':
    data_folder = '.\streamlit_saved_files' 
    st.subheader("Classification task", divider=header_divider)
    classification_intro()()
    st.markdown("""## Our Approach""")
    # Introduction
    classification_dl_approach()
    # First Model __________________________________________________________________________________________________________
    
    st.header('First Model', divider=header_divider)
    
    trained_model = 'dnn_FM_classification_no_os.h5'
    model_summary_filename = 'dnn_FM_model_summary.PNG' 
    tr_cm_cr_pickle_filename = 'dnn_FM_classification_data_no_os.pkl'
    
    #Displays a full report about a DNN model including    
    code = '''
    #Build the DNN model using Sequential()
    dnn = Sequential()
    dnn.add(Dense(units=32, activation='relu', name='dense1' ))
    dnn.add(Dense(units=128, activation='relu', name='dense2'))
    dnn.add(Dense(units=7, activation='softmax', name='dense3'))

    #initialize Adam Optimizer
    optimizer = Adam(learning_rate=0.001)

    #Compile the model
    dnn.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])

    #Train the model
    training_history = dnn.fit(X_train_dnn, y_train_dnn,
                                 epochs = 20,
                                 batch_size = 1024,
                                 validation_data=(X_test_dnn, y_test_dnn)) '''
    cm_first_model, df_cr_first_model, tr_hist_first_model = model_full_report('First Model', data_folder, trained_model, model_summary_filename, tr_cm_cr_pickle_filename, code)
    
    
    
    # Random Search __________________________________________________________________________________________________________
    st.header('Random Search', divider=header_divider)
    #st.markdown("#### Keras Model Building")
    #st.subheader('Keras Model Building')
    code = '''
    # Function to create a Keras model
    def create_model(input_dim, learning_rate=0.01, units_layer1=32, units_layer2=128, units_layer3=128):
        model = Sequential()
        model.add(Dense(units=units_layer1, activation='relu', input_dim=input_dim))
        model.add(Dense(units=units_layer2, activation='relu'))
        model.add(Dense(units=units_layer3, activation='relu'))
        model.add(Dense(units=7, activation='softmax'))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model '''
    
    best_accuracy = 0.95
    
    best_hyperparameters = {'Parameter': ['batch_size', 'epochs', 'learning_rate', 'units_layer1', 'units_layer2', 'units_layer3'],
						'Search Space': ['[512, 1024, 10240, 51200]', 'randint(35, 50, 100)', 'uniform(0.0001, 0.2)', 'randint(16, 64)', 'randint(64, 256)', 'randint(64, 256)'],
						'Selected Value': [10240, 148, 0.0075, 59, 111, 112]}
                        
    df_rs = pd.DataFrame(best_hyperparameters)
    df_rs.set_index(df_rs.columns[0], inplace=True)
    st.dataframe(df_rs)
    st.markdown(f'**Achieved Accuracy:** {best_accuracy:.2f}')
    
    #Selected Model - Without Oversampling  __________________________________________________________________________________________________________
    st.header('Selected Model - Without Oversampling', divider=header_divider)
    
    
    trained_model = 'dnn_RS_classification_no_oversampling.h5'
    model_summary_filename = 'dnn_RS_model_summary.PNG' 
    tr_cm_cr_pickle_filename = 'dnn_RS_classification_data_no_os.pkl'
    
    txt = '''
    The model obtained by Random Search is trained by incorporating callbacks: 
    - **Early Stopping**: Prevents overfitting and useless training Epochs 
    - **Learning Rate Scheduler**: Adapts the learning rate dynamically for optimized convergence
    '''
    st.markdown(txt)
    
    code = '''
        dnn.compile(loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', mean_roc_auc, mean_f1_score])

        # Example code for training with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=20)
        # Set up ReduceLROnPlateau callback
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=15, min_lr=1e-5, min_delta=0.05)

        # Train the model
        training_history = dnn.fit(X_train_dnn_ros, y_train_dnn_ros,
                                   epochs=148,
                                   batch_size=10240,
                                   validation_data=(X_valid_dnn_ros, y_valid_dnn_ros),
                                   callbacks=[lr_scheduler, early_stop]) '''    
    #Displays a full report about a DNN model including    
    cm_rs_model, df_cr_rs_model, tr_hist_rs_model = model_full_report('Selected Model (Without Oversampling)', data_folder, trained_model, model_summary_filename, tr_cm_cr_pickle_filename, code)
    
    hist_dict={
        'First Model': tr_hist_first_model,
        'Selected Model (Without Oversampling)': tr_hist_rs_model,
    }
    st.subheader('Random Search Impact')
    compare_history_plot(hist_dict)
    
    # Final Model __________________________________________________________________________________________________________
    st.header('Selected Model', divider=header_divider)
    
    st.subheader('Oversampling')
    txt = '''
    - The trained **DNN** model outperforms the initial model but doesn't match the performances of the **ML** classifiers. 
    - **Imbalanced classes** may be a contributing factor. 
    - Applying **oversampling** techniques could help mitigate this issue and improve classification performance.
    '''
    st.markdown(txt)

    trained_model = 'dnn_SM_classification_os.h5'
    model_summary_filename = 'dnn_SM_model_summary.PNG' 
    tr_cm_cr_pickle_filename = 'dnn_SM_classification_data_os.pkl'
    
    #Displays a full report about a DNN model including  
    cm_final_model, df_cr_final_model, tr_hist_final_model = model_full_report('Selected Model (With Oversampling)', data_folder, trained_model, model_summary_filename, tr_cm_cr_pickle_filename, code=None)
    
    # Compare history plot
    hist_dict={
        'Selected Model (Without Oversampling)': tr_hist_rs_model,
        'Selected Model (With Oversampling)': tr_hist_final_model
    }
    st.subheader('Oversampling Impact')
    compare_history_plot(hist_dict)
    
    #Results
    st.subheader('Results')

    st.subheader('Class Predictions')
    metric_choice = range(0, 7)
    option = st.selectbox('Class Predictions distribution', metric_choice)
    get_class_predictions(cm_final_model, option)
    
    # Compare 3 Models
    st.subheader('Model Comparison')
    metric_choice = ['Precision', 'Recall', 'F1-Score']
    option = st.selectbox('Choice of Metric', metric_choice)
    compare_prec_recall_f1score([df_cr_final_model, df_cr_rs_model, df_cr_first_model], option)
    
    
    st.subheader('Potential Improvement')
    txt = '''
    To enhance the model, we could consider addressing **skewness** in numerical features through appropriate transformations.
    '''
    st.markdown(txt)
#######################################################################################################################################################
#Interpretation
#######################################################################################################################################################

if page == 'Interpretation':
    # Path of saved model file
    grid_search = joblib.load('.\streamlit_saved_files\grid_search_r.joblib')

    # Load pretrained model (Best model)
    rf_optimized = rf_optimized_grid = grid_search.best_estimator_

    st.header('Feature Importance', divider=header_divider)

    # Explain Feature Importance
    st.write('Feature importance in a Random Forest Regression Model measures the **average reduction in the impurity (MSE)** brought about by each feature in the ensemble of decision trees.')

    # Load predictions
    y_pred_train = joblib.load('.\\streamlit_saved_files\\y_pred_rf_optimized_train.pkl')

    # Plt feature importance
    plot_feature_importance(rf_optimized, X_train)

    # Interpretation
    st.write('The 10 most important features accumulate to **98.87%** of the total feature importance. The 5 most important features contribute to **92.04%** of the total reduction in MSE. ')

    st.header('Feature impact analysis', divider=header_divider)

    # Selection box for regression models
    choice = ['Categorical', 'Numerical']
    option = st.selectbox('How do the 5 most important features effect influnce the predicted CO\u00B2 emission?', choice)

    # Display the chosen model
    if option == 'Categorical':

        # Boxplots
        plot_boxplots_interpretation(X_train, y_pred_train)
        
        # Compare means
        st.markdown("""Both petrol and hybrid cars have a **lower predicted CO\u00B2 emission**, compared to other fuel types and fuel modes.""")
 
    elif option == 'Numerical':
        
        # Plot scatterplots for most important numerical features
        st.image('.\\plots\\interpretation_scatterplot_z.png')
        st.write(f"There does **not seem to be a clear relationship** between Electric Energy consumption and predicted CO\u00B2 emission for hybrid cars. Non-hybrid cars don't have any electric energy consumption.")
                
        st.image('.\\plots\\interpretation_scatterplot_ep_m.png')
        st.write(f"There seems to be a **positive relationship** between **Engine Power and predicted CO\u00B2 emission** for both hybrid and non-hybrid cars. The relationship is stronger for non-hybrid cars.")
        st.write(f"**Heavier cars** tend to have a **higher predicted CO\u00B2 emission**. The relationship is stronger for non-hybrid cars.")

       # plot_scatterplots_interpretation(X_train_wo_fc, y_pred_train)


#######################################################################################################################################################
#Conclusion
#######################################################################################################################################################


if page == 'Conclusion':

    st.header('Summary and Insights', divider=header_divider)
    with st.expander('Summary'):
        st.markdown("""
    	- Successfully trained highly accurate models, with **Random Forest Regression achieving 99.7% R2** and **Decision Tree Classification achieving 97% Accuracy**
    	- Chose regression over classification for more nuanced insights""")

    with st.expander('Insights'):
        st.markdown("""
    	- Identified **fuel consumption** as the primary determinant of CO² emissions
    	- **Hybrid cars** and cars with **petrol fuel type** have a lower predicted CO² emission
    	- **Engine power** and **car mass** have a positive relationship with predicted CO² emission""")
    
    st.header('Difficulties', divider=header_divider)
    with st.expander('Computational Limitations'):
        st.markdown("""
        - We **could not use all data** from cars registered in europe due to computational limitations
        - Faced **challenges with hyperparameter tuning** due to the complexity of the selected algorithms but overcame hyperparameter tuning challenges by carefully selecting a smaller parameter grid
        - **Couldn't use SHAP values** for model interpretation due to computational limitations""")
    with st.expander('Dominant Features'):
        st.markdown("""
        - Initial model was overly reliant on fuel consumption and electric range, which reduced model interpretabilitiy""")

    st.header('Continuation', divider=header_divider)
    with st.expander('Further improve model performance'):
        st.markdown("""
        - **Expand dataset** to include all cars registered within the EU for better model generalizability
        - **Prioritize parameter optimization** before comparing performances for potentially superior model outcomes
        - Conduct **more exhaustive hyperparameter optimization** using Grid Search with a bigger hyperparameter grid for superior results
        - Utilize **SHAP values** for a deeper understanding of the model's behavior in model interpretation""")
    with st.expander('Estimate CO\u00B2 emission of hybrid cars'):
        st.markdown("""- Consider **electricity consumption** of hybrid cars and **estimate CO\u00B2 emissions** using the energy consumption and CO\u00B2 intensity:""")
        st.latex(r'''\mathrm{CO^2\ emissions\ (g/km)} = \mathrm{Energy\ consumption\ (kWh/km)} \times \mathrm{CO^2\ intensity\ (g/kWh)}''')
        st.markdown(""" - **Standardize WLTP testing procedure** to account **for energy consumption** and ensure accurate CO\u00B2 emissions estimation for hybrid and electric cars""")

    st.header('Potential Applications', divider=header_divider)
    with st.expander('Use insights'):
        st.markdown("""
                    Stakeholders can **use insights** about the most important factors influencing CO\u00B2 emissions
                    - **Government and Policymakers**: Use insights to create regulations and incentives for cleaner transportation
                    - **Insurance Companies**: Create tailored insurance products and pricing strategies encouraging customers to choose cleaner vehicles
                    - **Educational Institutions**: Use the model as a teaching tool for deeper understanding of factors influencing CO\u00B2 emissions""")

    with st.expander('Use predictions'):
        st.markdown("""- **Car manufacturers**: Directly use the model to estimate CO\u00B2 emissions, **reducing testing costs and development time**. Consider **trade-offs** between accuracy and reduced testing costs to meet regulatory requirements.""")