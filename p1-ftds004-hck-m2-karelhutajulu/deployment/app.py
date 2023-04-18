import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
import json

# Loading encoder and model
model = joblib.load("/Users/karelhutajulu/development/streamlit/P1ML2/deployment/decision_tree_best.joblib")
encoder = joblib.load("/Users/karelhutajulu/development/streamlit/P1ML2/deployment/encoder.joblib")

#categories options
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_types_collision = ['Vehicle with vehicle collision','Collision with roadside objects',
                           'Collision with pedestrians','Rollover','Collision with animals',
                           'Unknown','Collision with roadside-parked vehicles','Fall from vehicles',
                           'Other','With Train']

options_sex = ['Male','Female','Unknown']

options_education_level = ['Junior high school','Elementary school','High school',
                           'Unknown','Above high school','Writing & reading','Illiterate']

options_services_year = ['Unknown','2-5yrs','Above 10yr','5-10yrs','1-2yr','Below 1yr']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

# features list
features = ['Number_of_vehicles_involved','Number_of_casualties','Hour_of_Day','Type_of_collision','Age_band_of_driver','Sex_of_driver',
       'Educational_level','Service_year_of_vehicle','Day_of_week','Area_accident_occured']

st.image("/Users/karelhutajulu/Downloads/JTT logo.png", use_column_width=True)

with st.form("severity_form"):
    No_vehicles = st.slider("Number of vehicles involved:", 1, 7, value=1, format="%d")
    No_casualties = st.slider("Number of casualities:", 1, 8, value=1, format="%d")
    Hour = st.slider("Hour of the day:", 0, 23, value=0, format="%d")
    collision = st.selectbox("Type of collision:",options=options_types_collision)
    Age_band = st.selectbox("Driver age group:", options=options_age)
    Sex = st.selectbox("Sex of the driver:", options=options_sex)
    Education = st.selectbox("Education of driver:",options=options_education_level)
    service_vehicle = st.selectbox("Service year of vehicle:", options=options_services_year)
    Day_week = st.selectbox("Day of the week:", options=options_day)
    Accident_area = st.selectbox("Area of accident:", options=options_acc_area)

    submitted = st.form_submit_button("Predict")


# encode using ordinal encoder and predict
if submitted:
    cat_features = [collision, Age_band, Sex, Education, service_vehicle, Day_week, Accident_area]
    encoded_arr = list(encoder.transform([cat_features]).ravel())
    num_arr = [No_vehicles,No_casualties,Hour]
    pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)

    # predict the target from all the input features
    prediction = model.predict(pred_arr)

    if prediction == 0:
        st.write(f"Fatal Injury")
        st.image("/Users/karelhutajulu/development/streamlit/P1ML2/deployment/WhatsApp-Image-2021-05-11-at-05.19.09.jpeg", use_column_width=True)
    elif prediction == 1:
        st.write(f"Serious injury")
        st.image("/Users/karelhutajulu/development/streamlit/P1ML2/deployment/nissan-zero-emission-ambulance.jpg", use_column_width=True)
    else:
        st.write(f"Slight injury")
        st.image("/Users/karelhutajulu/development/streamlit/P1ML2/deployment/20210417_162933jpg-20210417041829.jpg", use_column_width=True)
