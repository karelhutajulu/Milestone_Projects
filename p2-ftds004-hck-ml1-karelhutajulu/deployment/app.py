import pickle
import pandas as pd
import streamlit as st
import numpy
import json

    #Load All Files
with open('preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

with open('ann_best.pkl', 'rb') as f:
    ann_model = pickle.load(f)

header_image = 'Customer-Churn.png'
st.image(header_image, use_column_width=True)


# Create the form
st.write('# Customer Churn Prediction')
st.write('Phase 2 Milestone 1, deployed by: Karel Hutajulu')
st.write('Fill in the following fields to predict if a customer will churn.')

with st.form(key='Customer Information'):
    user_id = st.text_input('User ID')
    age = st.number_input('Age', min_value=0)
    gender = st.selectbox('Gender', ['F', 'M'])
    region_category = st.selectbox('Region Category', ['City', 'Village', 'Town'])
    membership_category = st.selectbox('Membership Category', ['No Membership', 'Basic Membership', 'Silver Membership', 'Premium Membership', 'Gold Membership', 'Platinum Membership'])
    joining_date = st.number_input('Joining Date in days since 2000', min_value=0)
    joined_through_referral = st.selectbox('Joined Through Referral', ['Yes', 'No'])
    preferred_offer_types = st.selectbox('Preferred Offer Types', ['Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'])
    medium_of_operation = st.selectbox('Medium of Operation', ['Desktop', 'Smartphone', 'Both'])
    internet_option = st.selectbox('Internet Option', ['Wi-Fi', 'Fiber_Optic', 'Mobile_Data'])
    last_visit_time = st.number_input('Visit Time in seconds since midnight', min_value=0)
    days_since_last_login = st.number_input('Days Since Last Login', min_value=0)
    avg_time_spent = st.number_input('Average Time Spent (in minutes)', min_value=0)
    avg_transaction_value = st.number_input('Average Transaction Value', min_value=0)
    avg_frequency_login_days = st.number_input('Average Frequency Login Days', min_value=0)
    points_in_wallet = st.number_input('Points in Wallet', min_value=0)
    used_special_discount = st.selectbox('Used Special Discount', ['Yes', 'No'])
    offer_application_preference = st.selectbox('Offer Application Preference', ['Yes', 'No'])
    past_complaint = st.selectbox('Past Complaint', ['Yes', 'No'])
    complaint_status = st.selectbox('Complaint Status', ['No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'])
    feedback = st.selectbox('Feedback', ['Poor Website', 'Poor Customer Service', 'Too many ads', 'Poor Product Quality', 'No reason specified', 'Products always in Stock', 'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'])
    st.markdown('---')

    # Add a submit button to the form
    submitted = st.form_submit_button('Predict')

# Preprocess the input data
data_inf = {
    'age': age,
    'gender': gender,
    'region_category': region_category,
    'membership_category': membership_category,
    'joining_date': joining_date,
    'joined_through_referral': joined_through_referral,
    'preferred_offer_types': preferred_offer_types,
    'medium_of_operation': medium_of_operation,
    'internet_option': internet_option,
    'last_visit_time': last_visit_time,
    'days_since_last_login': days_since_last_login,
    'avg_time_spent': avg_time_spent,
    'avg_transaction_value': avg_transaction_value,
    'avg_frequency_login_days': avg_frequency_login_days,
    'points_in_wallet': points_in_wallet,
    'used_special_discount': used_special_discount,
    'offer_application_preference': offer_application_preference,
    'past_complaint': past_complaint,
    'complaint_status': complaint_status,
    'feedback': feedback
}

data_inf = pd.DataFrame([data_inf])
st.dataframe(data_inf)

if submitted:

# # Apply the preprocessing pipeline to the input data
    data_inf = preprocessing_pipeline.transform(data_inf)

# # Make the prediction using the ANN model
    y_pred = ann_model.predict(data_inf)

# # Write the prediction to the user

    st.write('Prediction:')

    if y_pred[0] > 0.5:
        st.write("LIKELY TO CHURN DETECTED")
        st.write(f"User {user_id} is predicted to churn with a probability of {y_pred[0].item():.2f}.")
    else:
        st.write("NOT LIKELY TO CHURN DETECTED")
        st.write(f"User {user_id} is predicted not to churn with a probability of {1 - y_pred[0].item():.2f}.")
