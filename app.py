import streamlit as st
import pickle
import pandas as pd
import numpy as np

def load_models():
    with open('ohe_model.pkl', 'rb') as file:
        ohe = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('best_model.pkl', 'rb') as file:
        best_model = pickle.load(file)
    
    le_dict = {}
    le_cols = ['Occupation', 'Designation', 'ProductPitched']
    for col in le_cols:
        with open(f'le_{col}_model.pkl', 'rb') as file:
            le_dict[col] = pickle.load(file)
    
    return ohe, scaler, best_model, le_dict

def preprocess(X_new, ohe, scaler, le_dict):
    le_cols = ['Occupation', 'Designation', 'ProductPitched']
    ohe_cols = ['TypeofContact', 'Gender', 'MaritalStatus']
    
    for col in le_cols:
        X_new[col] = le_dict[col].transform(X_new[col])
    
    X_new_encoded = ohe.transform(X_new[ohe_cols])
    X_new_encoded_df = pd.DataFrame(X_new_encoded, columns=ohe.get_feature_names_out(ohe_cols))
    X_new = pd.concat([X_new.drop(columns=ohe_cols), X_new_encoded_df], axis=1)
    
    feature_order = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
                     'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
                     'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
                     'MonthlyIncome', 'Total', 'TypeofContact_Company Invited',
                     'TypeofContact_Self Enquiry', 'Gender_Female', 'Gender_Male',
                     'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Unmarried',
                     'Occupation', 'Designation', 'ProductPitched']
    X_new = X_new[feature_order]
    X_new_scaled = scaler.transform(X_new)
    return X_new_scaled

def predict(X_new, ohe, scaler, best_model, le_dict):
    X_new_preprocessed = preprocess(X_new, ohe, scaler, le_dict)
    predictions = best_model.predict(X_new_preprocessed)
    return predictions

ohe, scaler, best_model, le_dict = load_models()

st.title("Customer Response Prediction")
with st.form("prediction_form"):
    st.subheader("Enter Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100)
    typeofcontact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    duration = st.number_input("Duration of Pitch", min_value=0.0)
    occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    num_visiting = st.number_input("Number of Person Visiting", min_value=1)
    num_followups = st.number_input("Number of Followups", min_value=0)
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe'])
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ['Married', 'Unmarried', 'Divorced', 'Single'])
    num_trips = st.number_input("Number of Trips", min_value=0)
    passport = st.selectbox("Has Passport", [0, 1])
    pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Own Car", [0, 1])
    num_children = st.number_input("Number of Children Visiting", min_value=0, value=1)
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP'])
    monthly_income = st.number_input("Monthly Income", min_value=1000)
    
    submit_button = st.form_submit_button("Predict")

if submit_button:
    data = pd.DataFrame({
        'Age': [age], 'TypeofContact': [typeofcontact], 'CityTier': [citytier], 'DurationOfPitch': [duration],
        'Occupation': [occupation], 'Gender': [gender], 'NumberOfPersonVisiting': [num_visiting],
        'NumberOfFollowups': [num_followups], 'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [preferred_star], 'MaritalStatus': [marital_status],
        'NumberOfTrips': [num_trips], 'Passport': [passport], 'PitchSatisfactionScore': [pitch_score],
        'OwnCar': [own_car], 'NumberOfChildrenVisiting': [num_children], 'Designation': [designation],
        'MonthlyIncome': [monthly_income], 'Total': [num_children+num_visiting]
    })
    
    prediction = predict(data, ohe, scaler, best_model, le_dict)
    st.success(f"Prediction: {'Interested' if prediction[0] == 1 else 'Not Interested'}")

st.subheader("Upload CSV for Batch Predictions")
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    predictions = predict(df, ohe, scaler, best_model, le_dict)
    df['Prediction'] = ['Interested' if pred == 1 else 'Not Interested' for pred in predictions]
    st.write(df)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
