import streamlit as st
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBRegressor
with open("model2.pkl", "rb") as f:
    final_model = pickle.load(f)
final_model.set_params(device="cpu")

with open("le2.pkl", "rb") as f:
    le = pickle.load(f)

with open("ohe_2.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("scaler2.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Car Price Prediction App 🚗💰")

models = ['Alto', 'Grand', 'i20', 'Ecosport', 'Wagon R', 'i10', 'Venue',
          'Swift', 'Verna', 'Duster', 'Cooper', 'Ciaz', 'C-Class', 'Innova',
          'Baleno', 'Swift Dzire', 'Vento', 'Creta', 'City', 'Bolero',
          'Fortuner', 'KWID', 'Amaze', 'Santro', 'XUV500', 'KUV100', 'Ignis',
          'RediGO', 'Scorpio', 'Marazzo', 'Aspire', 'Figo', 'Vitara',
          'Tiago', 'Polo', 'Seltos', 'Celerio', 'GO', '5', 'CR-V',
          'Endeavour', 'KUV', 'Jazz', '3', 'A4', 'Tigor', 'Ertiga', 'Safari',
          'Thar', 'Hexa', 'Rover', 'Eeco', 'A6', 'E-Class', 'Q7', 'Z4', '6',
          'XF', 'X5', 'Hector', 'Civic', 'D-Max', 'Cayenne', 'X1', 'Rapid',
          'Freestyle', 'Superb', 'Nexon', 'XUV300', 'Dzire VXI', 'S90',
          'WR-V', 'XL6', 'Triber', 'ES', 'Wrangler', 'Camry', 'Elantra',
          'Yaris', 'GL-Class', '7', 'S-Presso', 'Dzire LXI', 'Aura', 'XC',
          'Ghibli', 'Continental', 'CR', 'Kicks', 'S-Class', 'Tucson',
          'Harrier', 'X3', 'Octavia', 'Compass', 'CLS', 'redi-GO', 'Glanza',
          'Macan', 'X4', 'Dzire ZXI', 'XC90', 'F-PACE', 'A8', 'MUX',
          'GTC4Lusso', 'GLS', 'X-Trail', 'XE', 'XC60', 'Panamera', 'Alturas',
          'Altroz', 'NX', 'Carnival', 'C', 'RX', 'Ghost', 'Quattroporte',
          'Gurkha']

seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
transmission_types = ['Manual', 'Automatic']
model = st.selectbox("Car Model", models)  # Dropdown with only valid car models
vehicle_age = st.number_input("Vehicle Age (in years)", min_value=0, max_value=29, value=10)
km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=3800000, value=1000)
seller_type = st.selectbox("Seller Type", seller_types)
fuel_type = st.selectbox("Fuel Type", fuel_types)
transmission_type = st.selectbox("Transmission Type", transmission_types)
mileage = st.number_input("Mileage (kmpl)", min_value=4.0, max_value=33.54, value=20.0)
engine = st.number_input("Engine Capacity (CC)", min_value=793, max_value=6951, value=1000)
max_power = st.number_input("Max Power (BHP)", min_value=38.4, max_value=626.0, value=46.3)
seats = st.number_input("Number of Seats", min_value=2, max_value=9, value=5)

if st.button("Predict Price"):
    try:
        car_data = pd.DataFrame({
            "model": [model],
            "vehicle_age": [vehicle_age],
            "km_driven": [km_driven],
            "seller_type": [seller_type],
            "fuel_type": [fuel_type],
            "transmission_type": [transmission_type],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats]
        })
        car_data["model"] = le.transform(car_data["model"])
        car_data_ohe = pd.DataFrame(ohe.transform(car_data[["seller_type", "fuel_type", "transmission_type"]]),
                                    columns=ohe.get_feature_names_out(["seller_type", "fuel_type", "transmission_type"]))
        car_data = car_data.drop(columns=["seller_type", "fuel_type", "transmission_type"])
        car_data = pd.concat([car_data, car_data_ohe], axis=1)
        desired_columns = [
            'model', 'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power',
            'seats', 'seller_type_Dealer', 'seller_type_Individual',
            'seller_type_Trustmark Dealer', 'fuel_type_CNG', 'fuel_type_Diesel',
            'fuel_type_Electric', 'fuel_type_LPG', 'fuel_type_Petrol',
            'transmission_type_Automatic', 'transmission_type_Manual'
        ]
        for col in desired_columns:
            if col not in car_data:
                car_data[col] = 0
        car_data = car_data[desired_columns]
        car_data_scaled = pd.DataFrame(scaler.transform(car_data), columns=desired_columns)
        prediction = final_model.predict(car_data_scaled)
        st.success(f"Predicted Car Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
