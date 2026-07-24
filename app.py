import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and data
model = pickle.load(open("LinearRegression.pkl", "rb"))
car = pd.read_csv("Car_Dekho_Cleaned.csv")

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗Car Price Prediction System")

# Company
companies = ["Select Company"] + sorted(car['company'].unique())
company = st.selectbox("Select Company", companies, index=0)

# Car Model
if company == "Select Company":
    car_model = st.selectbox("Select Car Model", [], disabled=True)
else:
    car_models = ["Select Car Model"] + sorted(
        car[car['company'] == company]['name'].unique()
    )
    car_model = st.selectbox("Select Car Model", car_models, index=0)

# Year
years = ["Select Year"] + sorted(car['year'].unique(), reverse=True)
year = st.selectbox("Select Year", years, index=0)

# Fuel Type
fuel_types = ["Select Fuel Type"] + sorted(car['fuel_type'].unique())
fuel_type = st.selectbox("Select Fuel Type", fuel_types, index=0)

# KMS Driven
kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1000,
    value=0
)

# Prediction
if st.button("Predict Price"):
    if (
        company == "Select Company"
        or car_model == "Select Car Model"
        or year == "Select Year"
        or fuel_type == "Select Fuel Type"
        or kms_driven == 0
    ):
        st.warning("⚠️ Please fill all fields correctly")
    else:
        input_df = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )

        prediction = model.predict(input_df)
        st.success(f"💰 Estimated Price: ₹ {round(prediction[0], 2)}")

