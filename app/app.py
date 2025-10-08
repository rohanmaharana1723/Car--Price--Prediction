import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load the trained model ---
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Automobile Price Prediction", page_icon="🚗", layout="centered")
st.title("🚗 Automobile Price Prediction")
st.write("Predict the price of a car using its specifications.")

# --- Sidebar Inputs ---
st.sidebar.header("Enter Car Specifications")

symboling = st.sidebar.number_input("Symboling (Insurance risk rating)", -3, 3, 0)
normalized_losses = st.sidebar.number_input("Normalized Losses", 50, 300, 100)
make = st.sidebar.selectbox("Make", 
    ["audi", "bmw", "chevrolet", "dodge", "honda", "jaguar", "mazda",
     "mercedes-benz", "mitsubishi", "nissan", "peugeot", "plymouth",
     "porsche", "renault", "saab", "subaru", "toyota", "volkswagen", "volvo"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["gas", "diesel"])
body_style = st.sidebar.selectbox("Body Style", ["convertible", "hatchback", "sedan", "wagon", "hardtop"])
drive_wheels = st.sidebar.selectbox("Drive Wheels", ["fwd", "rwd", "4wd"])
engine_location = st.sidebar.selectbox("Engine Location", ["front", "rear"])
width = st.sidebar.number_input("Width (in inches)", 60.0, 80.0, 66.0)
height = st.sidebar.number_input("Height (in inches)", 45.0, 65.0, 52.0)
engine_type = st.sidebar.selectbox("Engine Type", ["dohc", "ohcv", "ohc", "rotor"])
engine_size = st.sidebar.number_input("Engine Size (cc)", 50, 400, 130)
horsepower = st.sidebar.number_input("Horsepower (hp)", 40, 300, 100)
city_mpg = st.sidebar.number_input("City MPG", 10, 60, 25)
highway_mpg = st.sidebar.number_input("Highway MPG", 10, 60, 30)

# --- Create DataFrame ---
input_data = pd.DataFrame({
    "symboling": [symboling],
    "normalized-losses": [normalized_losses],
    "make": [make],
    "fuel-type": [fuel_type],
    "body-style": [body_style],
    "drive-wheels": [drive_wheels],
    "engine-location": [engine_location],
    "width": [width],
    "height": [height],
    "engine-type": [engine_type],
    "engine-size": [engine_size],
    "horsepower": [horsepower],
    "city-mpg": [city_mpg],
    "highway-mpg": [highway_mpg],
})

# --- One-hot encode (match training columns) ---
input_encoded = pd.get_dummies(input_data)

# Align with training columns
try:
    input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
except AttributeError:
    st.warning("⚠️ Your model does not store feature names. Make sure your input encoding matches training columns.")

# --- Predict Button ---
if st.button("Predict Price 💰"):
    try:
        prediction = model.predict(input_encoded)[0]
        st.success(f"Estimated Car Price: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Make sure model.pkl was trained on encoded (numeric) features using pd.get_dummies or LabelEncoder.")

st.markdown("---")
st.caption("Made with ❤️ by Jupiter | Data Analytics & Machine Learning Project")
