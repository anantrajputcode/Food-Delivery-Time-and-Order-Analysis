
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Food Delivery Predictor",
    page_icon="🍔",
    layout="centered"
)

# Load trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_delivery_model.pkl"

model = joblib.load(MODEL_PATH)

# Application title
st.title("Food Delivery Time Predictor")
st.write("Enter order details to estimate delivery duration.")

st.divider()

# Input features
distance = st.number_input(
    "Delivery distance (km)",
    min_value=0.1,
    max_value=50.0,
    value=8.0
)

order_hour = st.slider(
    "Order hour",
    min_value=0,
    max_value=23,
    value=20
)

age = st.number_input(
    "Delivery partner age",
    min_value=18,
    max_value=70,
    value=28
)

rating = st.slider(
    "Delivery partner rating",
    min_value=1.0,
    max_value=5.0,
    value=4.5,
    step=0.1
)

traffic = st.selectbox(
    "Traffic density",
    ["Low", "Medium", "High", "Jam"]
)

weather = st.selectbox(
    "Weather conditions",
    ["Sunny", "Cloudy", "Windy", "Stormy", "Fog"]
)

order_type = st.selectbox(
    "Order type",
    ["Snack", "Meal", "Drinks", "Buffet"]
)

vehicle = st.selectbox(
    "Vehicle type",
    ["motorcycle", "scooter", "electric_scooter"]
)

vehicle_condition = st.selectbox(
    "Vehicle condition",
    [0, 1, 2, 3]
)

multiple_deliveries = st.selectbox(
    "Multiple deliveries",
    [0, 1, 2, 3]
)

festival = st.selectbox(
    "Festival",
    ["Yes", "No"]
)

city = st.selectbox(
    "City category",
    ["Metropolitian", "Urban", "Semi-Urban"]
)

# Prediction
if st.button("Predict Delivery Time", type="primary"):

    sample = pd.DataFrame([{
        "distance_km": distance,
        "order_hour": order_hour,
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Road_traffic_density": traffic,
        "Weather_conditions": weather,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries,
        "Festival": festival,
        "City": city
    }])

    prediction = model.predict(sample)[0]

    st.success(
        f"Estimated delivery time: {prediction:.1f} minutes"
    )

    st.caption(
        "This is a model estimate, not a guaranteed delivery time."
    )
