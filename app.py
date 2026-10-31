
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DeliverIQ | Food Delivery Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_delivery_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "food_delivery.csv"

# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111a2e;
    border-right: 1px solid #25334b;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e3a5f, #164e63);
    padding: 32px;
    border-radius: 18px;
    border: 1px solid #315477;
    margin-bottom: 26px;
}

.hero h1 {
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 15px;
    margin-bottom: 0;
}

.section-heading {
    font-size: 19px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 15px 0 12px 0;
}

.metric-card {
    background: #151f33;
    border: 1px solid #293951;
    padding: 20px;
    border-radius: 14px;
    min-height: 110px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    color: #f8fafc;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}

.result-card {
    background: linear-gradient(135deg, #123b37, #164e63);
    border: 1px solid #2dd4bf;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 22px;
}

.result-label {
    color: #a7f3d0;
    font-size: 14px;
}

.result-value {
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin: 8px 0;
}

.result-subtitle {
    color: #ccfbf1;
    font-size: 14px;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-weight: 700;
    background: #2563eb;
    color: white;
    border: none;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

div[data-testid="stForm"] {
    background: #111a2e;
    padding: 22px;
    border: 1px solid #293951;
    border-radius: 16px;
}

hr {
    border-color: #293951;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL AND DATASET
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_dataset()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def get_options(column):
    """Get actual categories from the project dataset."""
    values = df[column].dropna().unique().tolist()

    # Normalize text categories
    if values and isinstance(values[0], str):
        values = sorted(set(v.strip() for v in values))

    else:
        values = sorted(values)

    return values


def get_default(options, preferred):
    """Choose a preferred default if it exists."""
    return options.index(preferred) if preferred in options else 0


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🚀 DeliverIQ")
    st.caption("Food Delivery Intelligence")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Delivery Prediction", "About the Model"]
    )

    st.divider()

    st.markdown("### Model Information")
    st.markdown("**Algorithm:** Random Forest")
    st.markdown("**Target:** Delivery Time")
    st.markdown("**Unit:** Minutes")

    st.divider()

    st.caption("Built with Python, Streamlit & Scikit-learn")


# =========================================================
# 6. HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">
    <h1>Food Delivery Intelligence</h1>
    <p>
        Predict delivery duration using machine learning.
        Explore how distance, traffic, weather and other factors
        influence delivery time.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 7. PREDICTION PAGE
# =========================================================

if page == "Delivery Prediction":

    # Top metrics
    col1, col2, col3 = st.columns(3)

    metrics = [
        ("MODEL", "Random Forest"),
        ("R² SCORE", "81.07%"),
        ("AVERAGE ERROR", "3.22 min")
    ]

    for col, (label, value) in zip([col1, col2, col3], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">📦 Order Details</div>',
                unsafe_allow_html=True)

    with st.form("prediction_form"):

        # -----------------------------
        # DELIVERY INFORMATION
        # -----------------------------

        st.markdown("#### 📍 Delivery Information")

        c1, c2 = st.columns(2)

        with c1:
            distance = st.number_input(
                "Delivery distance (km)",
                min_value=0.1,
                max_value=50.0,
                value=8.0,
                step=0.1
            )

            order_hour = st.slider(
                "Order hour",
                0, 23, 20
            )

            city_options = get_options("City")
            city = st.selectbox(
                "City category",
                city_options,
                index=get_default(city_options, "Urban")
            )

        with c2:
            traffic_options = get_options("Road_traffic_density")
            traffic = st.selectbox(
                "Traffic density",
                traffic_options,
                index=get_default(traffic_options, "Low")
            )

            weather_options = get_options("Weather_conditions")
            weather = st.selectbox(
                "Weather conditions",
                weather_options,
                index=get_default(weather_options, "Sunny")
            )

        st.divider()

        # -----------------------------
        # DELIVERY PARTNER
        # -----------------------------

        st.markdown("#### 🛵 Delivery Partner")

        c3, c4 = st.columns(2)

        with c3:
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

        with c4:
            condition_options = get_options("Vehicle_condition")
            vehicle_condition = st.selectbox(
                "Vehicle condition",
                condition_options
            )

            vehicle_options = get_options("Type_of_vehicle")
            vehicle = st.selectbox(
                "Vehicle type",
                vehicle_options
            )

        st.divider()

        # -----------------------------
        # ORDER INFORMATION
        # -----------------------------

        st.markdown("#### 🍔 Order Information")

        c5, c6 = st.columns(2)

        with c5:
            order_options = get_options("Type_of_order")
            order_type = st.selectbox(
                "Order type",
                order_options
            )

            delivery_options = get_options("multiple_deliveries")
            multiple_deliveries = st.selectbox(
                "Multiple deliveries",
                delivery_options
            )

        with c6:
            festival_options = get_options("Festival")
            festival = st.selectbox(
                "Festival",
                festival_options
            )

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Predict Delivery Time",
            use_container_width=True
        )

    # -----------------------------
    # PREDICTION
    # -----------------------------

    if submitted:

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

        try:
            prediction = float(model.predict(sample)[0])

            if not pd.notna(prediction):
                raise ValueError("The model returned an invalid prediction.")

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">ESTIMATED DELIVERY TIME</div>
                <div class="result-value">{prediction:.1f} min</div>
                <div class="result-subtitle">
                    Approximately {round(prediction)} minutes
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "This is a machine learning estimate, not a guaranteed "
                "delivery time."
            )

            with st.expander("View prediction inputs"):
                st.dataframe(sample, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# =========================================================
# 8. MODEL INFORMATION PAGE
# =========================================================

elif page == "About the Model":

    st.markdown("## Model Performance")
    st.write(
        "The Random Forest Regressor was trained to estimate "
        "food delivery duration."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", "3.22 min")
    c2.metric("RMSE", "4.09 min")
    c3.metric("R² Score", "0.8107")

    st.divider()

    st.markdown("### What do these metrics mean?")

    st.markdown("""
    - **MAE:** Average absolute prediction error is approximately
      3.22 minutes.
    - **RMSE:** Penalizes larger prediction errors more heavily.
    - **R²:** The model explains approximately 81.07% of the
      variance in the test target values.
    """)

    st.divider()

    st.markdown("### Features used for training")

    features = [
        "distance_km",
        "order_hour",
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Road_traffic_density",
        "Weather_conditions",
        "Type_of_order",
        "Type_of_vehicle",
        "Vehicle_condition",
        "multiple_deliveries",
        "Festival",
        "City"
    ]

    st.write("The model uses 12 input features.")

    for i in range(0, len(features), 2):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"✓ {features[i]}")

        if i + 1 < len(features):
            with c2:
                st.markdown(f"✓ {features[i + 1]}")

    st.divider()

    st.markdown("### Dataset")
    st.metric("Dataset records", f"{len(df):,}")
    st.caption("Source: Project dataset")
