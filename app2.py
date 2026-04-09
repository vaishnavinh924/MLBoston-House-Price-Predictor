import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

# Page config
st.set_page_config(page_title="🏡 House Price Predictor", layout="wide")

# Load model & scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏡 Boston House Price Predictor</h1>", unsafe_allow_html=True)
st.write("### Enter the property details below to predict house price (MEDV)")

# Sidebar for inputs
st.sidebar.header("🔧 Input Features")

def user_input():
    CRIM = st.sidebar.slider("Crime Rate (CRIM)", 0.0, 100.0, 0.1)
    ZN = st.sidebar.slider("Residential Land (ZN)", 0.0, 100.0, 12.5)
    INDUS = st.sidebar.slider("Industrial Area (INDUS)", 0.0, 30.0, 7.0)
    CHAS = st.sidebar.selectbox("Near River (CHAS)", [0, 1])
    NOX = st.sidebar.slider("Pollution (NOX)", 0.3, 1.0, 0.5)
    RM = st.sidebar.slider("Rooms (RM)", 3.0, 9.0, 6.0)
    AGE = st.sidebar.slider("Old Houses (%)", 0.0, 100.0, 50.0)
    DIS = st.sidebar.slider("Distance to Employment (DIS)", 1.0, 15.0, 4.0)
    RAD = st.sidebar.slider("Accessibility (RAD)", 1, 24, 5)
    TAX = st.sidebar.slider("Property Tax (TAX)", 100, 800, 300)
    PTRATIO = st.sidebar.slider("Pupil-Teacher Ratio", 10.0, 25.0, 15.0)
    B = st.sidebar.slider("Black Population Index (B)", 0.0, 400.0, 350.0)
    LSTAT = st.sidebar.slider("Lower Status (%)", 1.0, 40.0, 10.0)

    features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    return features

input_data = user_input()

# Layout columns
col1, col2 = st.columns([1, 1])

# Prediction button
if st.button("🔍 Predict Price"):
    
    with st.spinner("⏳ Predicting..."):
        time.sleep(1)

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)[0]

    # Success animation
    st.success("✅ Prediction Complete!")

    # Display result
    st.markdown(f"<h2 style='text-align:center; color:#FF5733;'>💰 Predicted MEDV: {prediction:.2f}</h2>", unsafe_allow_html=True)

    # Gauge-like visualization
    fig, ax = plt.subplots()
    ax.barh(["Price"], [prediction])
    ax.set_xlim(0, 50)
    ax.set_title("Predicted House Price")
    st.pyplot(fig)

# Feature Importance (if available)
if hasattr(model, "feature_importances_"):
    st.subheader("📊 Feature Importance")

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    importance = model.feature_importances_

    fig2, ax2 = plt.subplots()
    ax2.barh(feature_names, importance)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("✨ Built with Streamlit | ML Model Integration")

