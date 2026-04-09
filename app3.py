# import streamlit as st
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import time

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(page_title="🏡 House Price Predictor", layout="wide")

# # ---------------- LOAD MODEL ----------------
# model = joblib.load("model.joblib")
# scaler = joblib.load("scaler.joblib")

# # ---------------- CUSTOM CSS ----------------
# page_bg = """
# <style>
# [data-testid="stAppViewContainer"] {
#     background-image: url("https://images.unsplash.com/photo-1568605114967-8130f3a36994");
#     background-size: cover;
#     background-position: center;
#     background-repeat: no-repeat;
# }

# .main {
#     background: rgba(0,0,0,0.6);
#     padding: 20px;
#     border-radius: 15px;
# }

# h1, h2, h3, h4 {
#     color: #ffffff;
# }

# .stButton>button {
#     background: linear-gradient(90deg, #00C9FF, #92FE9D);
#     color: black;
#     font-size: 18px;
#     border-radius: 10px;
#     height: 3em;
#     width: 100%;
# }

# .stSidebar {
#     background: rgba(255,255,255,0.1);
# }

# .block-container {
#     padding-top: 2rem;
# }
# </style>
# """
# st.markdown(page_bg, unsafe_allow_html=True)

# # ---------------- TITLE ----------------
# st.markdown("<h1 style='text-align:center;'>🏡 Boston House Price Predictor</h1>", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align:center;'>AI-powered Real Estate Price Estimator</h4>", unsafe_allow_html=True)

# # ---------------- SIDEBAR ----------------
# st.sidebar.header("🔧 Property Features")

# def user_input():
#     CRIM = st.sidebar.slider("Crime Rate", 0.0, 100.0, 0.1)
#     ZN = st.sidebar.slider("Residential Land (%)", 0.0, 100.0, 12.5)
#     INDUS = st.sidebar.slider("Industrial Area", 0.0, 30.0, 7.0)
#     CHAS = st.sidebar.selectbox("Near River", [0, 1])
#     NOX = st.sidebar.slider("Pollution Level", 0.3, 1.0, 0.5)
#     RM = st.sidebar.slider("Rooms", 3.0, 9.0, 6.0)
#     AGE = st.sidebar.slider("Old Houses (%)", 0.0, 100.0, 50.0)
#     DIS = st.sidebar.slider("Distance to Jobs", 1.0, 15.0, 4.0)
#     RAD = st.sidebar.slider("Accessibility", 1, 24, 5)
#     TAX = st.sidebar.slider("Tax Rate", 100, 800, 300)
#     PTRATIO = st.sidebar.slider("Student-Teacher Ratio", 10.0, 25.0, 15.0)
#     B = st.sidebar.slider("Population Index", 0.0, 400.0, 350.0)
#     LSTAT = st.sidebar.slider("Lower Status %", 1.0, 40.0, 10.0)

#     return np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
#                       DIS, RAD, TAX, PTRATIO, B, LSTAT]])

# input_data = user_input()

# # ---------------- MAIN LAYOUT ----------------
# col1, col2 = st.columns([1, 1])

# # ---------------- PREDICTION ----------------
# if st.button("🚀 Predict House Price"):
    
#     progress = st.progress(0)
#     for i in range(100):
#         time.sleep(0.01)
#         progress.progress(i + 1)

#     scaled_data = scaler.transform(input_data)
#     prediction = model.predict(scaled_data)[0]

#     st.success("✅ Prediction Completed!")

#     # Result Card
#     st.markdown(f"""
#     <div style="
#         background: rgba(255,255,255,0.1);
#         padding: 30px;
#         border-radius: 15px;
#         text-align: center;
#         backdrop-filter: blur(10px);
#     ">
#         <h2 style="color:#00FFCC;">💰 Predicted Price</h2>
#         <h1 style="color:white;">{prediction:.2f} (MEDV)</h1>
#     </div>
#     """, unsafe_allow_html=True)

#     # ---------------- CHART ----------------
#     fig, ax = plt.subplots()
#     ax.barh(["Price"], [prediction])
#     ax.set_xlim(0, 50)
#     ax.set_title("Predicted House Price")
#     st.pyplot(fig)

# # ---------------- FEATURE IMPORTANCE ----------------
# if hasattr(model, "feature_importances_"):
#     st.markdown("## 📊 Feature Importance")

#     feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
#                      'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

#     importance = model.feature_importances_

#     fig2, ax2 = plt.subplots()
#     ax2.barh(feature_names, importance)
#     ax2.set_title("Feature Importance")
#     st.pyplot(fig2)

# # ---------------- FOOTER ----------------
# st.markdown("""
# <hr>
# <p style='text-align:center; color:white;'>
# ✨ Built with Streamlit | Machine Learning Project
# </p>
# """, unsafe_allow_html=True)

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🏡 Boston House Price Predictor",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# ---------------- CUSTOM CSS ----------------
page_bg = """
<style>

/* Background Image + Dark Overlay */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        rgba(0, 0, 0, 0.65),
        rgba(0, 0, 0, 0.65)
    ),
    url("https://images.unsplash.com/photo-1568605114967-8130f3a36994");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Main container */
.main {
    background: rgba(0, 0, 0, 0.55);
    padding: 20px;
    border-radius: 15px;
}

/* Titles */
h1 {
    color: white;
    text-align: center;
    font-weight: 700;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.9);
}

h4 {
    color: #e0e0e0;
    text-align: center;
    text-shadow: 1px 1px 6px rgba(0,0,0,0.9);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.75);
}

/* Labels */
label {
    color: white !important;
    font-weight: 500;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: black;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
}

/* Result Card */
.result-card {
    background: rgba(255, 255, 255, 0.15);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
    margin-top: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: white;
    margin-top: 30px;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🏡 Boston House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4>AI-powered Real Estate Price Estimator</h4>", unsafe_allow_html=True)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🔧 Property Features")

def user_input():
    CRIM = st.sidebar.slider("Crime Rate", 0.0, 100.0, 0.1)
    ZN = st.sidebar.slider("Residential Land (%)", 0.0, 100.0, 12.5)
    INDUS = st.sidebar.slider("Industrial Area", 0.0, 30.0, 7.0)
    CHAS = st.sidebar.selectbox("Near River (1=Yes, 0=No)", [0, 1])
    NOX = st.sidebar.slider("Pollution Level", 0.3, 1.0, 0.5)
    RM = st.sidebar.slider("Average Rooms", 3.0, 9.0, 6.0)
    AGE = st.sidebar.slider("Old Houses (%)", 0.0, 100.0, 50.0)
    DIS = st.sidebar.slider("Distance to Jobs", 1.0, 15.0, 4.0)
    RAD = st.sidebar.slider("Accessibility Index", 1, 24, 5)
    TAX = st.sidebar.slider("Tax Rate", 100, 800, 300)
    PTRATIO = st.sidebar.slider("Student-Teacher Ratio", 10.0, 25.0, 15.0)
    B = st.sidebar.slider("Population Index", 0.0, 400.0, 350.0)
    LSTAT = st.sidebar.slider("Lower Status (%)", 1.0, 40.0, 10.0)

    return np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM,
                      AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

input_data = user_input()

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict House Price"):

    # Animation
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)[0]

    st.success("✅ Prediction Completed!")

    # Result Card
    st.markdown(f"""
    <div class="result-card">
        <h2 style="color:#00FFCC;">💰 Predicted Price</h2>
        <h1>{prediction:.2f} (MEDV)</h1>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- CHART ----------------
    st.subheader("📊 Prediction Visualization")
    fig, ax = plt.subplots()
    ax.barh(["Predicted Price"], [prediction])
    ax.set_xlim(0, 50)
    ax.set_title("House Price")
    st.pyplot(fig)

# ---------------- FEATURE IMPORTANCE ----------------
if hasattr(model, "feature_importances_"):
    st.subheader("📊 Feature Importance")

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                     'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    importance = model.feature_importances_

    fig2, ax2 = plt.subplots()
    ax2.barh(feature_names, importance)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.markdown("<div class='footer'>✨ Built with Streamlit | Machine Learning Project</div>", unsafe_allow_html=True)