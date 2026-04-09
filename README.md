# 🏡 Boston House Price Prediction App

An interactive **Machine Learning web application** built using **Streamlit** that predicts house prices based on various features of a property. This project demonstrates end-to-end ML workflow including data preprocessing, model building, and deployment with a modern UI.

---

## 🚀 Live Demo

👉 *(Add your Streamlit Cloud link here after deployment)*

---

## 📌 Project Overview

This application predicts the **Median Value of Owner-Occupied Homes (MEDV)** using multiple housing features such as crime rate, number of rooms, tax rate, etc.

The app provides:

* 🎯 Real-time predictions
* 📊 Visual insights
* 🎨 Attractive UI with background and animations

---

## 🧠 Machine Learning Details

* **Model Used:** Regression Model (e.g., Linear Regression / Random Forest)
* **Target Variable:** MEDV (House Price)
* **Feature Scaling:** StandardScaler
* **Model Serialization:** Joblib

---

## 📊 Input Features

| Feature | Description                                   |
| ------- | --------------------------------------------- |
| CRIM    | Crime rate per capita                         |
| ZN      | Residential land zoned (%)                    |
| INDUS   | Non-retail business acres                     |
| CHAS    | Charles River dummy variable (1 = near river) |
| NOX     | Nitric oxide concentration                    |
| RM      | Average number of rooms                       |
| AGE     | % of owner-occupied units built before 1940   |
| DIS     | Distance to employment centers                |
| RAD     | Accessibility to highways                     |
| TAX     | Property tax rate                             |
| PTRATIO | Student-teacher ratio                         |
| B       | Population index                              |
| LSTAT   | Lower status population (%)                   |

---

## 🎯 Output

* **MEDV (Median House Price)**
  Displayed with:
* 💰 Highlighted prediction card
* 📊 Visualization chart

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Libraries:**

  * NumPy
  * Matplotlib
  * Joblib
* **ML:** Scikit-learn

---

## 🎨 UI Features

* 🌄 Background image with overlay
* 💎 Glassmorphism cards
* 🎞️ Smooth animations
* 📊 Graphical output
* 🎯 Sidebar-based inputs

---

## 📂 Project Structure

```
📦 Boston-House-Price-Prediction
│
├── app.py                 # Streamlit application
├── model1.joblib          # Trained ML model
├── scaler1.joblib         # Scaler for preprocessing
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/boston-house-price-predictor.git
cd boston-house-price-predictor
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📦 Requirements

Create a `requirements.txt` file:

```
streamlit
numpy
matplotlib
scikit-learn
joblib
```

---

## 📸 Screenshots

*(Add screenshots here after running the app)*

Example:

* Home Page UI
* Prediction Output
* Feature Importance Graph

---

## 🚀 Future Improvements

* 🔥 Add SHAP for model explainability
* 📈 Interactive Plotly visualizations
* 🌐 Deploy on Streamlit Cloud
* 🧠 Try advanced models (XGBoost, Gradient Boosting)
* 📱 Make mobile-responsive UI

---

## 🙌 Acknowledgements

* Boston Housing Dataset
* Scikit-learn Documentation
* Streamlit Community

---

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!

---
