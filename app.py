from model import gdRegression
import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD FILES ---------------- #
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")
st.markdown("Estimate property prices using machine learning.")

# ---------------- FEATURE LABELS ---------------- #
feature_labels = {
    "GrLivArea": "Living Area (sq ft)",
    "BedroomAbvGr": "Bedrooms (BHK)",
    "GarageCars": "Garage Capacity",
    "OverallQual": "Overall Quality (1-10)",
    "YearBuilt": "Year Built",
    "TotalBsmtSF": "Basement Area (sq ft)",
    "FullBath": "Full Bathrooms",
    "TotRmsAbvGrd": "Total Rooms",
}

# ---------------- IMPORTANT FEATURES ---------------- #
important_features = [
    "GrLivArea", "OverallQual", "GarageCars", "BedroomAbvGr"
]

# ---------------- SIDEBAR INPUT ---------------- #
st.sidebar.header("🏡 Basic Details")

user_data = {}

# Basic Inputs
for feature in important_features:
    label = feature_labels.get(feature, feature)

    if feature == "OverallQual":
        val = st.sidebar.slider(label, 1, 10, 5)
    elif feature == "GarageCars":
        val = st.sidebar.slider(label, 0, 5, 1)
    elif feature == "BedroomAbvGr":
        val = st.sidebar.slider(label, 1, 6, 3)
    else:
        val = st.sidebar.number_input(label, value=1500.0)

    user_data[feature] = val

# ---------------- ADVANCED FEATURES ---------------- #
st.sidebar.markdown("### ⚙️ Advanced Features")

with st.sidebar.expander("Show Advanced Options"):
    for feature in features:
        if feature not in important_features:
            label = feature_labels.get(feature, feature)
            val = st.sidebar.number_input(label, value=0.0)
            user_data[feature] = val

# ---------------- PREPARE INPUT ---------------- #
input_list = [user_data.get(f, 0) for f in features]
input_array = np.array(input_list).reshape(1, -1)

# ---------------- PREDICTION ---------------- #
if st.button("Predict Price 💰"):

    if np.all(input_array == 0):
        st.error("⚠️ Please enter valid property details")
    else:
        try:
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)

            price = prediction[0]*94

            # ---------------- OUTPUT ---------------- #
            st.markdown(f"""
            ## 💰 Estimated Price  
            ### ₹ {price:,.0f}
            """)

            # ---------------- INPUT SUMMARY ---------------- #
            st.markdown("### 📋 Your Inputs")
            for f, v in user_data.items():
                label = feature_labels.get(f, f)
                st.write(f"**{label}**: {v}")

            # ---------------- WARNING ---------------- #
            st.warning("⚠️ This is an estimated price. Actual market value may vary.")

        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ---------------- INSIGHTS ---------------- #
st.markdown("### 📊 Model Insights")
st.write("Top influencing factors include:")
st.write("- Living Area")
st.write("- Overall Quality")
st.write("- Garage Capacity")
