# Project Logistisc Regression Titanic Survival Prediction
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# 2. Load Models
@st.cache_resource
def load_models():
    model = pickle.load(open("titanic_model.pkl", "rb"))
    return model

try:
    model = load_models()
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'mode_titanic.pkl' exists.")
    st.stop()

# 3. Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stButton>button:hover {
    background-color: green;
    color: white;}
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Header
st.header("🚢 Titanic Survival Prediction",divider="rainbow")
st.subheader("Predict if a passenger would survive the Titanic")
st.write("Please enter the passenger details below to predict survival.")

# 5. Input Form
with st.container():
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        Pclass = st.number_input("Pclass (Passenger Class)", min_value=1, max_value=3, value=1)
        Sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        Age = st.number_input("Age", min_value=0, max_value=100, value=30)
        SibSp = st.number_input("SibSp (Siblings/Spouse)", min_value=0, max_value=10, value=0)

    with col2:
        Parch = st.number_input("Parch (Parents/Children)", min_value=0, max_value=10, value=0)
        Fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
        Embarked = st.selectbox("Embarked", options=[0, 1, 2], format_func=lambda x: ["Cherbourg", "Queenstown", "Southampton"][x] if x < 3 else "Unknown")

    st.markdown("###")
    predict_btn = st.button("Predict Survival")

# 6. Prediction Logic
if predict_btn:
    input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = model.predict(input_data)

    st.markdown("---")
    if prediction[0] == 1:
        st.success("### ✅ Result: Survived")
        st.info("Based on the data provided, the passenger likely survived.")
    else:
        st.error("### 🚩 Result: Not Survived")
        st.warning("Based on the data provided, the passenger likely did not survive.")

# 7. Footer
st.markdown("---")

st.header("Yanaguntikar Meesal 😊",divider="rainbow")

st.markdown("**Connect with me:**")

col1, col2, col3 = st.columns(3)

with col1:
    st.link_button("📧 Email", "mailto:yanaguntikarm@gmail.com")

with col2:
    st.link_button("📱 WhatsApp", "https://wa.me/918618219261")

with col3:
    st.link_button("🐙 GitHub", "https://github.com/yanaguntikarmeesal")

