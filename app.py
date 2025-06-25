import pickle
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd

# Load the model
with open("diabetes_pred.pkl", 'rb') as f:
    model = pickle.load(f)

# ---- CSS Styling ----
st.markdown("""
    <style>
    /* Animated dark gradient background */
.stApp {
    animation: gradientShift 20s ease infinite;
    background: linear-gradient(-45deg, #2e2a3b, #3b4c5e, #2f3e46, #2a2f34);
    background-size: 400% 400%;
    padding: 2rem;
    color: #f0f0f0;
    font-family: 'Segoe UI', sans-serif;
    text-shadow: none;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}


    /* Centered form card */
    .main {
        background-color: rgba(30, 30, 30, 0.85);
        padding: 2rem;
        border-radius: 15px;
        max-width: 800px;
        margin: auto;
        box-shadow: 0 0 25px rgba(0, 0, 0, 0.6);
        color: #f0f0f0;
    }

    h1, h3 {
        text-align: center;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    .stButton>button {
        background-color: #26a69a;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75em 1.5em;
        margin-top: 1em;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #1e857d;
        transform: scale(1.05);
    }

    .stNumberInput label {
        color: #ffffff !important;
    }

    hr {
        border: 0.5px solid #666;
    }

    /* Hide top black header bar only */
    header {
        visibility: hidden;
    }

    .css-18ni7ap.e8zbici2 {
        display: none; /* hides hamburger menu */
    }
    </style>
""", unsafe_allow_html=True)

# ---- App Layout ----
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.markdown("## 🩺 Diabetes Prediction App")
    st.markdown("Fill in your medical details to get a prediction:")

    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("👶 Pregnancies", 0, 20, 1)
        Glucose = st.number_input("🍬 Glucose Level", 50, 320, 100)
        BloodPressure = st.number_input("💓 Blood Pressure", 0, 150, 70)
        SkinThickness = st.number_input("🧪 Skin Thickness", 0, 100, 20)

    with col2:
        Insulin = st.number_input("💉 Insulin Level", 0, 900, 80)
        BMI = st.number_input("⚖️ BMI", 10.0, 70.0, 25.0)
        DiabetesPedigreeFunction = st.number_input("🧬 Diabetes Pedigree", 0.0, 3.0, 0.3)
        Age = st.number_input("🎂 Age", 5, 150, 25)

    if st.button("🔍 Predict"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])

        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0][1]  # Confidence score for diabetic

        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"⚠️ You are likely to have diabetes.\n\n**Accuracy:** {proba:.2%}")
        else:
            st.success(f"✅ You are not likely to have diabetes.\n\n**Accuracy:** {1 - proba:.2%}")

        # 📊 Radar Chart for Input Overview
        

        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        values = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age]

        df = pd.DataFrame(dict(Metric=features, Value=values))

        fig = px.line_polar(df, r='Value', theta='Metric', line_close=True,
                            title="🧭 Your Health Metrics Overview",
                            template='plotly_dark', color_discrete_sequence=["#26a69a"])

        st.plotly_chart(fig)


    st.markdown("""
        <hr>
        <div style='text-align: center; color: #aaa; font-size: 13px'>
            Created by Ishika 
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
