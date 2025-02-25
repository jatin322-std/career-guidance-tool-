import streamlit as st
import numpy as np
import joblib
import time

# Load trained model, scaler, and encoder
model = joblib.load("career_model.pkl")
scaler = joblib.load("score_scaler.pkl")
encoder = joblib.load("career_encoder.pkl")

# Set Streamlit page configuration
st.set_page_config(page_title="AI Career Guidance", page_icon="🚀", layout="wide")

# Custom CSS for improved UI
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            background: linear-gradient(to right, #FF8C00, #FF4500);
            padding: 20px;
            border-radius: 12px;
            font-size: 30px;
            font-weight: bold;
            color: white;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        .result-box {
            text-align: center;
            background-color: #32CD32;
            color: white;
            padding: 20px;
            border-radius: 12px;
            font-size: 26px;
            font-weight: bold;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
            animation: fadeIn 2s;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .stButton>button {
            background-color: #FF4500;
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FF6347;
        }
    </style>
    <div class='main-header'>🚀 AI Career Guidance Tool 🎯</div>
""", unsafe_allow_html=True)

st.write("### Enter your aptitude scores to receive personalized career recommendations.")

# Sidebar with branding and instructions
st.sidebar.image("DALL·E 2025-02-25 12.24.20 - A modern and professional logo for a career guidance tool. The design should incorporate a sleek compass symbol representing direction and guidance, c.webp", width=150)
st.sidebar.markdown("""
## 📌 Instructions:
- Adjust the sliders to reflect your skills.
- Click 'Predict Career' to get recommendations.
- Explore your potential career path!
- Receive a detailed career roadmap based on your scores.
""")

# Input fields with columns and icons
col1, col2, col3 = st.columns(3)
with col1:
    analytical = st.slider("🧠 Analytical Thinking", 0, 100, 50)
    logical = st.slider("🔍 Logical Reasoning", 0, 100, 50)
with col2:
    problem_solving = st.slider("🛠 Problem-Solving", 0, 100, 50)
    math = st.slider("📊 Mathematical Aptitude", 0, 100, 50)
with col3:
    verbal = st.slider("🗣 Verbal Ability", 0, 100, 50)
    market_demand = st.slider("📈 Market Demand Score", 0, 100, 50)

# Predict Career button with animation
if st.button("🎯 Predict My Career", help="Click to generate your career recommendation"):
    with st.spinner("🔍 Analyzing your strengths..."):
        time.sleep(2)  # Simulate processing time
        
        # Prepare input data with 6 features
        input_data = np.array([[analytical, logical, problem_solving, math, verbal, market_demand]])
        input_data = scaler.transform(input_data)
        
        # Predict career
        career_encoded = model.predict(input_data)
        career = encoder.inverse_transform(career_encoded)[0]
        
        # Display result with enhanced styling and animation
        st.markdown(f"""
            <div class='result-box'>
                🎉 Your Recommended Career: <strong>{career}</strong>
                <p>📌 Get started on this career path with recommended courses and skills.</p>
            </div>
        """, unsafe_allow_html=True)

        # Additional career resources
        st.markdown("## 📚 Suggested Resources")
        st.write("🔗 [Coursera Career Courses](https://www.coursera.org)")
        st.write("🔗 [LinkedIn Learning](https://www.linkedin.com/learning)")
        st.write("🔗 [Udemy Skill Development](https://www.udemy.com)")