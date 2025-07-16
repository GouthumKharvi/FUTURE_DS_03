
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# Load model and vectorizer
model = joblib.load("sentiment_rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit page settings
st.set_page_config(
    page_title="🎓 Student Sentiment Analyzer",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="auto"
)

# Sidebar Info
with st.sidebar:
    st.title("📘 About")
    st.markdown("""
    This app analyzes **student feedback** and classifies the sentiment as:
    
    - 🟢 Positive  
    - 🟡 Neutral  
    - 🔴 Negative

    Built using **Streamlit**, **scikit-learn**, and **NLP (TF-IDF + Textblob + Logistic_Regression + RandomForest)**.
    """)

# Main Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 Student Feedback Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)
st.write("")

# Input Box
user_input = st.text_area("✍️ Enter Student Comment Below:")

# Predict Button
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment before predicting.")
    else:
        input_vector = vectorizer.transform([user_input])
        pred = model.predict(input_vector)[0]

        sentiment_map = {0: "🔴 Negative", 1: "🟡 Neutral", 2: "🟢 Positive"}
        sentiment_label = sentiment_map.get(pred, str(pred))

        # Typing Animation
        with st.spinner("Analyzing..."):
            time.sleep(1)

        result_placeholder = st.empty()
        animated_text = f"Prediction: {sentiment_label}"
        for i in range(len(animated_text)):
            time.sleep(0.05)
            result_placeholder.markdown(
                f"<h2 style='text-align: center; color: #2E8B57;'>{animated_text[:i+1]}</h2>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.success("✅ Sentiment analysis complete.")

# Footer
st.markdown(
    "<hr><div style='text-align: center;'>Made with ❤️ by Gouthum • Future Interns Task 3</div>",
    unsafe_allow_html=True
)
