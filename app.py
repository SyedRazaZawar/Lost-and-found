import streamlit as st
import google.generativeai as genai

# Title
st.title("🧠 Gemini Recipe Bot")

# Configure Gemini API
genai.configure(api_key="YOUR_API_KEY_HERE")

# Input from user
user_input = st.text_input("🍛 Enter a food name:")

if user_input:
    with st.spinner("Generating recipe..."):
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Write a complete recipe for {user_input}")
        st.subheader("📜 Recipe")
        st.write(response.text)
