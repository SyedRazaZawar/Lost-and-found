import os
import streamlit as st
import google.generativeai as genai
import PIL.Image
from dotenv import find_dotenv, load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file (optional, for other setups)
load_dotenv(find_dotenv(), override=True)

# Streamlit app title
st.title("👨‍🍳 I'm Senior Chef - Upload Food Image to Get Recipe")

# Set the default API key directly (not user-editable)
default_api_key = "AIzaSyD7YOrZrkH4SBkphu50VMJIU2780C7eUQA"
os.environ["GOOGLE_API_KEY"] = default_api_key
genai.configure(api_key=default_api_key)

# Configure generation settings
generation_config = {'temperature': 0.9}

# File uploader for food image
uploaded_file = st.file_uploader("📤 Upload an image of food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and show the image
    img = PIL.Image.open(uploaded_file)
    st.image(img, caption="🍽️ Uploaded Food Image", use_column_width=True)

    # Simple heuristic to ensure it's a reasonably sized image
    if img.size[0] < 100 or img.size[1] < 100:
        st.warning("⚠️ This image may not contain food. Please upload a clearer food image.")
    else:
        # Generate content using Gemini from image
        with st.spinner("🧠 Analyzing the image..."):
            model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
            response_from_image = model.generate_content([img])

        if not response_from_image.text:
            st.error("❌ Couldn't generate description from image.")
        else:
            # Show raw AI description
            st.subheader("📝 AI Description of Image")
            st.write(response_from_image.text)

            # Extract dish name and ingredients
            prompt = f"Extract the dish name and main ingredients from this description: {response_from_image.text}"
            with st.spinner("🔍 Extracting key details..."):
                extraction = model.generate_content(prompt)

            st.subheader("🍴 Dish Info")
            st.write(extraction.text)

            # Generate recipe using LangChain interface
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)
            messages = [
                SystemMessage(content="You are a professional chef. Given the dish name and ingredients, generate a complete detailed recipe with cooking steps and mention the country of origin."),
                HumanMessage(content=extraction.text)
            ]

            with st.spinner("👨‍🍳 Generating full recipe..."):
                recipe = llm.invoke(messages)

            # Display final recipe
            st.subheader("📜 Detailed Recipe")
            st.write(recipe.content)
