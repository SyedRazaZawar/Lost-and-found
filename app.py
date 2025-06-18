import os
import streamlit as st
import google.generativeai as genai
import PIL.Image
from dotenv import find_dotenv, load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env if present
load_dotenv(find_dotenv(), override=True)

# App Title
st.title("👨‍🍳 I'm Senior Chef - Upload Food Image to Get Recipe")

# Set your API Key (hardcoded for security-free demos)
GOOGLE_API_KEY = "AIzaSyD7YOrZrkH4SBkphu50VMJIU2780C7eUQA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Generation settings
generation_config = {"temperature": 0.9}

# Upload image
uploaded_file = st.file_uploader("📤 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Basic image size check
    if image.size[0] < 100 or image.size[1] < 100:
        st.warning("⚠️ Image too small or unclear. Please upload a better food image.")
    else:
        with st.spinner("🔎 Analyzing image using Gemini..."):
            model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
            response = model.generate_content([image])

        if not response.text:
            st.error("❌ Couldn't describe the image.")
        else:
            st.subheader("📝 AI Description")
            st.write(response.text)

            # Step 2: Extract dish name & ingredients
            prompt = f"Extract the dish name and main ingredients from this description: {response.text}"
            with st.spinner("🔍 Extracting dish info..."):
                extraction = model.generate_content(prompt)

            st.subheader("🍽️ Dish Information")
            st.write(extraction.text)

            # Step 3: Use LangChain to create recipe
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)
            messages = [
                SystemMessage(content="You are a professional chef. Based on the dish and ingredients, write a complete detailed recipe. Include cooking steps and country of origin."),
                HumanMessage(content=extraction.text)
            ]

            with st.spinner("👨‍🍳 Generating recipe..."):
                recipe = llm.invoke(messages)

            st.subheader("📜 Full Recipe")
            st.write(recipe.content)
