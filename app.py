import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# Set the page title
st.set_page_config(page_title="🖼️ Image Captioning with BLIP", layout="centered")

# App title
st.title("🧠 Image to Text using BLIP")
st.markdown("Upload an image, and I'll describe it using the **BLIP** model (`Salesforce/blip-image-captioning-large`).")

# Load model (only once)
@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

pipe = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Display image and generate caption
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        result = pipe(image)
        caption = result[0]['generated_text']
        st.success("Caption generated!")
        st.markdown(f"### 📝 Caption: `{caption}`")
