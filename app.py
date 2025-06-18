import streamlit as st
from transformers import pipeline
from PIL import Image

# App title
st.set_page_config(page_title="🖼️ Image Captioning")
st.title("🧠 Image to Text with BLIP")
st.markdown("Upload an image and I'll describe it using AI!")

# Load the BLIP model (cached to avoid reloading)
@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

captioning_pipe = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        results = captioning_pipe(image)
        caption = results[0]['generated_text']
        st.success("Caption generated:")
        st.markdown(f"### 📝 {caption}")
