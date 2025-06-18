import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Image Captioning with BLIP")

st.title("Image to Text Captioning")
st.write("Upload an image and the model will describe it.")

@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

pipe = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        result = pipe(image)
        caption = result[0]["generated_text"]
        st.success("Caption generated!")
        st.markdown(f"### Caption: {caption}")
