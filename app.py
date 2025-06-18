import os
import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

# Set your Hugging Face API token here directly or from environment
HF_TOKEN = os.getenv("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")

# Initialize Hugging Face inference client
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

# Streamlit app UI
st.set_page_config(page_title="NSFW Image Detection", page_icon="🖼️")
st.title("🧠 NSFW Image Detection using Hugging Face")
st.write("Upload an image to classify whether it's safe-for-work (SFW) or not (NSFW).")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporarily to pass the path to the model
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Run image classification
    with st.spinner("Analyzing image..."):
        results = client.image_classification(
            image="temp_image.jpg",
            model="Falconsai/nsfw_image_detection"
        )

    # Display results
    st.subheader("Classification Results:")
    for res in results:
        st.write(f"**{res['label']}**: {round(res['score'] * 100, 2)}%")
