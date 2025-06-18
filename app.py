import os
import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

# Hugging Face Token (keep this private in production!)
HF_TOKEN = "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"

# Initialize Hugging Face Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

# Streamlit UI
st.set_page_config(page_title="NSFW Image Detection", page_icon="🔍")
st.title("🔍 NSFW Image Detection")
st.write("Upload an image to detect whether it is safe-for-work (SFW) or not (NSFW).")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp path
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Run model
    with st.spinner("Classifying..."):
        try:
            results = client.image_classification(
                image=temp_path,
                model="Falconsai/nsfw_image_detection"
            )

            st.subheader("Results:")
            if isinstance(results, list):
                for res in results:
                    label = res.get("label", "Unknown")
                    score = res.get("score", 0)
                    st.write(f"**{label}**: {round(score * 100, 2)}%")
            else:
                st.warning("Unexpected result format.")

        except Exception as e:
            st.error(f"Error during classification: {e}")

    # Remove temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
