import streamlit as st

# Create chatbot selection in the left sidebar
st.sidebar.title("Chatbot Functionality")
chatbot_functionality = st.sidebar.selectbox(
    "Choose Chatbot Functionality",
    ["My Chatbot", "Image Editing", "Video Editing" , "Text to Image" , "Text to Audio" , "Senior Chef for Recipe" , "PDF and Wikipedia Summarization" , "Youtube Video summarizer" , "Text Summarization" , "Audio to text" , "Stock Prices Analysis App" , "Image to Text" , "Text to Music", ]
)








###############################################################################################################











# Define chatbot functionalities
if chatbot_functionality == "My Chatbot":
    # Q&A Chatbot code remains unchanged
    st.title("Chatbot Developed by Raza Zawar")
    # Add the rest of the chatbot code here for the Q&A functionality
    # Make sure the functionality is the same, no changes in logic

    # 1                           Ai Chatbot

    import streamlit as st

    import json

    from langchain_core.output_parsers import StrOutputParser

    from langchain_core.prompts import ChatPromptTemplate

    from langchain_google_genai import ChatGoogleGenerativeAI

    from dotenv import find_dotenv, load_dotenv

    import os

    ChatGoogleGenerativeAI.model_rebuild()

# Load environment variables (if needed)

    load_dotenv(find_dotenv(), override=True)


# Predefined Gemini API Key

    API_KEY = "AIzaSyB6XnVrLA1mzEytOvvQL7Fczn3tfDyCyQQ"  


# File to save sessions

    SESSION_FILE = "sessions.json"


# Prompt Template

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a Gemini chatbot. Fully behave like gemini. This program is developed by Syed Muhammad Raza Zawar. He is an AI Engineer and developed many chatbots like messenger chatbot, chat with PDF, chat with website, stock exchange prediction, text summarization , image editing, video editing, and many other chatbots. Syed Muhammad Raza Zawar has completed his graduation in BSCS with 3.62 CGPA from National College of Business Administration & Economics Lahore. He has also completed his Certification in Web Development which is a 6 months duration course from Corvit Systems Lahore. In this Web development certification he has developed many projects on Front End development using HTML,CSS,Javascript,Bootstrap and for database use sql server. HE has also completed wordpress certification from digiskills which is a 3 months duration course.Also he have completed certification of C# Language from Microsoft.And Also he has Completed Certification of Computer networks from Microsoft.In Computer Networks course he have experienced to work on cisco packet tracer where he learn how to connect different hardware devices with one another. You can check his LinkedIn profile through this link https://www.linkedin.com/in/syed-muhammad-raza-zawar-17905b251/ if anyone questions about myself the show LinkedIn profile must."),
            ('user', "Question: {question}")
        ]
    )

    # Singleton instance for the model
    llm_instance = None

    def get_llm(api_key, engine):
        global llm_instance
        if llm_instance is None:
            llm_instance = ChatGoogleGenerativeAI(
                model=engine,
                google_api_key=api_key,
                allow_reuse=True  # Allow reuse to prevent duplicates
            )
        return llm_instance

    def generate_response(question, engine, temperature, max_token):
        try:
            llm = get_llm(API_KEY, engine)
            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser
            answer = chain.invoke({'question': question})
            return answer
        except Exception as e:
            return f"An error occurred: {e}"

    def load_sessions():
        """Load sessions from the JSON file."""
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as file:
                return json.load(file)
        return {}

    def save_sessions(sessions):
        """Save sessions to the JSON file."""
        with open(SESSION_FILE, "w") as file:
            json.dump(sessions, file)

    # Streamlit application

    # Load sessions from JSON file
    if 'sessions' not in st.session_state:
        st.session_state.sessions = load_sessions()
        if not st.session_state.sessions:
            # Create the first session automatically if no sessions exist
            st.session_state.sessions["Session 1"] = []
        st.session_state.current_session = list(st.session_state.sessions.keys())[0] if st.session_state.sessions else None

    # Sidebar for session management
    st.sidebar.title("Session Management")
    if st.sidebar.button("Create New Session"):
        # Automatically generate a session name
        session_count = len(st.session_state.sessions) + 1
        new_session_name = f"Session {session_count}"
        st.session_state.sessions[new_session_name] = []
        st.session_state.current_session = new_session_name
        save_sessions(st.session_state.sessions)  # Save after creating a new session

    # Select Previous Session
    if st.session_state.sessions:
        previous_session = st.sidebar.selectbox("Select Previous Session", list(st.session_state.sessions.keys()), index=list(st.session_state.sessions.keys()).index(st.session_state.current_session))
        if previous_session:
            st.session_state.current_session = previous_session

    # Display previous conversations
    if st.session_state.current_session:
        st.sidebar.write(f"Current Session: **{st.session_state.current_session}**")
        # Show messages from the current session
        for msg in st.session_state.sessions[st.session_state.current_session]:
            st.write(f"**{msg['role']}:** {msg['content']}")

    # Button to delete the current session
    if st.sidebar.button("Delete Current Session"):
        del st.session_state.sessions[st.session_state.current_session]
        st.session_state.current_session = list(st.session_state.sessions.keys())[0] if st.session_state.sessions else None
        save_sessions(st.session_state.sessions)  # Save after deleting a session
        st.success("Current session deleted successfully.")

    # Button to clear all text in the current session
    if st.sidebar.button("Clear Text of this session"):
        st.session_state.sessions[st.session_state.current_session] = []  # Clear messages in current session
        save_sessions(st.session_state.sessions)  # Save after clearing text
        st.success("All text in the current session cleared successfully.")

    # Button to delete all sessions
    if st.sidebar.button("Delete All Sessions"):
        st.session_state.sessions.clear()  # Clear all sessions in session state
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)  # Remove the JSON file
        st.session_state.current_session = None
        st.success("All sessions deleted successfully.")

    # Select Gemini Model
    engine = st.sidebar.selectbox('Select Gemini Model', ['gemini-1.5-flash'])

    # Adjust Temperature and Token Value
    temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)
    max_token = st.sidebar.slider('Max Token', min_value=100, max_value=300, value=150)

    # Main interface for user input
    st.write("Please ask your Question")
    user_input = st.chat_input("Your Prompt:")

    if user_input:
        # Generate and display the response
        response = generate_response(user_input, engine, temperature, max_token)

        # Save the question and response in the current session
        if st.session_state.current_session:
            st.session_state.sessions[st.session_state.current_session].append({"role": "user", "content": user_input})
            st.session_state.sessions[st.session_state.current_session].append({"role": "assistant", "content": response})
            save_sessions(st.session_state.sessions)  # Save after updating session

            

        st.write("**Response:**")
        st.write(response)
















#####################################################################################################











elif chatbot_functionality == "Image Editing":
    # Image Editing chatbot code remains unchanged
    st.title("Image Editing")

    # Add the rest of the chatbot code here for the Image Editing functionality
    # Make sure the functionality is the same, no changes in logic




###############################################################################################


    import streamlit as st
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    import cv2
    import io

    # Function to enhance image with auto-contrast
    def auto_contrast(image):
        image = np.array(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_image)

    # Function to create pencil sketch effect
    def pencil_sketch(image):
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inverted_image = cv2.bitwise_not(gray_image)
        blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
        inverted_blurred = cv2.bitwise_not(blurred_image)
        sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
        return Image.fromarray(sketch)

    # Function to apply vintage filter
    def vintage_filter(image):
        image = np.array(image)
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                  [0.349, 0.686, 0.168],
                                  [0.272, 0.534, 0.131]])
        vintage_image = cv2.transform(image, sepia_filter)
        vintage_image = np.clip(vintage_image, 0, 255)
        return Image.fromarray(vintage_image.astype(np.uint8))

    # Function to convert image to grayscale
    def convert_to_grayscale(image):
        return image.convert("L")

    # Function to adjust brightness
    def adjust_brightness(image, factor):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    # Function to adjust contrast
    def adjust_contrast(image, factor):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    # Function to enhance colors
    def enhance_colors(image, factor):
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    # Function to sharpen the image
    def sharpen_image(image):
        return image.filter(ImageFilter.SHARPEN)

    # Function to apply blur effect
    def blur_image(image):
        return image.filter(ImageFilter.GaussianBlur(radius=5))

    # Function to rotate the image
    def rotate_image(image, angle):
        return image.rotate(angle)

    # Function to flip the image
    def flip_image(image, direction):
        if direction == "Horizontal":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "Vertical":
            return image.transpose(Image.FLIP_TOP_BOTTOM)

    # Function to crop the image
    def crop_image(image, left, upper, right, lower):
        return image.crop((left, upper, right, lower))

    # Function to invert colors
    def invert_colors(image):
        return Image.fromarray(255 - np.array(image))

    # Function to add noise
    def add_noise(image):
        image_array = np.array(image)
        noise = np.random.randint(0, 50, image_array.shape, dtype=np.uint8)
        noisy_image = cv2.add(image_array, noise)
        return Image.fromarray(noisy_image)

    # Function to resize the image
    def resize_image(image, width, height):
        return image.resize((width, height))

    # Function to convert an image to bytes for download
    def image_to_bytes(image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # You can change to 'JPEG' if needed
        img_byte_arr.seek(0)
        return img_byte_arr

    # Streamlit application
    st.write("Enhance your images with auto-contrast, pencil sketch effects, vintage filters, and more!")

    # Uploading an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Initialize image history
    if 'image_history' not in st.session_state:
        st.session_state.image_history = []

    # Delete image history option
    if st.sidebar.button("Clear History"):
        st.session_state.image_history.clear()
        st.success("Image history cleared!")
    

    if uploaded_file is not None:
        # Read and display the original image
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)

        # Sidebar for enhancement options
        st.sidebar.header("Enhancement Options")

        # Auto-contrast
        if st.sidebar.button("Auto Contrast"):
            enhanced_image = auto_contrast(image)
            st.image(enhanced_image, caption='Auto Contrast Image', use_column_width=True)
            st.download_button("Download Auto Contrast Image", data=image_to_bytes(enhanced_image), file_name="auto_contrast_image.png")
            st.session_state.image_history.insert(0, ("Auto Contrast Image", enhanced_image))

        # Pencil Sketch
        if st.sidebar.button("Pencil Sketch"):
            sketch_image = pencil_sketch(image)
            st.image(sketch_image, caption='Pencil Sketch Image', use_column_width=True)
            st.download_button("Download Pencil Sketch Image", data=image_to_bytes(sketch_image), file_name="pencil_sketch_image.png")
            st.session_state.image_history.insert(0, ("Pencil Sketch Image", sketch_image))

        # Vintage Filter
        if st.sidebar.button("Vintage Filter"):
            vintage_image = vintage_filter(image)
            st.image(vintage_image, caption='Vintage Filter Image', use_column_width=True)
            st.download_button("Download Vintage Filter Image", data=image_to_bytes(vintage_image), file_name="vintage_filter_image.png")
            st.session_state.image_history.insert(0, ("Vintage Filter Image", vintage_image))

        # Grayscale Conversion
        if st.sidebar.button("Grayscale"):
            gray_image = convert_to_grayscale(image)
            st.image(gray_image, caption='Grayscale Image', use_column_width=True)
            st.download_button("Download Grayscale Image", data=image_to_bytes(gray_image), file_name="grayscale_image.png")
            st.session_state.image_history.insert(0, ("Grayscale Image", gray_image))

        # Brightness Adjustment
        brightness_factor = st.sidebar.slider("Brightness Factor", 0.0, 3.0, 1.0)
        if st.sidebar.button("Adjust Brightness"):
            bright_image = adjust_brightness(image, brightness_factor)
            st.image(bright_image, caption='Brightened Image', use_column_width=True)
            st.download_button("Download Brightened Image", data=image_to_bytes(bright_image), file_name="brightened_image.png")
            st.session_state.image_history.insert(0, ("Brightened Image", bright_image))

        # Contrast Adjustment
        contrast_factor = st.sidebar.slider("Contrast Factor", 0.0, 3.0, 1.0)
        if st.sidebar.button("Adjust Contrast"):
            contrast_image = adjust_contrast(image, contrast_factor)
            st.image(contrast_image, caption='Contrasted Image', use_column_width=True)
            st.download_button("Download Contrasted Image", data=image_to_bytes(contrast_image), file_name="contrasted_image.png")
            st.session_state.image_history.insert(0, ("Contrasted Image", contrast_image))

        # Color Enhancement
        color_factor = st.sidebar.slider("Color Enhancement Factor", 0.0, 3.0, 1.0)
        if st.sidebar.button("Enhance Colors"):
            color_enhanced_image = enhance_colors(image, color_factor)
            st.image(color_enhanced_image, caption='Color Enhanced Image', use_column_width=True)
            st.download_button("Download Color Enhanced Image", data=image_to_bytes(color_enhanced_image), file_name="color_enhanced_image.png")
            st.session_state.image_history.insert(0, ("Color Enhanced Image", color_enhanced_image))

        # Sharpening
        if st.sidebar.button("Sharpen Image"):
            sharp_image = sharpen_image(image)
            st.image(sharp_image, caption='Sharpened Image', use_column_width=True)
            st.download_button("Download Sharpened Image", data=image_to_bytes(sharp_image), file_name="sharpened_image.png")
            st.session_state.image_history.insert(0, ("Sharpened Image", sharp_image))

        # Blur Effect
        if st.sidebar.button("Blur Image"):
            blurred_image = blur_image(image)
            st.image(blurred_image, caption='Blurred Image', use_column_width=True)
            st.download_button("Download Blurred Image", data=image_to_bytes(blurred_image), file_name="blurred_image.png")
            st.session_state.image_history.insert(0, ("Blurred Image", blurred_image))

        # Rotate Image
        rotation_angle = st.sidebar.slider("Rotation Angle", 0, 360, 0)
        if st.sidebar.button("Rotate Image"):
            rotated_image = rotate_image(image, rotation_angle)
            st.image(rotated_image, caption='Rotated Image', use_column_width=True)
            st.download_button("Download Rotated Image", data=image_to_bytes(rotated_image), file_name="rotated_image.png")
            st.session_state.image_history.insert(0, ("Rotated Image", rotated_image))

        # Flip Image
        flip_direction = st.sidebar.selectbox("Flip Direction", ("None", "Horizontal", "Vertical"))
        if st.sidebar.button("Flip Image"):
            if flip_direction != "None":
                flipped_image = flip_image(image, flip_direction)
                st.image(flipped_image, caption='Flipped Image', use_column_width=True)
                st.download_button("Download Flipped Image", data=image_to_bytes(flipped_image), file_name="flipped_image.png")
                st.session_state.image_history.insert(0, ("Flipped Image", flipped_image))

        # Crop Image
        left = st.sidebar.number_input("Left", 0)
        upper = st.sidebar.number_input("Upper", 0)
        right = st.sidebar.number_input("Right", image.width)
        lower = st.sidebar.number_input("Lower", image.height)
        if st.sidebar.button("Crop Image"):
            cropped_image = crop_image(image, left, upper, right, lower)
            st.image(cropped_image, caption='Cropped Image', use_column_width=True)
            st.download_button("Download Cropped Image", data=image_to_bytes(cropped_image), file_name="cropped_image.png")
            st.session_state.image_history.insert(0, ("Cropped Image", cropped_image))

        # Invert Colors
        if st.sidebar.button("Invert Colors"):
            inverted_image = invert_colors(image)
            st.image(inverted_image, caption='Inverted Colors Image', use_column_width=True)
            st.download_button("Download Inverted Image", data=image_to_bytes(inverted_image), file_name="inverted_image.png")
            st.session_state.image_history.insert(0, ("Inverted Colors Image", inverted_image))

        # Add Noise
        if st.sidebar.button("Add Noise"):
            noisy_image = add_noise(image)
            st.image(noisy_image, caption='Noisy Image', use_column_width=True)
            st.download_button("Download Noisy Image", data=image_to_bytes(noisy_image), file_name="noisy_image.png")
            st.session_state.image_history.insert(0, ("Noisy Image", noisy_image))

        # Resize Image
        resize_width = st.sidebar.number_input("Resize Width", 50, 1000, 150)
        resize_height = st.sidebar.number_input("Resize Height", 50, 1000, 150)
        if st.sidebar.button("Resize Image"):
            resized_image = resize_image(image, resize_width, resize_height)
            st.image(resized_image, caption='Resized Image', use_column_width=True)
            st.download_button("Download Resized Image", data=image_to_bytes(resized_image), file_name="resized_image.png")
            st.session_state.image_history.insert(0, ("Resized Image", resized_image))

        # Display Image History
        st.header("Image History")
        if st.session_state.image_history:
            for caption, img in st.session_state.image_history:
                st.image(img, caption=caption, use_column_width=100)  # Medium size of 300 pixels









###############################################################################################################

































elif chatbot_functionality == "Youtube Video summarizer":
    # Senior Chef chatbot code remains unchanged
    

    # Add the rest of the chatbot code here for the Senior Chef functionality
    # Make sure the functionality is the same, no changes in logic








    import streamlit as st
    from pytube import YouTube
    from PIL import Image
    import requests
    from io import BytesIO
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    from youtube_transcript_api.formatters import SRTFormatter
    import time  # To add a delay between retries

    # API URLs and headers for Hugging Face
    API_URL_SUMMARIZATION = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    API_URL_TTS = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}  # Replace with your actual Hugging Face API Key

    # Function to fetch video info and thumbnail
    def fetch_video_info(url):
        try:
            yt = YouTube(url)
            video_id = yt.video_id
            thumbnail_url = yt.thumbnail_url
            return video_id, thumbnail_url
        except Exception as e:
            st.error("Error fetching video info: " + str(e))
            return None, None

    # Function to fetch alternative thumbnails
    def try_alternative_thumbnail(video_id):
        alt_urls = [
            f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",  
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",      
            f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",      
            f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",      
            f"https://img.youtube.com/vi/{video_id}/default.jpg"         
        ]
        for url in alt_urls:
            response = requests.get(url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        return None

    # Function to display the thumbnail
    def display_thumbnail(url, video_id):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Video Thumbnail", use_column_width=True)
            else:
                st.warning(f"Default thumbnail not found. Status code: {response.status_code}. Trying alternatives...")
                alt_image = try_alternative_thumbnail(video_id)
                if alt_image:
                    st.image(alt_image, caption="Alternative Video Thumbnail", use_column_width=True)
                else:
                    st.error("No valid thumbnail found.")
        except Exception as e:
            st.error("Failed to display thumbnail: " + str(e))

    # Function to fetch available caption languages
    def fetch_available_languages(video_id, selected_language_code):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            languages = {transcript.language_code: transcript.language for transcript in transcript_list}
            return languages
        except (NoTranscriptFound, TranscriptsDisabled):
            # Attempt retry for fetching captions if no transcript is found
            captions = retry_until_success(video_id, selected_language_code)
            if captions:
                st.session_state['captions'] = captions
                return captions
            return {}

    # Function to fetch captions
    def fetch_captions(video_id, language_code='en'):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript([language_code])
            formatter = SRTFormatter()
            srt = formatter.format_transcript(transcript.fetch())
            return srt
        except Exception as e:
            return ""  # If no captions available, return an empty string

    # Function to automatically fetch captions until successful
    def retry_until_success(video_id, selected_language_code):
        retry_count = 0
        max_retries = 2000  # Limit the number of retries
        wait_time = 0.00000000000000001  # Wait time between retries in seconds

        while retry_count < max_retries:
            captions = fetch_captions(video_id, selected_language_code)
            if captions:
                return captions
            else:
                st.warning(f"Attempt {retry_count+1}: I'm generating your transcripts it will take time due to large traffic on server. Retrying in {wait_time} seconds...")
                retry_count += 1
                time.sleep(wait_time)  # Wait for a few seconds before retrying

        st.error("Failed to fetch captions after multiple attempts.")
        return ""

    # Function to call Hugging Face Summarization API
    def query_summarization_api(text, min_length, max_length):
        response = requests.post(API_URL_SUMMARIZATION, headers=headers, json={"inputs": text, "parameters": {"min_length": min_length, "max_length": max_length}})
        return response.json()

    # Function to call Hugging Face Text-to-Speech API
    def query_tts_api(text):
        response = requests.post(API_URL_TTS, headers=headers, json={"inputs": text})
        return response.content  # Binary audio data

    # Main function to handle UI and functionality
    def main():
        st.title("YouTube Caption, Summarizer, and TTS")

        # Initialize session state for captions and summary
        if 'captions' not in st.session_state:
            st.session_state['captions'] = ""
        if 'summary' not in st.session_state:
            st.session_state['summary'] = ""
        if 'video_id' not in st.session_state:
            st.session_state['video_id'] = None
        if 'available_languages' not in st.session_state:
            st.session_state['available_languages'] = {}

        # Sidebar for managing summary length
        st.sidebar.header("Summary Settings")
        min_length = st.sidebar.slider("Min Length", 10, 500, 50)
        max_length = st.sidebar.slider("Max Length", 50, 1000, 200)

        st.warning("Please enter the link of the YouTube video which has English transcripts to avoid any error. Because at this time I have learned only English. Thanks for your cooperation.") 

        # Get URL input
        url = st.text_input("Enter YouTube video URL", "")

        # Fetch Video Info Button
        if st.button("Fetch Video Info"):
            if url:
                # Fetch video information and captions
                video_id, thumbnail_url = fetch_video_info(url)
                if video_id:
                    st.session_state['video_id'] = video_id
               
                    # Fetch available languages
                    available_languages = fetch_available_languages(video_id, 'en')
                    if isinstance(available_languages, dict) and available_languages:
                        st.session_state['available_languages'] = available_languages
                        language_options = list(available_languages.values())
                        selected_language = st.selectbox("Select Caption Language", language_options)
                        selected_language_code = list(available_languages.keys())[language_options.index(selected_language)]

                        # Automatically retry fetching transcripts until successful
                        captions = retry_until_success(video_id, selected_language_code)
                        if captions:
                            st.session_state['captions'] = captions
                            st.session_state['summary'] = ""  # Reset summary when new captions are fetched

                        else:
                            st.warning("Click on fetch video info button again")
                        
                    
                    else:
                        st.warning("I'm trying my services best. Thanks for your cooperation !!!")
                else:
                    st.error("Failed to fetch video data. Check the provided URL.")
    
        # Display the thumbnail if the video_id is valid
        if url:
            video_id, thumbnail_url = fetch_video_info(url)
            if video_id:
                st.session_state['video_id'] = video_id
                if thumbnail_url:
                    display_thumbnail(thumbnail_url, video_id)
                else:
                    st.warning("No thumbnail available for this video.")
    
        # Display captions if already fetched
        if st.session_state['captions']:
            st.text_area("Captions", st.session_state['captions'], height=300, key="captions_area_display")

            # Summarize Captions Button
            if st.button("Summarize Captions"):
                with st.spinner('Summarizing...'):
                    output = query_summarization_api(st.session_state['captions'], min_length, max_length)
                    if isinstance(output, list) and len(output) > 0 and 'summary_text' in output[0]:
                        st.session_state['summary'] = output[0]['summary_text']
                    elif 'error' in output:
                        st.error(f"API Error: {output['error']}")
                    else:
                        st.error("Unexpected response format. Please try again later.")
    
        # Display summary if available
        if st.session_state['summary']:
            st.subheader("Summary:")
            st.text_area("Summary", st.session_state['summary'], height=200, key="summary_area")

            # Text-to-Speech Button
            if st.button("Generate Audio for Summary"):
                with st.spinner('Generating audio...'):
                    audio_data = query_tts_api(st.session_state['summary'])
                    if audio_data:
                        st.success("Audio generated successfully!")

                        # Save audio file
                        audio_file_path = "output.wav"
                        with open(audio_file_path, "wb") as f:
                            f.write(audio_data)

                        # Play the audio
                        st.audio(audio_file_path)

                        # Download button for audio
                        st.download_button(label="Download Audio", data=audio_data, file_name="summary_audio.wav", mime="audio/wav")
                    else:
                        st.error("Failed to generate audio.")

    if __name__ == "__main__":
        main()









































##################################################################################################################












##################################################################################################################












    

elif chatbot_functionality == "Image to Text":
    # Image to Text chatbot code remains unchanged
    st.title("Image to Text")
    # Add the rest of the chatbot code here for the Image to Text functionality
    # Make sure the functionality is the same, no changes in logic

    import streamlit as st
    import requests

    # Set the Hugging Face API details
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": "Bearer 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"}

    # Function to send the image file to the API and get the caption
    def query_image(image_data):
        response = requests.post(API_URL, headers=headers, data=image_data)
        return response.json()

    # Streamlit app layout

    # Image upload widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image file is uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Add a button for generating the caption
        if st.button("Generate Caption"):
            # Read the image as binary data
            image_data = uploaded_file.read()

            # Call the Hugging Face API to get the caption
            with st.spinner('Generating caption...'):
                output = query_image(image_data)

            # Display the generated caption
            if 'error' not in output:
                caption = output[0]['generated_text']
                st.success("Generated Caption:")
                st.write(caption)
            else:
                st.error("Please Reload the Page and then give me image again. I will Caption it for you.")


































elif chatbot_functionality == "PDF and Wikipedia Summarization":
    # PDF and Wikipedia Summarization chatbot code remains unchanged
    st.title("PDF and Wikipedia Summarization")

    # Add the rest of the chatbot code here for the PDF and Wikipedia Summarization functionality
    # Make sure the functionality is the same, no changes in logic











##################################################################################################################

#  3                          PDF and Wikipedia Summarization






    import streamlit as st
    from PyPDF2 import PdfReader
    import wikipediaapi
    import requests
    from gtts import gTTS
    import tempfile
    import os

    # Initialize Wikipedia API with custom user-agent
    def create_wikipedia_instance():
        return wikipediaapi.Wikipedia(
            language='en',
            user_agent='RazaZawar/1.0 (https://www.linkedin.com/in/syed-muhammad-raza-zawar-17905b251/; syedrazazawar352@gmail.com)'
        )

    wiki_wiki = create_wikipedia_instance()

    # Function to load PDF and extract text using PdfReader
    def load_pdf(pdf_files):
        text = ""
        for pdf_file in pdf_files:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    # Function to load Wikipedia content
    def load_wikipedia_page(topic):
        page = wiki_wiki.page(topic)
        if page.exists():
            return page.text
        else:
            return "This topic does not exist on Wikipedia."

    # Hugging Face API setup
    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    headers = {"Authorization": "Bearer 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"}

    # Function to query the model
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    # Function to convert text to speech using gTTS
    def text_to_speech(text):
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts.save(temp_audio_file.name)
            return temp_audio_file.name

    # Streamlit App layout

    # Sidebar options
    st.sidebar.header("Options")
    num_pdfs = st.sidebar.slider("Select Number of PDFs to Upload", min_value=1, max_value=5, value=1)
    language_code = st.sidebar.text_input("Enter Language Code (e.g., 'en' for English)", value='en')

    # Summarization length settings
    min_length = st.sidebar.number_input("Minimum Summary Length", min_value=10, max_value=500, value=30)
    max_length = st.sidebar.number_input("Maximum Summary Length", min_value=10, max_value=500, value=130)

    # Initialize session state for audio file
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

    # Option to upload PDF file or search Wikipedia
    option = st.radio("Select an option", ('Upload PDF', 'Search Wikipedia'))

    if option == 'Upload PDF':
        # PDF upload section
        st.subheader("Upload PDF Document")
        pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if pdf_files is not None and len(pdf_files) <= num_pdfs:
            pdf_text = load_pdf(pdf_files)
            st.write("**Extracted Text from PDFs:**")
            st.text_area("PDF Content", pdf_text, height=400)

            # Summarization button for extracted PDF text
            if st.button("Summarize PDF Text"):
                with st.spinner('Summarizing...'):
                    output = query({"inputs": pdf_text, "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False}})
                    if isinstance(output, list) and len(output) > 0 and 'summary_text' in output[0]:
                        summary = output[0]['summary_text']
                    

                        # Store summary in session state
                        st.session_state['summary_text'] = summary

            # Show "Convert Summary to Audio" button after summarization
            if 'summary_text' in st.session_state and st.session_state['summary_text']:
                output = query({"inputs": pdf_text, "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False}})
                if isinstance(output, list) and len(output) > 0 and 'summary_text' in output[0]:
                    summary = output[0]['summary_text']
                    st.subheader("Summary:")
                    st.write(summary)

                        # Store summary in session state
                    st.session_state['summary_text'] = summary

                if st.button("Convert Summary to Audio"):
                    with st.spinner('Generating audio...'):
                        audio_file = text_to_speech(st.session_state['summary_text'])
                        st.session_state.audio_file = audio_file  # Store the audio file in session state

            # Display audio if it has been generated
            if st.session_state.audio_file:
                st.audio(st.session_state.audio_file, format='audio/mp3')
                st.download_button('Download Audio', data=open(st.session_state.audio_file, 'rb').read(), file_name='summary_audio.mp3', mime='audio/mp3')

    elif option == 'Search Wikipedia':
        # Wikipedia search section
        st.subheader("Search Wikipedia")
        query_input = st.text_input("Enter a Wikipedia topic")

        if query_input:
            wiki_content = load_wikipedia_page(query_input)
            st.write(f"**Wikipedia Content for '{query_input}':**")
            st.text_area("Wikipedia Content", wiki_content, height=400)

            # Summarization button for Wikipedia content
            if st.button("Summarize Wikipedia Content"):
                with st.spinner('Summarizing...'):
                    output = query({"inputs": wiki_content, "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False}})
                    if isinstance(output, list) and len(output) > 0 and 'summary_text' in output[0]:
                        summary = output[0]['summary_text']
                    
                    

                        # Store summary in session state
                        st.session_state['summary_text'] = summary

            # Show "Convert Summary to Audio" button after summarization
            if 'summary_text' in st.session_state and st.session_state['summary_text']:
                output = query({"inputs": wiki_content, "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False}})
                if isinstance(output, list) and len(output) > 0 and 'summary_text' in output[0]:
                    summary = output[0]['summary_text']
                    st.subheader("Summary:")
                    st.write(summary)
                if st.button("Convert Summary to Audio"):
                    with st.spinner('Generating audio...'):
                        audio_file = text_to_speech(st.session_state['summary_text'])
                        st.session_state.audio_file = audio_file  # Store the audio file in session state

            # Display audio if it has been generated
            if st.session_state.audio_file:
                st.audio(st.session_state.audio_file, format='audio/mp3')
                st.download_button('Download Audio', data=open(st.session_state.audio_file, 'rb').read(), file_name='summary_audio.mp3', mime='audio/mp3')

    # Footer
    st.caption("This app allows you to seamlessly access content from uploaded PDFs or Wikipedia for quick research.")


##################################################################################################################


































elif chatbot_functionality == "Senior Chef for Recipe":
    # Senior Chef chatbot code remains unchanged
    

    # Add the rest of the chatbot code here for the Senior Chef functionality
    # Make sure the functionality is the same, no changes in logic




##################################################################################################################

#   4                         Senior Chef for recipe

    import os
    import streamlit as st
    import google.generativeai as genai
    import PIL.Image
    from dotenv import find_dotenv, load_dotenv
    from langchain.schema import SystemMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Load environment variables from .env file
    load_dotenv(find_dotenv(), override=True)

    # Streamlit app title
    st.title("I'm Senior Chef I will give you the detailed recipe of food")

    # Use the default API key
    default_api_key = "AIzaSyD7YOrZrkH4SBkphu50VMJIU2780C7eUQA"
    os.environ["GOOGLE_API_KEY"] = default_api_key

    genai.configure(api_key=default_api_key)

    # Configure the generation parameters
    generation_config = {'temperature': 0.9}

    # File uploader for the image
    uploaded_file = st.file_uploader("Upload an image of Food", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image using PIL
        img = PIL.Image.open(uploaded_file)

        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Basic heuristic to check if the image might contain food
        img_size = img.size
        is_food_image = (img_size[0] > 100 and img_size[1] > 100)  # Check if image dimensions are reasonable

        # If it's not likely a food image, show a warning
        if not is_food_image:
            st.warning("This image does not seem to be food. Please select an image of food only for the recipe.")
        else:
            # Generate content from the image
            model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
            with st.spinner("Generating content from the image..."):
                response = model.generate_content(img)

            # Extract dish name and ingredients
            prompt = f'Extract dish name and main ingredients from this description: {response}'
            with st.spinner("Extracting dish name and ingredients..."):
                response = model.generate_content(prompt)

            # Display the extracted information
            st.subheader("Extracted Information")
            st.write(response.text)

            # ChatGPT-like recipe response
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)
            message = [
                SystemMessage(content="You are a chef, and you have to provide a detailed recipe and country name."),
                HumanMessage(content=response.text)
            ]
        
            with st.spinner("Generating recipe..."):
                recipe = llm.invoke(message)

            # Display the recipe content
            st.subheader("Generated Recipe")
            st.write(recipe.content)

##################################################################################################################

























elif chatbot_functionality == "Stock Prices Analysis App":

    # Stock Prices Analysis App code remains unchanged
    st.title("📈 Stock Prices Analysis App")

    # Add the rest of the chatbot code here for the Stock Prices Analysis App functionality
    # Make sure the functionality is the same, no changes in logic



##################################################################################################################

#   5                         Stock Prices Analysis App

    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import matplotlib.dates as mdates

    # App title and description

    st.markdown("A comprehensive solution to analyze stock price data, calculate percentage changes, identify significant price movements, and predict future trends.")

    # List of stock tickers categorized by sectors
    stock_tickers = {
        "Technology": ["AAPL - Apple Inc.", "MSFT - Microsoft Corporation", "GOOGL - Alphabet Inc. (Class A)", 
                       "AMZN - Amazon.com Inc.", "FB - Meta Platforms Inc. (formerly Facebook)"],
        "Finance": ["JPM - JPMorgan Chase & Co.", "BAC - Bank of America Corporation", "C - Citigroup Inc.", 
                    "WFC - Wells Fargo & Company", "GS - The Goldman Sachs Group, Inc."],
        "Healthcare": ["JNJ - Johnson & Johnson", "PFE - Pfizer Inc.", "MRK - Merck & Co., Inc.", 
                       "UNH - UnitedHealth Group Incorporated", "ABBV - AbbVie Inc."],
        "Consumer Goods": ["PG - Procter & Gamble Co.", "KO - The Coca-Cola Company", "PEP - PepsiCo, Inc.", 
                           "WMT - Walmart Inc.", "COST - Costco Wholesale Corporation"],
        "Energy": ["XOM - Exxon Mobil Corporation", "CVX - Chevron Corporation", "BP - BP plc", 
                   "SLB - Schlumberger Limited", "HAL - Halliburton Company"],
        "Utilities": ["NEE - NextEra Energy, Inc.", "DUK - Duke Energy Corporation", "SO - The Southern Company", 
                      "D - Dominion Energy, Inc.", "ED - Consolidated Edison, Inc."],
        "Telecommunications": ["T - AT&T Inc.", "VZ - Verizon Communications Inc.", "TMUS - T-Mobile US, Inc.", 
                              "S - Dish Network Corporation"],
        "Consumer Discretionary": ["TSLA - Tesla, Inc.", "NFLX - Netflix, Inc.", "NKE - Nike, Inc.", 
                                    "LVS - Las Vegas Sands Corp."]
    }

    # List of cryptocurrency tickers (including the new ones)
    crypto_tickers = {
        "Cryptocurrencies": [
            "BTC-USD - Bitcoin", 
            "ETH-USD - Ethereum", 
            "XRP-USD - Ripple", 
            "LTC-USD - Litecoin", 
            "BCH-USD - Bitcoin Cash",
            "HMSTR-USD - Hamster",  # Updated ticker symbol
            "MYRO-USD - Myro", 
            "DOGE-USD - Dogecoin", 
            "CATT-USD - Catti"
        ]
    }

    # Combine stock and crypto tickers
    all_tickers = {**stock_tickers, **crypto_tickers}

    # Sidebar for stock ticker selection or manual entry
    sector = st.sidebar.selectbox("Select Sector or Cryptocurrency", list(all_tickers.keys()))
    ticker_options = all_tickers[sector]
    selected_ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
    manual_ticker = st.sidebar.text_input("Or enter a stock or crypto ticker (e.g., AAPL or BTC-USD)", value="")
    ticker = manual_ticker if manual_ticker else selected_ticker.split(" - ")[0]  # Get only the ticker symbol

    period = st.sidebar.selectbox("Select Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'], index=4)

    # Fetch stock data from yfinance
    def fetch_stock_data(ticker, period):
        stock_data = yf.Ticker(ticker)
        return stock_data.history(period=period)

    # Load the stock data
    data = fetch_stock_data(ticker, period)

    # Display stock data
    if data.empty:
        st.error("No data found for the selected ticker and period. Please try another one.")
    else:
        st.subheader(f"Stock Data for {ticker}")
        st.write(data.tail())

        # Calculate percentage change
        data['Pct Change'] = data['Close'].pct_change()
        st.subheader(f"Percentage Change in {ticker}")
        st.line_chart(data['Pct Change'])

        # Identify significant price movements
        st.subheader(f"Significant Price Movements for {ticker}")
        significant_moves = data[data['Pct Change'].abs() > 0.02]
        st.write(significant_moves)

        # Moving Average Calculation
        st.subheader("Moving Average")
        window_size = st.sidebar.slider("Moving Average Window Size", 5, 100, 20)
        data['Moving Average'] = data['Close'].rolling(window=window_size).mean()
        st.line_chart(data[['Close', 'Moving Average']])

        # Prediction Section
        st.subheader("Stock Price Prediction")

        # Prepare data for prediction
        data['Date'] = data.index
        data['Date'] = pd.to_numeric(pd.to_datetime(data['Date']))
        X = data[['Date']]
        y = data['Close']

        # Split the data
        if len(data) > 1:  # Ensure there's enough data for splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display predictions
            st.subheader("Predicted vs Actual Prices")
            fig, ax = plt.subplots()

            # Plot actual and predicted prices
            ax.plot(y_test.index, y_test, label="Actual", marker='o')
            ax.plot(y_test.index, y_pred, label="Predicted", linestyle='--', marker='x')

            # Improve date formatting on x-axis
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Set interval for x-axis ticks
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format x-axis ticks
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')  # Rotate x-axis labels

            ax.set_title(f"Predicted vs Actual Prices for {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

        else:
            st.error("Not enough data for prediction. Please select a different period.")

        # Simulated currency prices (Example Feature)
        st.subheader("Simulated Currency Prices")
        currency_options = ['USD', 'EUR', 'GBP', 'JPY', 'AUD']
        selected_currency = st.sidebar.selectbox("Select Currency", currency_options)
        base_price = st.sidebar.number_input("Enter Base Price", value=100)

        # Simulate price variations
        currency_variation = np.random.uniform(-0.05, 0.05, len(data))
        simulated_prices = base_price * (1 + currency_variation)

        # Display simulated prices
        st.line_chart(simulated_prices)

##################################################################################################################


























elif chatbot_functionality == "Text Summarization":

    # Text Summarization code remains unchanged
    
    st.title("Text Summarization")
    
    # Add the rest of the chatbot code here for the Text Summarization functionality
    
    # Make sure the functionality is the same, no changes in logic



















    import streamlit as st
    import requests

    # Hugging Face API setup
    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    headers = {"Authorization": "Bearer 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"}

    # Function to query the model
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    # Streamlit App

    # Input from user
    input_text = st.text_area("Enter the text to summarize:", height=200)

    # Button to trigger summarization
    if st.button("Summarize"):
        if input_text:
            with st.spinner('Summarizing...'):
                # Querying the Hugging Face API
                output = query({"inputs": input_text})

                # Safely accessing the summary text
                if isinstance(output, list) and len(output) > 0 and 'summary_text' in output[0]:
                    summary = output[0]['summary_text']
                    st.subheader("Summary:")
                    st.write(summary)
                elif 'error' in output:
                    st.error(f"API Error: {output['error']}")
                else:
                    st.error("Unexpected response format. Please try again later.")
        else:
            st.error("Please enter text to summarize.")























elif chatbot_functionality == "Text to Audio":
    
    # Text to Audio code remains unchanged
    
    st.title("Text to Audio")
    
    # Add the rest of the chatbot code here for the Text to Audio functionality
    
    # Make sure the functionality is the same, no changes in logic











##################################################################################################################

#     7                       Text To Audio

    import streamlit as st
    import requests
    import json
    from IPython.display import Audio

    # API URL and Headers for Hugging Face
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}  # Replace with your actual API token

    # Function to call the Hugging Face API
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content  # Get binary audio data

    # Streamlit App Layout

    st.write("Enter some text, and the AI will generate speech for you!")

    # Text Input
    input_text = st.text_area("Enter Text", value="The answer to the universe is 42")

    # Generate Audio when button is clicked
    if st.button("Generate Audio"):
        if input_text.strip():
            with st.spinner("Generating audio..."):
                audio_data = query({"inputs": input_text})
                st.success("Audio generated successfully!")

                # Save audio file
                with open("output.wav", "wb") as f:
                    f.write(audio_data)

                # Play the audio
                st.audio("output.wav")


 # Add a download button for the audio file
                st.download_button(
                    label="Download Audio",
                    data=audio_data,
                    file_name="generated_audio.wav",
                    mime="audio/wav"
                )

        else:
            st.error("Please enter some text.")

    # To run this app, use `streamlit run streamlit_app.py` in your terminal.

##################################################################################################################













elif chatbot_functionality == "Audio to text":

    # Stock Prices Analysis App code remains unchanged
    st.title("Audio to text")

    # Add the rest of the chatbot code here for the Stock Prices Analysis App functionality
    # Make sure the functionality is the same, no changes in logic














    import streamlit as st
    import requests
    import os
    import speech_recognition as sr
    from gtts import gTTS

    # Hugging Face API setup for transcription and summarization models
    TRANSCRIPTION_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    SUMMARIZATION_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    headers = {"Authorization": "Bearer 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"}

    # Initialize recognizer for live audio recording
    recognizer = sr.Recognizer()

    # Function to query the transcription model
    def transcribe_audio(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(TRANSCRIPTION_API_URL, headers=headers, data=data)
        return response.json()

    # Function to query the summarization model
    def summarize_text(text):
        response = requests.post(SUMMARIZATION_API_URL, headers=headers, json={"inputs": text})
        return response.json()

    # Function to convert summarized text to audio using gTTS
    def convert_text_to_audio(text, output_filename="summarized_audio.mp3"):
        tts = gTTS(text)
        tts.save(output_filename)
        return output_filename

    # Initialize session state variables if not already initialized
    def initialize_session_state():
        if "transcription" not in st.session_state:
            st.session_state.transcription = ""
        if "summary" not in st.session_state:
            st.session_state.summary = ""
        if "live_transcription" not in st.session_state:
            st.session_state.live_transcription = ""
        if "live_summary" not in st.session_state:
            st.session_state.live_summary = ""
        if "show_summarize_button" not in st.session_state:
            st.session_state.show_summarize_button = False
        if "show_summarize_live_button" not in st.session_state:
            st.session_state.show_summarize_live_button = False

    # Streamlit app
    def main():
        # Initialize session state variables
        initialize_session_state()

        # Radio button to select the method of audio input
        st.subheader("Select an Option")
        audio_input_method = st.radio("Choose how you want to convert audio to text:", 
                                      ("Upload Audio File to Text"))

        # Option 1: File uploader for audio files
        if audio_input_method == "Upload Audio File to Text":
            st.subheader("Upload an Audio File")
            uploaded_file = st.file_uploader("Choose an audio file (.flac, .wav, .mp3)", type=["flac", "wav", "mp3"])

            if uploaded_file is not None:
                # Display the audio player for the uploaded file
                st.audio(uploaded_file, format=uploaded_file.type)

                # Save the file temporarily
                temp_file_path = os.path.join("temp_audio_file")
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Generate Transcription button
                if st.button('Generate Transcription from File'):
                    # Call the transcribe_audio function to send the audio to Hugging Face API
                    with st.spinner('Transcribing the audio...'):
                        transcription_output = transcribe_audio(temp_file_path)

                    # Display the transcription output
                    if "text" in transcription_output:
                        st.session_state.transcription = transcription_output["text"]
                        st.session_state.show_summarize_button = True
                    else:
                        st.subheader("Error:")
                        st.write(transcription_output)

                # Clean up the temp file after use
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            # Show the transcription if available from file
            if st.session_state.transcription:
                st.subheader("Generated Transcription:")
                st.write(st.session_state.transcription)

            # Conditionally show the summarize button for file transcription
            if st.session_state.show_summarize_button and st.session_state.transcription:
                if st.button("Summarize Transcription"):
                    with st.spinner('Summarizing the transcription...'):
                        summarization_output = summarize_text(st.session_state.transcription)

                    # Safely accessing the summary text
                    if isinstance(summarization_output, list) and len(summarization_output) > 0 and 'summary_text' in summarization_output[0]:
                        st.session_state.summary = summarization_output[0]['summary_text']

                        # Hide the summarize button after clicking it
                        st.session_state.show_summarize_button = False
                    elif 'error' in summarization_output:
                        st.error(f"API Error: {summarization_output['error']}")
                    else:
                        st.error("Unexpected response format. Please try again later.")

            # Show the summary if available
            if st.session_state.summary:
                st.subheader("Summarized Transcription:")
                st.write(st.session_state.summary)

                # Button to convert summarized text to audio
                if st.button("Convert Summary to Audio"):
                    with st.spinner('Converting summary to audio...'):
                        audio_file = convert_text_to_audio(st.session_state.summary)

                    # Play the generated audio
                    st.audio(audio_file, format="audio/mp3")

                    # Provide a download button for the generated audio
                    with open(audio_file, "rb") as f:
                        st.download_button(
                            label="Download Summary Audio",
                            data=f,
                            file_name="summary_audio.mp3",
                            mime="audio/mp3"
                        )

       
    # Run the app
    if __name__ == "__main__":
        main()







































##################################################################################################################





































###########################################################################################




elif chatbot_functionality == "Text to Image":

    # Text to Image code remains unchanged
    
    st.title("Text to Image")
    
    # Add the rest of the chatbot code here for the Text to Image functionality
    
    # Make sure the functionality is the same, no changes in logic












##################################################################################################################

#   8                         Text to image

    import streamlit as st
    import requests
    import io
    from PIL import Image

    # Hugging Face API URL and authorization
    
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": "Bearer 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"}

    # Function to send the query and retrieve the image from the API
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        # Check if the response status is OK (200)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Streamlit UI setup

    st.write("Enter a prompt to generate an image:")

    # Input field for the user to enter their prompt
    user_input = st.text_input("Your prompt", value="")

    # Button to trigger image generation
    if st.button("Generate Image"):
        with st.spinner('Generating Image...'):
            if user_input:
                try:
                    # Send the prompt to the API
                    image_bytes = query({"inputs": user_input})

                    # Open the image and display it in the app
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption="Generated Image", use_column_width=True)

                    # Convert the image to bytes for download
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    buffered.seek(0)

                # Add a download button
                    st.download_button(
                        label="Download Image",
                        data=buffered,
                        file_name="generated_image.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a prompt.")


##################################################################################################################

















elif chatbot_functionality == "Text to Music":

    
    # Text to Music code remains unchanged
    
    st.title("Text to Music")
    
    # Add the rest of the chatbot code here for the Text to Music functionality
    
    # Make sure the functionality is the same, no changes in logic







##################################################################################################################

#    9                        Text to Music

    import streamlit as st
    import requests

    # Set the API URL and headers
    API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
    headers = {"Authorization": "Bearer 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"}

    # Function to query the API
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    # Streamlit application
    
    st.write("Enter a description of the music you want to generate:")

    # Text input from user
    user_input = st.text_input("Description", "liquid drum and bass, atmospheric synths, airy sounds")

    if st.button("Generate Audio"):
        if user_input:
            with st.spinner("Generating audio..."):
                audio_bytes = query({"inputs": user_input})

            # Play the audio
            st.audio(audio_bytes, format="audio/wav")

            # Add download button
            st.download_button(
                label="Download Audio",
                data=audio_bytes,
                file_name="generated_audio.wav",
                mime="audio/wav"
            )
        else:
            st.warning("Please enter a description.")


##################################################################################################################

























elif chatbot_functionality == "Video Editing":
    
    # Video Editing code remains unchanged
    
    st.title("Video Editing")
    
    # Add the rest of the chatbot code here for the Video Editing functionality
    
    # Make sure the functionality is the same, no changes in logic
















##################################################################################################################

#  10                          Video Editing

    import streamlit as st
    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.all import speedx, fadein, fadeout
    import tempfile

    # Title of the app

    # Upload video file
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Load video file into a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        video_clip = VideoFileClip(temp_file_path)

        # Display the original video
        st.video(temp_file_path)

        # Sidebar for editing options
        with st.sidebar:
            # Speed adjustment
            speed_factor = st.slider("Adjust Speed", 0.1, 5.0, 1.0)  # Speed factor range
            fade_in_duration = st.slider("Fade In Duration (seconds)", 0, 10, 0)
            fade_out_duration = st.slider("Fade Out Duration (seconds)", 0, 10, 0)

        # Apply speed change
        edited_clip = speedx(video_clip, speed_factor)

        # Apply fade effects after speed adjustment
        if fade_in_duration > 0:
            edited_clip = fadein(edited_clip, fade_in_duration)
        if fade_out_duration > 0:
            edited_clip = fadeout(edited_clip, fade_out_duration)

        # Exporting the edited video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
            edited_video_path = output_file.name
            edited_clip.write_videofile(edited_video_path, codec="libx264")

        st.success("Video edited successfully!")

        # Display the edited video
        st.video(edited_video_path)

        # Download link for the edited video
        with open(edited_video_path, "rb") as file:
            st.download_button("Download Edited Video", file, "edited_video.mp4")

        # Clean up the video clips to free memory
        video_clip.close()
        edited_clip.close()

####################################################################################################################


