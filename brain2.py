import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import base64
import os


st.set_page_config(page_title="Brain Tumor Detection", layout="centered")


MODEL_PATH = '/Users/varshanathkm/Desktop/project deep learning/cnn_model.h5' 
IMG_SIZE = (128, 128)       
BACKGROUND_IMAGE = '/Users/varshanathkm/Desktop/project deep learning/brain1.jpg'


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


def get_base64_img(path):
    with open(path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background(path):
    if os.path.exists(path):
        bg = get_base64_img(path)
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bg}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_background(BACKGROUND_IMAGE)

st.markdown(
    "<h1 style='color:black;'>Brain Tumor Detection from MRI Scan</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:black;'>Upload an MRI image to predict if a tumor exists.</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
predict_button = st.button("Upload")

if uploaded_file and predict_button:
    try:
        img = image.load_img(uploaded_file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "Tumor Detected (YES)" if prediction >= 0.5 else "No Tumor (NO)"
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if prediction >= 0.5:
            st.markdown(
                f"<div style='color:black; font-weight:bold; font-size:20px;'>{label}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='color:black; font-weight:bold; font-size:20px;'>{label}</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"<p style='color:black;'>Confidence Score: {prediction:.2f}</p>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")