import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("digit_recognition_model.h5")

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✏️", layout="centered")

st.title("✏️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (28x28, grayscale) or draw one below.")

# ==============================
# 1. Upload an image option
# ==============================
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Preprocess image
    img_resized = cv2.resize(image, (28, 28))
    img_resized = img_resized / 255.0
    img_resized = img_resized.reshape(1, 28, 28, 1)

    # Prediction
    prediction = np.argmax(model.predict(img_resized))
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.subheader(f"Prediction: {prediction}")

# ==============================
# 2. Draw on canvas option
# ==============================
st.subheader("Or draw a digit below")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Preprocess drawing
    img_resized = cv2.resize(gray, (28, 28))
    img_resized = img_resized / 255.0
    img_resized = img_resized.reshape(1, 28, 28, 1)

    # Prediction
    prediction = np.argmax(model.predict(img_resized))
    st.image(gray, caption="Drawn Digit", use_container_width=True)
    st.subheader(f"Prediction: {prediction}")
