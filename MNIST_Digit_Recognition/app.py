
import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
from pathlib import Path

# Load trained model
model_path = Path(__file__).parent / "mnist_digit_model_small.pkl"
model = joblib.load(model_path)

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9).")

uploaded_file = st.file_uploader(
    "Upload your digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.subheader("Uploaded Image")
    st.image(image, width=200)

    if st.button("Predict Digit"):

        # Resize image to MNIST dimensions
        image = ImageOps.invert(image)
        image = image.resize((28, 28))

        # Convert image to numerical array
        image_array = np.array(image)

        # Normalize pixel values
        image_array = image_array.astype("float32") / 255.0

        # Flatten image into 784 features
        image_array = image_array.reshape(1, 784)

        # Predict digit
        prediction = model.predict(image_array)

        st.success(f"Predicted Digit: {prediction[0]}")
