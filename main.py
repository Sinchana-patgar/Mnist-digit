import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("model_new.h5")
st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28) / 255.0

    # If background is white/light, invert colors to match MNIST style
    if img_array.mean() > 0.5:
        img_array = 1 - img_array

    prediction = model.predict(img_array)
    st.write(f"Predicted digit: {np.argmax(prediction)}")
