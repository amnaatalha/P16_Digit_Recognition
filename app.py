import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2

# Load trained ANN model
with open("assets/model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Digit Recognition", page_icon="🔢", layout="centered")
st.title("🔢 Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9). The app will enhance and predict it using a trained ANN model.")

uploaded_file = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display original
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="🖼 Original Uploaded Image", use_column_width=True)

    # Convert to numpy
    img = np.array(image)

    # --- AUTO ENHANCEMENT PIPELINE ---
    # 1️⃣ Resize for consistency
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 2️⃣ Adaptive thresholding for cleaner digits
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 3️⃣ Denoise slightly with Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 4️⃣ Enhance contrast dynamically using histogram equalization
    img = cv2.equalizeHist(img)

    # 5️⃣ Normalize
    img = img / 255.0

    # 6️⃣ Reshape for model input
    img = img.reshape(1, 28, 28)

    # --- PREDICTION ---
    pred = model.predict(img)
    predicted_digit = np.argmax(pred)

    # Display Results
    st.success(f"✅ **Predicted Digit: {predicted_digit}**")
    st.image(
        (img.reshape(28, 28) * 255).astype(np.uint8),
        caption="🧠 Processed Image (Enhanced for Model)",
        width=140,
    )

    # Confidence visualization
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(range(10), pred[0])
    ax.set_xticks(range(10))
    ax.set_xlabel("Digits (0–9)")
    ax.set_ylabel("Prediction Confidence")
    ax.set_title("Model Confidence per Digit")
    st.pyplot(fig)
