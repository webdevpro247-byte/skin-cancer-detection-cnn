import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Skin Cancer Analyzer", layout="centered")

st.title("🩺 Skin Cancer Analyzer")
st.write("Upload a skin lesion image to get a model prediction (Benign vs. Malignant).")

# ⚠️ EDUCATIONAL MEDICAL DISCLAIMER
st.warning(
    "⚠️ **Disclaimer: Educational Use Only**

"
    "This application is an educational prototype developed for learning purposes using a convolutional neural network (CNN).
"
    "It is NOT a medical tool, has NOT been clinically validated, and MUST NOT be used for medical diagnosis, "
    "screening, treatment, or clinical decision-making. Always consult a qualified medical professional for skin concerns."
)

# Load Model
MODEL_PATH = 'skin_cancer_cnn.keras'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.error(f"Model file '{MODEL_PATH}' was not found or failed to load. Please ensure it is uploaded or placed in the same directory.")
else:
    st.success("Model loaded successfully! Ready for analysis.")
    
    uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        st.write("Analyzing image...")
        
        # Exact 224x224 RGB resizing as used during training
        resized_image = image.resize((224, 224))
        img_array = np.array(resized_image) / 255.0  # Normalized normalization
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        try:
            predictions = model.predict(img_array)
            # Assuming binary classification with sigmoid activation outputting probability of malignancy
            malignant_prob = float(predictions[0][0])
            benign_prob = 1.0 - malignant_prob
            
            st.subheader("Analysis Results")
            
            if malignant_prob > 0.5:
                st.error(f"**Prediction: Malignant** ({malignant_prob * 100:.2f}% confidence)")
            else:
                st.success(f"**Prediction: Benign** ({benign_prob * 100:.2f}% confidence)")
                
            # Show comparative probabilities
            st.progress(malignant_prob)
            st.write(f"🔹 Benign probability: {benign_prob:.4f}")
            st.write(f"🔸 Malignant probability: {malignant_prob:.4f}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
