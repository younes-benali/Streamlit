import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np

st.set_page_config(page_title='Image Classifier',page_icon='🏞️',layout='wide')

# Load model safely
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('CNN_IC.keras')
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# Define classes
classes_dict = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

# App header
st.title("🏞️ Image Classifier")
st.markdown("Upload an image and let the CNN model classify it into one of six categories.")

# Sidebar for instructions
st.sidebar.header("ℹ️ Instructions")
st.sidebar.write("""
1. Upload an image (JPEG/PNG).
2. Click **Classify** to run the model.
3. See the predicted class and confidence score.
""")

# File uploader
file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

def preprocess_image(file):
    """Preprocess image for model prediction."""
    try:
        img_width, img_height = 150, 150
        img = Image.open(file).convert('RGB')
        img_resized = img.resize((img_width, img_height))
        img_array = np.array(img_resized) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)
        return img_tensor, img
    except UnidentifiedImageError:
        st.error("❌ Invalid image file. Please upload a valid picture.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None, None

if file:
    img_tensor, img = preprocess_image(file)
    if img is not None:
        st.success("✅ Image loaded successfully")
        st.image(img)#, use_container_width=True)

        col1,col2,col3 = st.columns(3)
        with col1:
            st.write('')
        with col2:
            button = st.button("🔍 Classify")
        with col3:
            st.write('')

        if button:
            if model is not None and img_tensor is not None:
                try:
                    prediction = model.predict(img_tensor, verbose=0)
                    predicted_index = np.argmax(prediction)
                    predicted_class = classes_dict[predicted_index]
                    confidence = np.max(prediction) * 100

                    st.success("🎉 Prediction successful!")
                    st.metric(label="Predicted Class", value=predicted_class)
                    st.progress(int(confidence))
                    st.write(f"Confidence: **{confidence:.2f}%**")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
            else:
                st.warning("⚠️ Model not available or image not processed.")
