import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model
import streamlit as st
from PIL import Image
import tempfile
import os




model = load_model('bestmodel.h5')


st.set_page_config(
    page_title="Brain Tumor Prediction Model",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header('Brain Tumor Prediction Model')

uploaded_file = st.file_uploader("Upload the MRI of Patient's Brain", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the image file
    image = Image.open(temp_file_path)

    # Convert image to grayscale
    image = image.convert("L")

    path = temp_file_path
    img= load_img(path,target_size=(224,224))
    input_arr = img_to_array(img)/255

    # Display the resized grayscale image
    left_frame, right_frame = st.columns(2)
    with left_frame:
        st.image(input_arr, caption="Uploaded MRI")

    # Right frame for the "HELLO" text
    with right_frame:
        input_arr = np.expand_dims(input_arr, axis=0)

        pred_prob = model.predict(input_arr)[0][0]
        # pred_class = 1 if pred_prob > 0.5 else 0
        st.write(f"The probability of having Brain Tumor is {pred_prob}" )
        if pred_prob >0.5:
            st.write("Do visit the doctor for further diagnosis and treatment")

st.write(f'The accuracy of this prediction model is 96.28 !')







