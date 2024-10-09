import time
import base64
from io import BytesIO
import requests
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

url = "http://localhost:2004/transform"

model_names = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
model_name = st.sidebar.selectbox("Select a model", options=model_names)

load_size = st.sidebar.slider("Set image size", 100, 800, 300, 20)
uploaded_image = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

# Replace st.beta_columns with st.columns
cols = st.columns((1, 1))

with cols[0]:
    input_image = st.empty()

with cols[1]:
    transformed_image = st.empty()

if uploaded_image is not None:
    # Using columns for 'Cartoonify!' button and 'Download!' button
    col1, col2 = st.sidebar.columns(2)

    with col1:
        transform = st.button("Cartoonify!")
    
    # Placeholder for the download button, will be updated later
    download = None

    pil_image = Image.open(uploaded_image)
    
    # Get width and height of the input image
    width, height = pil_image.size

    image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")

    data = {
        "image": image,
        "model_id": model_names.index(model_name),
        "load_size": load_size,
    }

    # Display the input image (full size)
    input_image.image(pil_image, use_column_width=True)

    if transform:
        t0 = time.time()
        response = requests.post(url, json=data)
        delta = time.time() - t0
        image = response.json()["output"]
        image = image[image.find(",") + 1:]
        dec = base64.b64decode(image + "===")
        binary_output = BytesIO(dec)

        st.sidebar.warning(f"Processing took {delta:.3} seconds")

        # Load the output image from the binary data
        output_image = Image.open(binary_output)
        
        # Resize the output image to match the input image's height
        output_image = output_image.resize((width, height))

        # Display the transformed image with the same height as the input image
        transformed_image.image(output_image, use_column_width=True)

        # Convert the output image to bytes for download
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_data = buffered.getvalue()

        # Display the 'Download' button next to 'Cartoonify!' after transformation
        with col2:
            st.download_button(
                label="Download!",
                data=img_data,
                file_name="after.png",
                mime="image/png"
            )