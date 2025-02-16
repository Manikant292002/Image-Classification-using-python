import streamlit as st
import tensorflow as tf
import numpy as np

# Ensure the model path is correct and the model exists
model_path = 'Image_classify.keras'
model = tf.keras.models.load_model(model_path)

# Define the data categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 
            'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 
            'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 
            'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 
            'turnip', 'watermelon']

img_height = 180
img_width = 180

# Streamlit header
st.header('Image Classification Model')

# Text input for image name
image_name = st.text_input('Enter Image name', 'apple.jpg')

try:
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(image_name, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)  # Create a batch

    # Predict the class of the image
    predictions = model.predict(img_bat)
    score = tf.nn.softmax(predictions[0])

    # Display the image and prediction results
    st.image(image_name, width=200)
    st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
    st.write('With accuracy of {:.2f}%'.format(np.max(score) * 100))
except Exception as e:
    st.error(f"Error loading or processing image: {e}")
