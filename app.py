import streamlit as st
import tensorflow as tf
import numpy as np

# Load your model
model = tf.keras.models.load_model('app/model/earthquake.h5')

# Streamlit UI
st.title('Earthquake Prediction')

# Inputs
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0)
depth = st.number_input('Depth', min_value=0.0)
mag = st.number_input('Magnitude', min_value=0.0)
year = st.number_input('Year', min_value=1900, max_value=2100)  # Example input for year
damage_description = st.text_input('Damage Description', '')  # Example input for damage description

# Prediction
if st.button('Predict'):
    # Convert the 'damage_description' to binary (1 if non-empty, 0 if empty)
    damage_binary = 1 if damage_description else 0

    # Prepare the data with 6 features
    data = np.array([[latitude, longitude, depth, mag, year, damage_binary]])

    # Ensure that the data shape is (1, 6)
    prediction = model.predict(data)
    st.write(f"Prediction (Magnitude and Range): {prediction}")

