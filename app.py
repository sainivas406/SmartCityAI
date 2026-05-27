import streamlit as st
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Website title
st.title("AI Smart City Project")

# Upload CSV dataset
uploaded_file = st.file_uploader(
    "Upload Traffic Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # Read dataset
    data = pd.read_csv(uploaded_file)

    st.write(data.head())

    # Input and output
    X = data[['Junction', 'Vehicles']]

    y = data['ID']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = LinearRegression()

    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    st.success("Traffic Prediction Completed")

    # Graph
    fig, ax = plt.subplots()

    ax.plot(y_test.values[:50], label='Actual')

    ax.plot(predictions[:50], label='Predicted')

    ax.legend()

    st.pyplot(fig)

# Crowd Image Upload
image_file = st.file_uploader(
    "Upload Crowd Image",
    type=["jpg", "png", "jpeg"]
)

if image_file is not None:

    # Convert image
    file_bytes = np.asarray(
        bytearray(image_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, 1)

    image = cv2.resize(image, (800, 600))

    # Gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (11,11), 0)

    # Threshold
    thresh = cv2.threshold(
        blur,
        200,
        255,
        cv2.THRESH_BINARY
    )[1]

    # Crowd density
    crowd_density = cv2.countNonZero(thresh)

    # Show image
    st.image(image, channels="BGR")

    st.write("Crowd Density:", crowd_density)

    # Alert
    if crowd_density > 50000:

        st.error("ALERT: Heavy Crowd Detected")

    else:

        st.success("Crowd is Normal")