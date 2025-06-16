import streamlit as st
import joblib
import pandas as pd
import re
import urllib.parse
import numpy as np
import requests
from bs4 import BeautifulSoup
import time


# This line imports the feature list and extraction function from 'features.py'
from features import feature_columns_from_training, extract_all_features_from_url

# Load the model
try:
    # --- Importing the pre-trained model ---
    model = joblib.load('website_classifier_model.pkl')
except FileNotFoundError:
    st.error("Error: 'website_classifier_model.pkl' not found. Please ensure the model file is in the same directory as your Streamlit app.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Phishing Website Detector", page_icon="🔒")
st.title("🔒 Phishing Website Detector")
st.write("Enter a URL below to check if it's **Benign** or **Malicious**.")

input_url = st.text_input("Enter website URL:")

if st.button("Check"):
    if not input_url.strip(): # Check if the URL is empty or just whitespace
        st.warning("Please enter a URL.")
    else:
        processed_url = input_url.strip()

        # Add https:// if no scheme is provided
        if not (processed_url.startswith("http://") or processed_url.startswith("https://")):
            processed_url = "https://" + processed_url
            st.info(f"Automatically prepending 'https://' to your URL: {processed_url}")

        with st.spinner("Analyzing website... This might take a moment as content is being fetched."):
            # Call the imported feature extraction function
            features_dict, content_fetch_successful = extract_all_features_from_url(processed_url, feature_columns_from_training)

            if not content_fetch_successful:
                st.warning(f"Could not fetch webpage content for: {processed_url}. This often indicates an inaccessible, non-existent, or problematic domain.")
                st.subheader("Prediction: Malicious ❌ (Content Unavailable)")
                st.write("For security, URLs that cannot be reached or resolved are considered suspicious and classified as malicious.")
            else:
                try:
                    features_df = pd.DataFrame([features_dict], columns=feature_columns_from_training)
                    prediction = model.predict(features_df)[0]
                    
                    label = "Malicious ❌" if prediction == 1 else "Benign ✅"
                    st.subheader(f"Prediction: {label}")

                    # Optional: Show feature values for debugging/insight
                    # st.write("---")
                    # st.write("Extracted Features:")
                    # st.dataframe(features_df.T, use_container_width=True)

                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction after feature extraction: {e}")
                    st.warning("Please check the URL and the model file.")
