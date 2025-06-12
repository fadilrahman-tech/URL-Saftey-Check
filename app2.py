import streamlit as st
import joblib
import pandas as pd
import re
import urllib.parse
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

# Load the model
try:
    # --- CHANGE MADE HERE: Model filename updated to 'website_classifier_model.pkl' ---
    model = joblib.load('website_classifier_model.pkl')
except FileNotFoundError:
    st.error("Error: 'website_classifier_model.pkl' not found. Please ensure the model file is in the same directory as your Streamlit app.")
    st.stop()

# --- IMPORTANT: MODEL'S EXPECTED FEATURES ---
# THIS LIST MUST EXACTLY MATCH THE 'MODEL_EXPECTED_FEATURES' list
# that was printed by your training script (your_updated_training_script.py).
# COPY AND PASTE IT HERE AFTER YOU HAVE RUN THE TRAINING SCRIPT SUCCESSFULLY!
feature_columns_from_training = [
    'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash', 'NumDashInHostname',
    'AtSymbol', 'TildeSymbol', 'NumUnderscore', 'NumPercent', 'NumQueryComponents',
    'NumAmpersand', 'NumHash', 'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
    'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname', 'HostnameLength', 'PathLength',
    'QueryLength', 'DoubleSlashInPath', 'NumSensitiveWords', 'EmbeddedBrandName',
    'PctExtHyperlinks', 'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
    'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction', 'PctNullSelfRedirectHyperlinks',
    'FrequentDomainNameMismatch', 'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
    'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle', 'ImagesOnlyInForm',
    'SubdomainLevelRT', 'UrlLengthRT', 'PctExtResourceUrlsRT', 'AbnormalExtFormActionR',
    'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
]

# --- Feature Extraction Function (Modified to return content_fetch_success) ---
def extract_all_features_from_url(url, target_feature_columns):
    """
    Extracts a predefined set of features from a given URL string.
    Features are initialized to 0 and populated based on the URL's characteristics
    and (optionally) its content.
    Returns: (features_dict, content_fetch_success_flag)
    """
    features_dict = {col: 0 for col in target_feature_columns} # Initialize all to 0
    content_fetch_success = False # Flag to track if web content was successfully fetched

    try:
        parsed_url = urllib.parse.urlparse(str(url))
        hostname = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        fragment = parsed_url.fragment

        # --- Lexical Features (derived directly from URL string) ---
        if 'NumDots' in features_dict: features_dict['NumDots'] = str(url).count('.')
        if 'UrlLength' in features_dict: features_dict['UrlLength'] = len(str(url))
        if 'HostnameLength' in features_dict: features_dict['HostnameLength'] = len(hostname)
        if 'PathLength' in features_dict: features_dict['PathLength'] = len(path)
        if 'QueryLength' in features_dict: features_dict['QueryLength'] = len(query)

        if 'SubdomainLevel' in features_dict:
            if hostname:
                parts = hostname.split('.')
                subdomain_count = len(parts) - 1
                if parts and parts[0].lower() == 'www':
                    subdomain_count -= 1
                features_dict['SubdomainLevel'] = max(0, subdomain_count)

        if 'PathLevel' in features_dict: features_dict['PathLevel'] = path.count('/') if path else 0
        if 'NumDash' in features_dict: features_dict['NumDash'] = str(url).count('-')
        if 'NumDashInHostname' in features_dict: features_dict['NumDashInHostname'] = hostname.count('-')
        if 'AtSymbol' in features_dict: features_dict['AtSymbol'] = str(url).count('@')
        if 'TildeSymbol' in features_dict: features_dict['TildeSymbol'] = str(url).count('~')
        if 'NumUnderscore' in features_dict: features_dict['NumUnderscore'] = str(url).count('_')
        if 'NumPercent' in features_dict: features_dict['NumPercent'] = str(url).count('%')
        if 'NumQueryComponents' in features_dict: features_dict['NumQueryComponents'] = len(query.split('&')) if query else 0
        if 'NumAmpersand' in features_dict: features_dict['NumAmpersand'] = query.count('&')
        if 'NumHash' in features_dict: features_dict['NumHash'] = str(url).count('#')
        if 'NumNumericChars' in features_dict: features_dict['NumNumericChars'] = sum(c.isdigit() for c in str(url))
        if 'NoHttps' in features_dict: features_dict['NoHttps'] = int(not str(url).startswith('https://'))
        if 'IpAddress' in features_dict: features_dict['IpAddress'] = int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname)))
        if 'HttpsInHostname' in features_dict: features_dict['HttpsInHostname'] = int('https' in hostname.lower())
        if 'DoubleSlashInPath' in features_dict: features_dict['DoubleSlashInPath'] = int('//' in path and path.index('//') > 0)

        if 'NumSensitiveWords' in features_dict:
            sensitive_keywords = ['login', 'signin', 'account', 'secure', 'webscr', 'cmd', 'paypal', 'bank', 'update', 'verify']
            features_dict['NumSensitiveWords'] = sum(1 for keyword in sensitive_keywords if keyword in str(url).lower())

        # Placeholder/Complex Features (set to 0 for now)
        if 'RandomString' in features_dict: features_dict['RandomString'] = 0
        if 'DomainInSubdomains' in features_dict: features_dict['DomainInSubdomains'] = 0
        if 'DomainInPaths' in features_dict: features_dict['DomainInPaths'] = 0
        if 'EmbeddedBrandName' in features_dict: features_dict['EmbeddedBrandName'] = 0

        # --- Content-based Features (require fetching webpage content) ---
        soup = None
        # Only attempt web fetch if content-based features are needed and URL starts with a scheme
        requires_web_fetch = any(f in features_dict for f in [
            'PctExtHyperlinks', 'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
            'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
            'PctNullSelfRedirectHyperlinks', 'MissingTitle', 'IframeOrFrame',
            'ImagesOnlyInForm', 'FrequentDomainNameMismatch', 'FakeLinkInStatusBar',
            'RightClickDisabled', 'PopUpWindow', 'SubmitInfoToEmail'
        ])

        if requires_web_fetch and str(url).startswith(('http://', 'https://')):
            try:
                response = requests.get(str(url), timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    content_fetch_success = True # Set flag to True if content was successfully fetched
            except requests.exceptions.RequestException:
                pass # Silently fail on network errors/timeouts, content_fetch_success remains False
            except Exception:
                pass # Catch other potential errors, content_fetch_success remains False

        if soup: # Only extract content-based features if soup was successfully created
            if 'PctExtHyperlinks' in features_dict:
                total_links = len(soup.find_all('a', href=True))
                ext_links = 0
                if total_links > 0:
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href']
                        if href and href.startswith('http') and urllib.parse.urlparse(href).netloc != hostname:
                            ext_links += 1
                    features_dict['PctExtHyperlinks'] = (ext_links / total_links) * 100
                else:
                    features_dict['PctExtHyperlinks'] = 0

            if 'PctExtResourceUrls' in features_dict:
                total_res = 0
                ext_res = 0
                for tag in soup.find_all(['link', 'script', 'img']):
                    src_or_href = tag.get('src') or tag.get('href')
                    if src_or_href:
                        total_res += 1
                        if src_or_href.startswith('http') and urllib.parse.urlparse(src_or_href).netloc != hostname:
                            ext_res += 1
                if total_res > 0:
                    features_dict['PctExtResourceUrls'] = (ext_res / total_res) * 100
                else:
                    features_dict['PctExtResourceUrls'] = 0

            if 'ExtFavicon' in features_dict:
                favicon_link = soup.find('link', rel=re.compile(r'icon', re.I))
                if favicon_link and favicon_link.get('href'):
                    href = favicon_link['href']
                    if href.startswith('http') and urllib.parse.urlparse(href).netloc != hostname:
                        features_dict['ExtFavicon'] = 1
                    else:
                        features_dict['ExtFavicon'] = 0
                else:
                    features_dict['ExtFavicon'] = -1

            if 'InsecureForms' in features_dict:
                insecure_forms = 0
                for form_tag in soup.find_all('form', action=True):
                    action = form_tag['action']
                    if not action.startswith('https://') or action.strip() == '':
                        insecure_forms = 1
                        break
                features_dict['InsecureForms'] = insecure_forms

            if 'RelativeFormAction' in features_dict:
                rel_forms = 0
                forms = soup.find_all('form', action=True)
                if forms:
                    for form_tag in forms:
                        action = form_tag['action']
                        if not action.startswith('http://') and not action.startswith('https://') and action.strip() != '':
                            rel_forms = 1
                            break
                    features_dict['RelativeFormAction'] = rel_forms
                else:
                    features_dict['RelativeFormAction'] = -1

            if 'ExtFormAction' in features_dict:
                ext_forms = 0
                forms = soup.find_all('form', action=True)
                if forms:
                    for form_tag in forms:
                        action = form_tag['action']
                        if action.startswith('http') and urllib.parse.urlparse(action).netloc != hostname:
                            ext_forms = 1
                            break
                    features_dict['ExtFormAction'] = ext_forms
                else:
                    features_dict['ExtFormAction'] = -1

            if 'AbnormalFormAction' in features_dict:
                abnormal_forms = 0
                forms = soup.find_all('form', action=True)
                if forms:
                    for form_tag in forms:
                        action = form_tag['action']
                        parsed_action = urllib.parse.urlparse(action)
                        if parsed_action.netloc and parsed_action.netloc != hostname:
                            abnormal_forms = 1
                            break
                    features_dict['AbnormalFormAction'] = abnormal_forms
                else:
                    features_dict['AbnormalFormAction'] = -1

            if 'PctNullSelfRedirectHyperlinks' in features_dict:
                total_links = len(soup.find_all('a', href=True))
                null_self_redirect_links = 0
                if total_links > 0:
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href'].strip()
                        if href == '#' or href == str(url) or href == parsed_url.path:
                            null_self_redirect_links += 1
                    features_dict['PctNullSelfRedirectHyperlinks'] = (null_self_redirect_links / total_links) * 100
                else:
                    features_dict['PctNullSelfRedirectHyperlinks'] = 0

            if 'MissingTitle' in features_dict:
                features_dict['MissingTitle'] = int(soup.find('title') is None or not soup.find('title').text.strip())

            if 'IframeOrFrame' in features_dict:
                features_dict['IframeOrFrame'] = int(bool(soup.find('iframe') or soup.find('frame')))

            if 'ImagesOnlyInForm' in features_dict: features_dict['ImagesOnlyInForm'] = 0 # Placeholder

            # Placeholder RT features for content-based checks (very complex, set to 0)
            if 'SubdomainLevelRT' in features_dict: features_dict['SubdomainLevelRT'] = 0
            if 'UrlLengthRT' in features_dict: features_dict['UrlLengthRT'] = 0
            if 'PctExtResourceUrlsRT' in features_dict: features_dict['PctExtResourceUrlsRT'] = 0
            if 'AbnormalExtFormActionR' in features_dict: features_dict['AbnormalExtFormActionR'] = 0
            if 'ExtMetaScriptLinkRT' in features_dict: features_dict['ExtMetaScriptLinkRT'] = 0
            if 'PctExtNullSelfRedirectHyperlinksRT' in features_dict: features_dict['PctExtNullSelfRedirectHyperlinksRT'] = 0

            # Other features requiring advanced analysis or real-time checks
            if 'FrequentDomainNameMismatch' in features_dict: features_dict['FrequentDomainNameMismatch'] = 0
            if 'FakeLinkInStatusBar' in features_dict: features_dict['FakeLinkInStatusBar'] = 0
            if 'RightClickDisabled' in features_dict: features_dict['RightClickDisabled'] = 0
            if 'PopUpWindow' in features_dict: features_dict['PopUpWindow'] = 0
            if 'SubmitInfoToEmail' in features_dict: features_dict['SubmitInfoToEmail'] = 0

    except Exception:
        pass # All features remain 0 (their initialized value), content_fetch_success remains False

    return features_dict, content_fetch_success

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