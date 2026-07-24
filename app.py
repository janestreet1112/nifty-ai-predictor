import os
import streamlit as st

def get_secret(key):
    """Fetches secrets from OS environment variables (Render) or Streamlit secrets."""
    return os.environ.get(key) or st.secrets.get(key)

# Read credentials
api_key = get_secret("SMARTAPI_API_KEY")
client_id = get_secret("SMARTAPI_USERNAME")
pin = get_secret("SMARTAPI_PASSWORD")
totp_secret = get_secret("SMARTAPI_TOTP_KEY")
