import streamlit as st
import pandas as pd
import joblib
from logistic_regression import ImprovedSuspiciousLoginDetector
from datetime import datetime
import requests

# Load model and threshold
model = ImprovedSuspiciousLoginDetector.load_model("suspicious_login_model_20250610_114330.joblib")
threshold = joblib.load("threshold.joblib")

# Configure page
st.set_page_config(page_title="Suspicious Login Detection", layout="centered")

st.title(":red[Suspicious Login Detection]")

st.header("Mandatory Features", divider="red")
user_input = {
    "login_timestamp": st.text_input("Login Timestamp (YYYY-MM-DDTHH:MM:SS)", datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
    "ip_address": st.text_input("IP Address", "192.168.1.1"),
    "latitude": round(st.number_input("Latitude", value=0.0, format="%.6f"), 6),
    "longitude": round(st.number_input("Longitude", value=0.0, format="%.6f"), 6),
    "is_blacklisted": st.selectbox("Is Blacklisted?", [0, 1]),
    "login_success": st.selectbox("Login Success?", [0, 1])
}

st.header("Optional Features", divider="red")
if st.checkbox("Add user_id"):
    user_input["user_id"] = st.text_input("User ID", "u001")
if st.checkbox("Add device_type"):
    user_input["device_type"] = st.selectbox("Device Type", ["mobile", "desktop", "tablet"])
if st.checkbox("Add browser"):
    user_input["browser"] = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Opera"])
if st.checkbox("Add is_off_hours"):
    user_input["is_off_hours"] = st.selectbox("Is Off Hours?", [0, 1])
if st.checkbox("Add pincode"):
    user_input["pincode"] = st.number_input("Pincode", value=560001)

# Session state to hold prediction result
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
    st.session_state.alert_message = ""

# Predict button
if st.button("Predict"):
    try:
        df = pd.DataFrame([user_input])
        df["login_timestamp"] = pd.to_datetime(df["login_timestamp"])
        prediction, scores, _ = model.predict_enhanced(df)
        score = float(scores[0])
        label = "suspicious" if score >= threshold else "normal"

        st.write(f"Suspicion Score: {round(score, 4)}")
        st.write(f"Threshold Used: {round(threshold, 4)}")

        if label == "normal":
            st.success("Prediction: NORMAL")
            st.session_state.prediction_done = False
        else:
            st.error("Prediction: SUSPICIOUS")
            st.session_state.prediction_done = True
            st.session_state.alert_message = (
                f"**Suspicious Login Detected!**\n"
                f" Timestamp: {user_input['login_timestamp']}\n"
                f" Location: ({user_input['latitude']}, {user_input['longitude']})\n"
                f" {round(score, 4)*100} % suspicious"
            )

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Discord webhook function
def send_discord_alert(webhook_url, message):
    try:
        data = {"content": message}
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            st.success("Discord alert sent successfully!")
        else:
            st.error(f"❌ Failed to send Discord alert: {response.text}")
    except Exception as e:
        st.error(f"Discord error: {str(e)}")

# Show alert button if prediction is suspicious
if st.session_state.prediction_done is True:
    if (st.button("Send Discord Alert") or st.button("Send SMS") or st.button("Send Message")): #placeholder
        webhook_url = st.secrets["DISCORD"]["webhook_url"]
        send_discord_alert(webhook_url, st.session_state.alert_message)
