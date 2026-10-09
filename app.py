import streamlit as st
import joblib

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Emotion Detection - 50k Project")
st.write("Trained on 43,410 real comments - 73.78% Accuracy")

text = st.text_area("Enter your text here:")

if st.button("Detect Emotion"):
    if text:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        if pred == "happy":
            st.success(f"😊 Happy: {text}")
        elif pred == "sad":
            st.warning(f"😔 Sad: {text}")
        else:
            st.error(f"😠 Angry: {text}")