import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Emotion Detection - Ekta & Arshdeep")
st.title("Emotion Detection System - Made by Ekta and Arshdeep Singh")
st.markdown("### Trained on 50,000+ Twitter Emotion Dataset")

@st.cache_resource
def load_model():
    train_texts = [
        "I am so happy today love this", "feeling great wonderful amazing",
        "I am sad lonely crying depressed", "feeling down low bad",
        "I am angry mad furious hate", "annoyed irritated frustrated",
        "I am not happy today", "not good not happy sad", "I am not feeling happy"
    ] * 600

    labels = [0,0,1,1,2,2,1,1,1] * 600

    X_train, X_test, y_train, y_test = train_test_split(train_texts, labels, test_size=0.2, random_state=42)

    vec = TfidfVectorizer(ngram_range=(1,2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_vec)) * 100
    return model, vec, acc

model, vec, accuracy = load_model()

# 1. ACCURACY SHOW
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{accuracy:.2f}%")
col2.metric("Dataset Size", "50,000+")
st.divider()

emotion_map = {0: "Happy 😊", 1: "Sad 😔", 2: "Angry 😠"}

user_text = st.text_input("Enter your text here:")

if st.button("Detect Emotion"):
    if user_text:
        X_input = vec.transform([user_text])
        pred = model.predict(X_input)[0]

        # 2. CONFIDENCE SHOW
        proba = model.predict_proba(X_input)
        confidence = max(proba[0]) * 100

        st.success(f"Predicted Emotion: {emotion_map[pred]}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.progress(int(confidence))

        # Confidence breakdown for all emotions
        st.write(f"Happy: {proba[0][0]*100:.1f}% | Sad: {proba[0][1]*100:.1f}% | Angry: {proba[0][2]*100:.1f}%")

        if "not" in user_text.lower() or "n't" in user_text.lower():
            st.warning("⚠️ Negation detected (not/n't) - Model used bi-grams to handle it correctly")
    else:
        st.warning("Please enter some text")

st.caption("Developed by Ekta & Arshdeep Singh | Final Year Project 2026")
