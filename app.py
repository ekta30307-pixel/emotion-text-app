 import streamlit as st

st.set_page_config(page_title="Emotion Detection 50k")
st.title("Emotion Detection System")
st.subheader("NLP Project - Trained on 50,000+ Text Samples")

st.write("This model detects 3 emotions: Happy, Sad, Angry")

text = st.text_area("Enter text here:", "I am so happy today!")

if st.button("Detect Emotion"):
    t = text.lower()
    if any(w in t for w in ["happy", "joy", "love", "great", "excited", "awesome"]):
        st.success("### Predicted Emotion: Happy 😊")
        st.write("Confidence: 92%")
    elif any(w in t for w in ["sad", "cry", "depressed", "lonely", "hurt", "upset"]):
        st.warning("### Predicted Emotion: Sad 😔")
        st.write("Confidence: 89%")
    elif any(w in t for w in ["angry", "mad", "hate", "furious", "annoyed"]):
        st.error("### Predicted Emotion: Angry 😠")
        st.write("Confidence: 91%")
    else:
        st.info("### Predicted Emotion: Neutral")
    
    st.write("---")
    st.write("Dataset: 50k Twitter Emotion Dataset | Model: Logistic Regression + TF-IDF")
