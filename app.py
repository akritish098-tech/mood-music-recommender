import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load music dataset
music_df = pd.read_csv("music.csv")

st.title("🎧 AI Mood Based Music Recommender")

user_text = st.text_area("How are you feeling today?")

if st.button("Recommend Music"):
    if user_text.strip() == "":
        st.warning("Please enter how you are feeling.")
    else:
        text_vec = vectorizer.transform([user_text])
        mood = model.predict(text_vec)[0]
        probs = model.predict_proba(text_vec)[0]
        confidence = np.max(probs) * 100

        st.success(f"Detected Mood: {mood.upper()} ({confidence:.2f}% confidence)")

        recommendations = music_df[music_df["mood"] == mood]

        st.write("### 🎵 Recommended Songs:")
        for song in recommendations["song"]:
            st.write("•", song)

