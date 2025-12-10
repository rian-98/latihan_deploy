import joblib
import streamlit as st

model = joblib.load("models/model_logistic_regression.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Aplikasi Klasifikasi Komentar")
st.write("Aplikasi ini merupakan implementasi model NLP menggunakan Logistic Regression")

input = st.text_area("Masukkan Komentar Anda!")

if st.button("Prediksi"):
    if input.strip() == "":
        st.warning("Komentar Tidak Boleh Kosong")
    else:
        vector = tfidf.transform([input])
        prediction = model.predict(vector)[0]
        
        label_mapping = {
            0: "Negatif",
            1: "Positif"
        }
        
        st.subheader("Hasil Analisis Komentar")
        st.write("**Sentiment Anda:**", label_mapping.get(prediction, prediction))