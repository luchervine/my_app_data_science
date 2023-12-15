import streamlit as st
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Interface utilisateur Streamlit pour la manipulation des fichiers audio avec SciPy


def main():
    st.title("Manipulation des fichiers audio avec SciPy")

    # Charger un fichier audio depuis l'ordinateur
    uploaded_audio = st.file_uploader(
        "Sélectionnez un fichier audio (.wav)", type=["wav"])

    if uploaded_audio is not None:
        # Lire le fichier audio avec SciPy
        sample_rate, audio_data = wavfile.read(uploaded_audio)

        # Afficher les informations sur le fichier audio
        st.subheader("Informations sur le fichier audio")
        st.write(f"Taux d'échantillonnage : {sample_rate} Hz")
        st.write(f"Durée : {len(audio_data) / sample_rate} secondes")
        st.write(
            f"Nombre de canaux : {1 if len(audio_data.shape) == 1 else audio_data.shape[1]}")

        # Afficher le signal audio
        st.subheader("Signal audio")
        st.audio(uploaded_audio, format="audio/wav")

        # Afficher le graphique du signal audio
        st.subheader("Graphique du signal audio")
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
        plt.xlabel("Temps (secondes)")
        plt.ylabel("Amplitude")
        plt.title("Signal audio")
        st.pyplot()

        # Appliquer une opération sur le signal audio
        st.subheader("Opération sur le signal audio")
        operation_type = st.selectbox("Choisissez une opération :", [
                                      "Aucune", "Normalisation", "Inversion"])

        if operation_type == "Normalisation":
            normalized_audio = normalize_audio(audio_data)
            st.audio(normalized_audio, format="audio/wav",
                     sample_rate=sample_rate)
            st.write("Audio normalisé : Signal audio après normalisation")
        elif operation_type == "Inversion":
            inverted_audio = invert_audio(audio_data)
            st.audio(inverted_audio, format="audio/wav",
                     sample_rate=sample_rate)
            st.write("Audio inversé : Signal audio après inversion")

# Fonction pour normaliser le signal audio


def normalize_audio(audio_data):
    normalized_audio = audio_data / np.max(np.abs(audio_data))
    return normalized_audio

# Fonction pour inverser le signal audio


def invert_audio(audio_data):
    inverted_audio = -audio_data
    return inverted_audio


# Appeler la fonction principale
if __name__ == "__main__":
    main()
