import streamlit as st
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

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
        plt.clf()
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
        plt.xlabel("Temps (secondes)")
        plt.ylabel("Amplitude")
        plt.title("Signal audio")
        st.pyplot()

        # Appliquer une opération sur le signal audio
        st.subheader("Opération sur le signal audio")
        operation_type = st.selectbox("Choisissez une opération :", [
            "Aucune", "Normalisation", "Inversion", "Filtrage passe-bas", "Filtrage passe-haut", "Effets sonores"])

        if operation_type == "Normalisation":
            normalized_audio = normalize_audio(audio_data)
            st.audio(normalized_audio, format="audio/wav",
                     sample_rate=sample_rate)
            st.write("Audio normalisé : Signal audio après normalisation")

            # Afficher le graphique du signal audio normalisé
            plt.clf()
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(len(normalized_audio)) /
                     sample_rate, normalized_audio)
            plt.xlabel("Temps (secondes)")
            plt.ylabel("Amplitude")
            plt.title("Signal audio normalisé")
            st.pyplot()

            # Ajoutez une section pour l'analyse fréquentielle
            st.subheader("Analyse Fréquentielle")
            plot_fft(normalized_audio, "FFT du Signal Normalisé", sample_rate)

        elif operation_type == "Inversion":
            inverted_audio = invert_audio(audio_data)
            st.audio(inverted_audio, format="audio/wav",
                     sample_rate=sample_rate)
            st.write("Audio inversé : Signal audio après inversion")

            # Afficher le graphique du signal audio inversé
            plt.clf()
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(len(inverted_audio)) /
                     sample_rate, inverted_audio)
            plt.xlabel("Temps (secondes)")
            plt.ylabel("Amplitude")
            plt.title("Signal audio inversé")
            st.pyplot()

            # Ajoutez une section pour l'analyse fréquentielle
            st.subheader("Analyse Fréquentielle")
            plot_fft(inverted_audio, "FFT du Signal Inversé", sample_rate)

        elif operation_type == "Filtrage passe-bas" or operation_type == "Filtrage passe-haut":
            cutoff_frequency = st.slider(
                "Fréquence de coupure (Hz)", min_value=0, max_value=int(sample_rate / 2)-50, value=1000, step=100)
            # Appliquer le filtre
            filtered_audio = apply_filter(
                audio_data, sample_rate, operation_type, cutoff_frequency)

            st.audio(filtered_audio, format="audio/wav",
                     sample_rate=sample_rate)
            st.write(
                f"Audio filtré ({operation_type} à {cutoff_frequency} Hz): Signal audio après filtrage")

            # Afficher le graphique du signal audio filtré
            plt.clf()
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(len(filtered_audio)) /
                     sample_rate, filtered_audio)
            plt.xlabel("Temps (secondes)")
            plt.ylabel("Amplitude")
            plt.title(
                f"Signal audio filtré ({operation_type} à {cutoff_frequency} Hz)")
            st.pyplot()

            # Ajoutez une section pour l'analyse fréquentielle
            st.subheader("Analyse Fréquentielle")
            plot_fft(filtered_audio, "FFT du Signal " +
                     operation_type, sample_rate)

        elif operation_type == "Effets sonores":
            effect_type = st.selectbox("Choisissez un effet sonore :", [
                "Aucun effet", "Écho", "Réverbération", "Inversion de phase"])

            # Appliquer l'effet sonore
            if effect_type == "Écho":
                audio_data = apply_echo(audio_data, sample_rate)
            elif effect_type == "Réverbération":
                audio_data = apply_reverb(audio_data, sample_rate)
            elif effect_type == "Inversion de phase":
                audio_data = apply_phase_inversion(audio_data)

            # Afficher le signal audio après l'effet sonore
            st.audio(audio_data, format="audio/wav", sample_rate=sample_rate)
            st.write(f"Audio après l'effet sonore : {effect_type}")

            # Afficher le graphique du signal audio après l'effet sonore
            plt.clf()
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
            plt.xlabel("Temps (secondes)")
            plt.ylabel("Amplitude")
            plt.title(f"Signal audio après l'effet sonore : {effect_type}")
            st.pyplot()

            # section pour l'analyse fréquentielle
            st.subheader("Analyse Fréquentielle")
            plot_fft(audio_data, "FFT du Signal "+effect_type, sample_rate)

        st.subheader("Traitement du signal audio")
        traitement_type = st.selectbox("Choisissez une opération :", [
            "Aucune", "Analyse des pics d'amplitude", "Spectrogramme"])

        if traitement_type == "Spectrogramme":
            spectogram(audio_data, sample_rate)
        elif operation_type == "Analyse des pics d'amplitude":
            threshold = st.slider(
                "Seuil d'amplitude", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

            # Appliquer l'analyse des pics d'amplitude
            peaks, peak_values = find_amplitude_peaks(audio_data, threshold)

            # Afficher le signal avec les pics d'amplitude
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
            ax.plot(peaks / sample_rate,
                    peak_values, 'ro', label='Pics d\'amplitude')
            ax.set_xlabel("Temps (secondes)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Analyse des pics d'amplitude")
            ax.legend()

            # Afficher le graphique dans Streamlit
            st.pyplot(fig)

# Fonction pour normaliser le signal audio


def normalize_audio(audio_data):
    normalized_audio = audio_data / np.max(np.abs(audio_data))
    return normalized_audio

# Fonction pour inverser le signal audio


def invert_audio(audio_data):
    inverted_audio = -audio_data
    return inverted_audio


def spectogram(audio_data, sample_rate):
    # Afficher le spectrogramme du signal audio
    st.subheader("Spectrogramme du signal audio")
    frequencies, times, Sxx = spectrogram(audio_data, sample_rate)
    plt.clf()
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))
    plt.ylabel('Fréquence [Hz]')
    plt.xlabel('Temps [sec]')
    plt.title('Spectrogramme du signal audio')
    plt.colorbar(label='Puissance [dB]')
    st.pyplot()

# Fonction pour appliquer le filtre


def apply_filter(audio_data, sample_rate, filter_type, cutoff_frequency):
    nyquist_frequency = 0.5 * sample_rate
    normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency

    # Utiliser un filtre passe-bas Butterworth
    if filter_type == "Filtrage passe-bas":
        b, a = butter(N=6, Wn=normalized_cutoff_frequency,
                      btype='low', analog=False)

    # Utiliser un filtre passe-haut Butterworth
    elif filter_type == "Filtrage passe-haut":
        b, a = butter(N=6, Wn=normalized_cutoff_frequency,
                      btype='high', analog=False)

    # Appliquer le filtre au signal audio
    filtered_audio = lfilter(b, a, audio_data)
    return filtered_audio

# Fonction pour modifier la vitesse


def change_speed(audio_data, sample_rate, speed_factor):
    audio = AudioSegment(
        audio_data.tobytes(), frame_rate=sample_rate, sample_width=audio_data.dtype.itemsize, channels=1)
    modified_audio = audio.speedup(playback_speed=speed_factor)
    return np.array(modified_audio.get_array_of_samples())

# Fonction pour trouver les pics d'amplitude


def find_amplitude_peaks(audio_data, threshold):
    peaks, _ = find_peaks(np.abs(audio_data), height=threshold)
    return peaks, audio_data[peaks]

# Fonctions pour appliquer des effets sonores


def apply_echo(audio_data, sample_rate):
    sound = AudioSegment(np.int16(audio_data),
                         frame_rate=sample_rate, sample_width=2, channels=1)
    sound_with_echo = sound * 2  # Répéter le son pour créer un effet d'écho
    return np.array(sound_with_echo.get_array_of_samples())


def apply_reverb(audio_data, sample_rate):
    sound = AudioSegment(np.int16(audio_data),
                         frame_rate=sample_rate, sample_width=2, channels=1)
    # Inverser le son pour créer un effet de réverbération
    sound_with_reverb = sound.reverse()
    return np.array(sound_with_reverb.get_array_of_samples())


def apply_phase_inversion(audio_data):
    return -audio_data

# Ajoutez une fonction pour l'analyse fréquentielle


def plot_fft(data, title, fs):
    fft_values = fft(data)
    freqs = fftfreq(len(fft_values), 1/fs)

    plt.clf()
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, np.abs(fft_values))
    plt.title(title)
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    # Définir les limites de l'axe des x
    # plt.xlim([-20000, 20000])
    st.pyplot()


# Appeler la fonction principale
if __name__ == "__main__":
    main()
