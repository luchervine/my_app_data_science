import table_files as tbf
import streamlit as st
import text_file as tf
import image_file as imgf
import audio_file as af

if __name__ == "__main__":
    sidebar = st.sidebar.radio(
        "Menu", ["Accueil", "Tabulaire", "Text", "Image", "Audio"])
    if sidebar == "Accueil":
        st.title("Bienvenue sur la page d'accueil")
    elif sidebar == "Tabulaire":
        tbf.main()
    elif sidebar == "Text":
        tf.main()
    elif sidebar == "Image":
        imgf.main()
    elif sidebar == "Audio":
        af.main()
