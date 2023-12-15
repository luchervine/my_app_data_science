import streamlit as st
import pandas as pd
import re

# Fonction pour charger un fichier texte en DataFrame


def load_text_file(file_path, separator=","):
    try:
        df = pd.read_csv(file_path, sep=separator)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier texte : {e}")
        return None

# Fonction pour manipuler un fichier texte avec la librairie "re"


def process_text_file(file_content, regex_pattern):
    try:
        # Convertir les octets en chaîne de caractères
        file_content_str = file_content.decode('utf-8')

        # Appliquer le motif regex au contenu du fichier
        matches = re.findall(regex_pattern, file_content_str)
        return matches
    except Exception as e:
        st.error(
            f"Erreur lors de la manipulation du fichier texte avec la librairie 're': {e}")
        return None

# Interface utilisateur Streamlit


def main():
    st.title("Manipulation de fichiers texte avec Pandas et 're' (regex)")

    # Charger le fichier texte
    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier texte (.txt, .csv, etc.)", type=["txt"])

    if uploaded_file is not None:
        # Afficher le contenu du fichier texte
        file_content = uploaded_file.read()
        btn_show = st.button("Afficher le texte")
        if btn_show:
            st.subheader("Contenu du fichier texte :")
            st.text(file_content.decode('utf-8'))

        # Ajouter un champ de texte pour saisir le motif regex
        regex_pattern = st.text_input("Entrez le motif regex :")

        # Appliquer le motif regex et afficher les correspondances
        if st.button("Appliquer le motif regex"):
            matches = process_text_file(file_content, regex_pattern)
            if matches is not None:
                st.subheader("Résultats de la manipulation avec 're' :")
                st.write(matches)


if __name__ == "__main__":
    main()
