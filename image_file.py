import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import numpy as np

# Désactiver l'avertissement lié à st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Fonction pour appliquer les filtres


def apply_filter(image, filter_type):
    if filter_type == "Flou":
        return cv.GaussianBlur(image, (15, 15), 0)
    elif filter_type == "Détection de bord":
        return cv.Canny(image, 50, 150)
    elif filter_type == "Clarification des bords":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv.filter2D(image, -1, kernel)
    elif filter_type == "Rognage":
        return image[50:150, 50:150, :]
    elif filter_type == "Relief":
        # Emboss
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        return cv.filter2D(image, -1, kernel)
    elif filter_type == "Accentuation":
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv.filter2D(image, -1, kernel)
    elif filter_type == "Contours":
        # Outline
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        return cv.filter2D(image, -1, kernel)

    else:
        return image  # Pas de filtre


# Translation


def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

 # Rotation


def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2, height//2)  # centre

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)

# Fonction pour redimensionner l'image


def resize_image(image, scale_percent):
    # Calculer les nouvelles dimensions
    width = int(image.shape[1] * (scale_percent / 100))
    height = int(image.shape[0] * (scale_percent / 100))
    dimensions = (width, height)
    # Redimensionner l'image
    resized_image = cv.resize(image, dimensions, interpolation=cv.INTER_AREA)
    return resized_image

# Fonction pour obtenir le code de retournement en fonction du choix de l'utilisateur


def get_flip_code(flip_type):
    if flip_type == "Horizontal":
        return 1
    elif flip_type == "Vertical":
        return 0
    elif flip_type == "Les deux":
        return -1
    else:
        return 0  # Aucun retournement

# Interface utilisateur Streamlit pour la manipulation des images avec Matplotlib


def main():
    st.title("Manipulation des images avec Matplotlib")

    # Charger une image depuis l'ordinateur
    uploaded_image = st.file_uploader(
        "Sélectionnez une image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_image is not None:
        # Afficher l'image
        image = plt.imread(uploaded_image)
        st.image(image, caption="Image originale", use_column_width=True)

        # Manipuler l'image avec Matplotlib
        st.subheader("Manipulation de l'image avec Matplotlib")
        btn_shape = st.button("Shape")
        if btn_shape:
            btn_shape,
            st.info(image.shape)

        channel_col, button_col = st.columns(2)
        # Boutons pour choisir le canal de couleur
        selected_channel = channel_col.radio(
            "Choisissez le canal de couleur :", ["Rouge", "Gris", "Originale"], horizontal=True)

        # Widget pour redimensionner l'image
        new_size = st.slider(
            "Redimensionner l'image (% de la taille d'origine):", 1, 200, 100)

        # Redimensionner l'image
        resized_image = Image.fromarray(image[:, :, 0])
        new_width = int(image.shape[1] * (new_size / 100))
        new_height = int(image.shape[0] * (new_size / 100))
        resized_image = resized_image.resize((new_width, new_height))

        fig, ax = plt.subplots()
        # Sélectionner le canal de couleur correspondant
        if selected_channel == "Rouge":
            ax.imshow(resized_image)
        elif selected_channel == "Gris":
            ax.imshow(resized_image, cmap='gray')
        elif selected_channel == "Originale":
            resized_image = Image.fromarray(image)
            resized_image = resized_image.resize((new_width, new_height))
            ax.imshow(resized_image)

        ax.axis("off")  # Masquer les axes

        # Passer explicitement la figure à st.pyplot()
        st.pyplot(fig)

        # Ajouter des filtres
        st.subheader("Filtres")
        filter_type = st.selectbox("Choisissez le filtre :", [
                                   "Aucun", "Flou", "Détection de bord", "Clarification des bords", "Rognage", "Relief", "Accentuation", "Contours"])

        if filter_type != "Aucun":
            # Convertir l'image en tableau NumPy
            img_array = np.array(image)

            # Appliquer le filtre sélectionné
            filtered_image = apply_filter(img_array, filter_type)

            # Afficher l'image filtrée
            fig_filter, ax_filter = plt.subplots()
            ax_filter.imshow(filtered_image)
            ax_filter.axis("off")
            st.pyplot(fig_filter)

        # Ajouter la translation
        st.subheader("Translation")
        translate_x = st.slider("Translation X:", -100, 100, 0)
        translate_y = st.slider("Translation Y:", -100, 100, 0)
        # Ajouter la rotation
        st.subheader("Rotation")
        rotation_angle = st.slider("Angle de rotation:", -180, 180, 0)

        # Appliquer la translation
        translated_image = translate(
            image, translate_x, translate_y)
        # Appliquer la rotation
        rotated_image = rotate(translated_image, rotation_angle)

        # Afficher l'image avec rotation
        fig_rotate, ax_rotate = plt.subplots()
        ax_rotate.imshow(rotated_image)
        ax_rotate.axis("off")
        st.pyplot(fig_rotate)

        # Ajouter le retournement
        st.subheader("Retournement")
        flip_type = st.radio("Choisissez le type de retournement :", [
                             "Horizontal", "Vertical", "Les deux"], horizontal=True)

        # Appliquer le retournement
        flip_code = get_flip_code(flip_type)
        flipped_image = cv.flip(image, flip_code)

        # Afficher l'image avec retournement
        fig_flip, ax_flip = plt.subplots()
        ax_flip.imshow(flipped_image)
        ax_flip.axis("off")
        st.pyplot(fig_flip)

        # Ajouter les cannaux
        st.subheader("Cannaux de couleurs")
        flip_type = st.radio("Choisissez le cannal :", [
                             "bgr", "gris", "Les deux"], horizontal=True)
        cannal_bgr(image, flip_type)

    if st.button("Detect visage"):
        image_faces = detect_faces(image)
        st.image(image_faces, caption="Visage(s) détecté(s) ",
                 use_column_width=True)


# Appeler la fonction principale
if __name__ == "__main__":
    main()


def cannal_bgr(image, cannal):
    # Diviser l'image en canaux B, G, R
    b, g, r = cv.split(image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Afficher les histogrammes
    st.subheader("Histogrammes des canaux")
    fig_hist, ax_hist = plt.subplots()

    # Calculer les histogrammes pour chaque canal
    hist_b = cv.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv.calcHist([r], [0], None, [256], [0, 256])
    hist_gray = cv.calcHist([gray], [0], None, [256], [0, 256])

    # Afficher les histogrammes
    if cannal == "bgr":
        ax_hist.plot(hist_b, color='b', label='Canal Bleu')
        ax_hist.plot(hist_g, color='g', label='Canal Vert')
        ax_hist.plot(hist_r, color='r', label='Canal Rouge')
        ax_hist.set_title('Histogrammes des canaux B, G, R')
    elif cannal == "gris":
        ax_hist.plot(hist_gray, color='gray', label='Canal Gris')
        ax_hist.set_title('Histogrammes du cannal Gris')
    elif cannal == "Les deux":
        ax_hist.plot(hist_b, color='b', label='Canal Bleu')
        ax_hist.plot(hist_g, color='g', label='Canal Vert')
        ax_hist.plot(hist_r, color='r', label='Canal Rouge')
        ax_hist.plot(hist_gray, color='gray', label='Canal Gris')
        ax_hist.set_title('Histogrammes des canaux')

    # Ajouter une légende et des étiquettes
    ax_hist.legend()
    ax_hist.set_xlabel('Valeur des pixels')
    ax_hist.set_ylabel('Nombre de pixels')

    # Passer explicitement la figure à st.pyplot()
    st.pyplot(fig_hist)

# Fonction pour détecter les visages


def detect_faces(image):
    # Convertir l'image en niveaux de gris
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Charger le fichier XML pour la détection de visages
    fname = "haarcascade_frontalface_default.xml"
    haar_cascade = cv.CascadeClassifier(fname)

    # Détecter les visages dans l'image
    faces_rect = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces_rect:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    return image
