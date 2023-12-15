import streamlit as st
import pandas as pd
from streamlit_modal import Modal
from io import StringIO
import contextlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, precision_score


modal = Modal(key="len_modal", title="Nombre d'éléments")

# Fonction pour charger un fichier Tabulaire en DataFrame


def load_text_file(file_path, separator=","):
    try:
        df = pd.read_csv(file_path, sep=separator)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier Tabulaire : {e}")
        return None


# Fonction pour afficher les premières lignes d'un DataFrame
def display_head(df, n=5):
    st.write(df.head(n))

# Fonction pour afficher les dernières lignes d'un DataFrame


def display_tail(df, n=5):
    st.write(df.tail(n))


def display_sample(df, nb=5):
    st.write(df.sample(n=nb))

# Fonction pour afficher les lignes en fonction du choix de l'utilisateur


def display_rows(df, n, display_option):
    if display_option == 'Premières lignes':
        display_head(df, n)
    elif display_option == 'Dernières lignes':
        display_tail(df, n)
    elif display_option == 'Un échantillon':
        display_sample(df, n)


@st.cache_data
def get_info(df):
    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        df.info()
    return buffer.getvalue()


def data_info(df):
    st.write("Informations de base sur les données :")
    # Afficher les informations obtenues avec df.info()
    with st.expander("Afficher les informations détaillées"):
        st.text(get_info(df))
    st.write("\nStatistiques descriptives :")
    st.write(df.describe())

# Interface utilisateur Streamlit


def main():
    uploaded_file = ''
    st.title("Manipulation de fichiers Tabulaire avec Pandas")

    # Ajouter une option pour lire le fichier à partir d'une URL
    file_location = st.radio("Sélectionnez l'emplacement du fichier :", [
                             "Télécharger un fichier", "Utiliser une URL"])

    if file_location == "Télécharger un fichier":
        # Chargement du fichier texte à partir de l'upload
        uploaded_file = st.file_uploader(
            "Sélectionnez un fichier tabulaire (.csv, .xlsx)", type=["txt", "csv", "xlsx"])
    else:
        # Saisir l'URL du fichier
        uploaded_file = st.text_input(
            "Entrez l'URL du fichier (par exemple, un fichier CSV sur le web):")

    if uploaded_file is not None and uploaded_file != '':
        separator = st.text_input("Séparateur (par défaut : ,)", ",")

        # Charger les données dans un DataFrame
        data = load_text_file(uploaded_file, separator)

        if data is not None:
            # Ajouter un bouton pour afficher le modal
            if st.button("Afficher le nombre d'éléments"):
                st.info(
                    f"Le nombre d'éléments dans le DataFrame est : {len(data)}")

            # Ajouter des boutons radio pour choisir entre les premières et les dernières lignes
            row_display_option, slider_col = st.columns([1, 3])
            with row_display_option:
                display_option = st.radio(
                    "Afficher :", ['Premières lignes', 'Dernières lignes', 'Utiliser loc/iloc'])  # ,'Un échantillion', ])

            with slider_col:
                # Ajouter un widget interactif (slider) pour le nombre de lignes à afficher
                selected_rows = st.slider(
                    "Sélectionnez un nombre de lignes :", 1, 100, 5)
                if display_option == 'Utiliser loc/iloc':
                    # Ajouter des champs de texte pour spécifier loc ou iloc
                    st.write(
                        "Utiliser loc ou iloc pour afficher une plage d'indices :")
                    loc_iloc_selector = st.radio("", ['loc', 'iloc'])
                    loc_iloc_input = st.text_input(
                        f"Entrez les indices des lignes  à afficher avec {loc_iloc_selector} (séparés par des virgules):")
                    start_col, end_col = st.columns(2)
                    with start_col:
                        loc_iloc_start = st.text_input(
                            f"Index de début avec {loc_iloc_selector}:")
                    with end_col:
                        loc_iloc_end = st.text_input(
                            f"Index de fin avec {loc_iloc_selector}:")
                    if display_option == 'Utiliser loc/iloc':
                        if loc_iloc_input:
                            try:
                                loc_iloc_input = [int(i.strip())
                                                  for i in loc_iloc_input.split(",")]
                                if loc_iloc_selector == 'loc':
                                    st.write(data.loc[loc_iloc_input])
                                else:
                                    st.write(data.iloc[loc_iloc_input])
                            except Exception as e:
                                st.warning(
                                    f"Erreur lors de l'utilisation de loc/iloc : {e}")
                        else:
                            try:
                                start = int(loc_iloc_start.strip())
                                end = int(loc_iloc_end.strip())
                                if loc_iloc_selector == 'loc':
                                    st.write(data.loc[start:end])
                                else:
                                    st.write(data.iloc[start:end])
                            except Exception as e:
                                st.warning(
                                    f"Erreur lors de l'utilisation de loc/iloc : {e}")

            # Afficher les lignes en fonction du choix de l'utilisateur
            st.subheader(
                f"Affichage des {display_option.lower()} du DataFrame :")
            if display_option == 'Premières lignes':
                display_head(data, selected_rows)
            elif display_option == 'Dernières lignes':
                display_tail(data, selected_rows)
            elif display_option == 'Un échantillion':
                data.sample(data, selected_rows)

            # Afficher des informations de base sur le DataFrame
            st.subheader("Informations sur les données :")
            st.write("Colonnes disponibles :")
            st.table([list(data.columns)])
            data_info(data)

            # Sélectionner les colonnes que vous souhaitez afficher
            selected_columns = st.multiselect(
                "Sélectionnez les colonnes à afficher:", data.columns)

            # Créer un nouveau DataFrame avec les colonnes sélectionnées
            new_df = data[selected_columns]

            # Afficher le nouveau DataFrame
            st.write("Nouveau DataFrame avec les colonnes sélectionnées:")
            st.write(new_df)
            old_data = data
            data = new_df

            row_display_filter, filter_col = st.columns([1, 3])
            with row_display_filter:
                display_filter = st.radio(
                    "Filtrer par :", ['Colonne', 'Condition', 'Groupby'])
            with filter_col:
                # Ajouter des widgets interactifs pour permettre à l'utilisateur de définir des filtres
                st.subheader("Filtrer les données :")

                if display_filter == 'Colonne':
                    # Exemple avec un filtre par colonne (utiliser st.multiselect pour permettre la sélection de plusieurs valeurs)
                    try:
                        selected_column = st.selectbox(
                            "Sélectionnez une colonne pour le filtre :", data.columns)
                        filter_value = st.text_input(
                            "Entrez la valeur à filtrer :")
                        filtered_data = data[data[selected_column]
                                             == filter_value]
                        # Afficher les données filtrées
                        st.subheader("Données filtrées :")
                        st.write(filtered_data)
                    except Exception as e:
                        st.warning(
                            f"Erreur lors de l'utilisation du filtre : {e}")
                elif display_filter == 'Condition':
                    try:
                        filter_value = st.text_input(
                            "Entrez la condition de filtre (Ex: Colonne1 > 150') :")
                        filtered_data = data.query(filter_value)
                        # Afficher les données filtrées
                        st.subheader("Données filtrées :")
                        st.write(filtered_data)
                    except Exception as e:
                        st.warning(
                            f"Erreur lors de l'utilisation du filtre : {e}")
                if display_filter == 'Groupby':
                    # Ajouter des widgets interactifs pour permettre à l'utilisateur de définir un groupby
                    st.subheader("Groupby et agrégation :")

                    # Sélectionner une colonne pour le groupby
                    groupby_column = st.selectbox(
                        "Sélectionnez une colonne pour le groupby :", data.columns)

                    # Sélectionner une ou plusieurs colonnes pour le groupby
                    groupby_columns = st.multiselect(
                        "Sélectionnez une ou plusieurs colonnes pour le groupby :", data.columns)

                    if groupby_columns:
                        # Sélectionner une fonction d'agrégation
                        aggregation_function = st.selectbox("Sélectionnez une fonction d'agrégation :", [
                                                            'sum', 'mean', 'min', 'max', 'count'])

                        # Appliquer le groupby et l'agrégation
                        grouped_data = data.groupby(
                            groupby_column)[groupby_columns].agg(aggregation_function)

                        # Afficher les données groupées
                        st.subheader("Données groupées :")
                        st.write(grouped_data)
                    else:
                        st.warning(
                            "Veuillez sélectionner au moins une colonne pour le groupby.")

            model = st.radio("Selectionnez le modèle de traitement : ", [
                             "Regression Linéaire", "Réseau Neuronal", "Classification NB"], horizontal=True)
            if model == "Regression Linéaire":
                linear_reg(old_data, selected_columns)
            elif model == "Réseau Neuronal":
                nnetwork(old_data, selected_columns)
            elif model == "Classification NB":
                class_nb(old_data[selected_columns], selected_columns)


def linear_reg(df, X):
    # Sélectionner y
    y_selected = st.selectbox(
        "Sélectionnez les colonnes à afficher:", df.columns, key="y_col")
    X = df[X]
    y = df[y_selected]
    col_x, col_y = st.columns(2)
    with col_x:
        st.write("X")
        st.write(X)
    with col_y:
        st.write("y")
        st.write(y)

    '''plt.scatter(X, y, s=20)
    plt.grid()
    plt.xlabel('x_1')
    plt.ylabel('y')
    st.pyplot()'''

    # Division des Données
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    # Paramètres du modèle: intercept=theta0 and coef=les autres theta
    lr.intercept_, lr.coef_

    st.write("Shape x_test")
    st.write(x_test.shape)

    st.write('y_pred')
    y_pred = lr.predict(x_test)
    st.write(y_pred)

    err = (y_pred-y_test)**2
    err_m = err.mean()
    err_m
    st.write("Score")
    st.write(lr.score(x_test, y_test))

    # Tracer le nuage de points
    plt.scatter(x_test, y_test, s=20, label='Actual')
    plt.scatter(x_test, y_pred, s=20, label='Predicted')
    plt.grid()
    plt.xlabel('x_1')
    plt.ylabel('y')
    plt.legend()
    st.pyplot()


def nnetwork(df, X):
    st.write("Valeurs manquantes :")
    st.write(df.isnull().sum())
    # Sélectionner y
    y = st.selectbox(
        "Sélectionnez la colonne à prédire:", df.columns, key="y_col")
    target_column = y
    X = df[X]
    y = df[y]
    col_x, col_y = st.columns(2)
    with col_x:
        st.write("X")
        st.write(X)
    with col_y:
        st.write("y")
        st.write(y)

    # Répartition des données
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    st.write("Shape x_train")
    st.write(x_train.shape)
    st.write('y_test')
    st.write(y_test)

    # st.subheader("Normalisation des données")
    scaler = MinMaxScaler()
    # Pour X
    x_train_scal = scaler.fit_transform(x_train)
    x_test_scal = scaler.fit_transform(x_test)

    # Pour y
    y_train_scal = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scal = scaler.fit_transform(y_test.values.reshape(-1, 1))

    st.title("Créer le réseau de neuronnes..")
    model = Sequential()
    st.subheader("Ajout de couches :")
    st.info(
        "Nous utilisons 3 couches avec la fonction d'actiation 'Relu' pour l'instant..", icon="❕")
    model.add(Dense(5, input_dim=5, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='relu'))
    st.write(model.summary())

    # définition de fonction coût, algorithme d'optimisation et les métriques
    myloss = tf.keras.losses.MeanSquaredError(
        reduction="auto", name="mean_squared_error")
    optim = tf.keras.optimizers.Adam(learning_rate=0.01)  # SGD
    metrics = ['mse', 'mae']

    # Compilation du modèle
    model.compile(optimizer=optim, loss=myloss, metrics=metrics)

    st.subheader("Entrainement : ")
    history = model.fit(x_train_scal, y_train_scal,
                        validation_split=0.2, epochs=20)
    st.write(history)

    st.write("## Evaluation")
    # ploss & validation loss
    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.grid()
    plt.legend()
    st.pyplot(plt)

    true_w = tf.constant([4.2, -3.5])
    true_b = tf.constant([5.])

    model.get_weights()
    w = model.get_weights()[0]
    print('Parameters: ')
    tf.print(w)
    b = model.get_weights()[1]
    print('Bias: ')
    tf.print(b)

    st.write('Score')
    scores = model.evaluate(x_test_scal, y_test_scal, verbose=0)
    st.info("Mean Squadred Error : " + str(scores[1]))
    st.info("Mean Absolute Error : " + str(scores[2]))

    Y_pred = model.predict(x_test_scal)
    # st.write(f"Y_pred = {Y_pred}")
    st.write('r2 score: ', r2_score(x_test_scal, Y_pred))

    Y_pred = model.predict(x_test_scal).flatten()
    plt.scatter(y_test_scal, Y_pred)
    st.pyplot(plt)


def class_nb(df, X):
    # Visualiser les distributions et les  nuages (utiliser pairplot de seaborn)
    sns.pairplot(df)
    pairplot_fig = plt.gcf()  # Récupérer la figure actuelle (figure 1)
    st.pyplot(pairplot_fig)

    # st.subheader("# Calcul de la corrélation")
    # Visualiser les distributions et les nuages (utiliser pairplot de seaborn)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # Calcul de la corrélation
    cor = df.corr()

    # Visualisation de la correlation (heatmap)
    st.subheader("## Visualisation de la correlation (heatmap)")
    sns.heatmap(cor, vmax=1, annot=True, cmap="Greens", ax=axes[1])
    axes[1].set_title('Correlation Heatmap')  # Titre de la figure 2

    st.pyplot(fig)

    st.subheader("## analyse de la corrélation")
    st.write("GArder les colonnes no corellées")
    df2 = st.multiselect(
        "Sélectionez les colonnes non correlées :", df.columns)
    df2 = df[df2]
    st.write(df2.head())

    # élaboration de X et y
    y = st.selectbox(
        "Sélectionnez la colonne à prédire:", df.columns, key="y_col")
    X = df[X]
    y = df[y]
    y.value_counts()

    # Découper dataset en 20% pour test et 80% pour traning
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

    # Training en utilisant Naive Bayes
    try:
        nbg = GaussianNB()
        nbg.fit(x_train, y_train)

        st.write("## Evaluation du modèle")
        y_pred = nbg.predict(x_test)
        cfx = confusion_matrix(y_test, y_pred)
        st.info(cfx)
    except ValueError as e:
        st.warning("Une erreur s'est produite : \nLe modèle Naive Bayes est généralement utilisé pour des problèmes de classification, où la variable à prédire est catégorique. Si votre tâche est une régression (prédiction d'une variable continue), vous devriez utiliser un modèle adapté à cette tâche. \nSi vous avez l'intention de résoudre un problème de régression, envisagez d'utiliser un autre algorithme, tel que la régression linéaire ou une forêt aléatoire, adapté aux valeurs continues.")
