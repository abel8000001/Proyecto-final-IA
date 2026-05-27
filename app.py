# Importación de librerías

import streamlit as st
import pandas as pd
import plotly.express as px


# Configuración de la página

st.set_page_config(
    page_title="Word Embeddings Visualization",
    layout="wide"
)

# Título y descripción

st.title("Visualización de Word Embeddings")
st.subheader("PCA y t-SNE sobre reseñas cinematográficas IMDb")

st.markdown("""
### Autor: Abel Santiago Garcia Ortiz

### Tecnologías utilizadas

- Streamlit
- Gensim
- Scikit-learn
- Plotly
- PCA
- t-SNE
""")

st.markdown("""
Este sistema inteligente utiliza técnicas de reducción de dimensionalidad
para visualizar relaciones semánticas entre palabras utilizando embeddings preentrenados GloVe.

El proyecto compara:

- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)

Los datos fueron obtenidos a partir de reseñas cinematográficas del dataset IMDb.
""")


# Carga de datos

@st.cache_data
def load_data():

    pca_df = pd.read_csv("pca_results.csv")
    tsne_df = pd.read_csv("tsne_results.csv")

    return pca_df, tsne_df


pca_df, tsne_df = load_data()


# Sidebar para opciones de visualización

st.sidebar.title("Opciones")

visualization_method = st.sidebar.selectbox(
    "Seleccionar método",
    ["PCA", "t-SNE"]
)


selected_cluster = st.sidebar.multiselect(
    "Filtrar clusters",
    options=sorted(tsne_df["cluster"].unique()),
    default=sorted(tsne_df["cluster"].unique())
)


# Filtrado de datos según selección

if visualization_method == "PCA":

    filtered_df = pca_df[
        pca_df["cluster"].isin(selected_cluster)
    ]

else:

    filtered_df = tsne_df[
        tsne_df["cluster"].isin(selected_cluster)
    ]


# Visualización de PCA o t-SNE según selección

if visualization_method == "PCA":

    fig = px.scatter(
        filtered_df,
        x="PCA_1",
        y="PCA_2",
        color=filtered_df["cluster"].astype(str),
        hover_name="word",
        title="Visualización PCA",
        width=1200,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)
else:

    fig = px.scatter(
        filtered_df,
        x="TSNE_1",
        y="TSNE_2",
        color=filtered_df["cluster"].astype(str),
        hover_name="word",
        title="Visualización t-SNE",
        width=1200,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)


# Tabla de datos

st.subheader("Datos utilizados")

st.dataframe(filtered_df)


# Busqueda de palabras

st.subheader("Buscar palabra")

search_word = st.text_input(
    "Ingrese una palabra"
)


if search_word:

    if visualization_method == "PCA":

        results = pca_df[
            pca_df["word"].str.contains(
                search_word,
                case=False
            )
        ]

    else:

        results = tsne_df[
            tsne_df["word"].str.contains(
                search_word,
                case=False
            )
        ]

    st.dataframe(results)