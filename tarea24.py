import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------------
st.set_page_config(page_title="K-Means Interactivo", layout="wide")
st.title("🎯 K-Means Interactivo con Parámetros Personalizables")
st.write("""
Esta aplicación permite realizar **Clustering con K-Means** y visualizar los resultados usando **PCA (2D o 3D)**.  
Puedes ajustar parámetros del modelo y comparar la distribución de los datos **antes y después** del agrupamiento.
""")

# --------------------------------------------------
# SUBIR ARCHIVO CSV
# --------------------------------------------------
st.sidebar.header("📂 Subir archivo de datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Seleccionar columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ Se necesitan al menos dos columnas numéricas para aplicar K-Means.")
    else:
        # --------------------------------------------------
        # CONFIGURACIÓN DEL MODELO EN LA SIDEBAR
        # --------------------------------------------------
        st.sidebar.header("⚙️ Configuración del modelo K-Means")

        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )

        k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)
        n_components = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

        # --- NUEVOS PARÁMETROS SEGÚN INSTRUCCIONES ---
        init_method = st.sidebar.selectbox(
            "Método de inicialización (init):",
            ['k-means++', 'random']
        )
        max_iter = st.sidebar.number_input(
            "Número máximo de iteraciones (max_iter):",
            min_value=100, max_value=1000, value=300, step=50
        )
        n_init = st.sidebar.number_input(
            "Número de inicializaciones (n_init):",
            min_value=1, max_value=50, value=10, step=1
        )
        random_state = st.sidebar.number_input(
            "Semilla aleatoria (random_state):",
            min_value=0, max_value=999, value=42, step=1
        )

        # --------------------------------------------------
        # APLICAR K-MEANS
        # --------------------------------------------------
        X = data[selected_cols]
        kmeans = KMeans(
            n_clusters=k,
            init=init_method,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state
        )
        kmeans.fit(X)
        data['Cluster'] = kmeans.labels_

        # --------------------------------------------------
        # PCA PARA VISUALIZAR EN 2D/3D
        # --------------------------------------------------
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        pca_cols = [f'PCA{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster']

        # --------------------------------------------------
        # VISUALIZACIÓN ANTES Y DESPUÉS
        # --------------------------------------------------
        st.subheader("📊 Distribución original (antes de K-Means)")
        if n_components == 2:
            fig_before = px.scatter(
                pca_df, x='PCA1', y='PCA2',
                title="Datos originales (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        else:
            fig_before = px.scatter_3d(
                pca_df, x='PCA1', y='PCA2', z='PCA3',
                title="Datos originales (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        st.plotly_chart(fig_before, use_container_width=True)

        st.subheader(f"🎨 Clusters obtenidos con K-Means (k = {k})")
        if n_components == 2:
            fig_after = px.scatter(
                pca_df, x='PCA1', y='PCA2',
                color=pca_df['Cluster'].astype(str),
                title="Clusters en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            fig_after = px.scatter_3d(
                pca_df, x='PCA1', y='PCA2', z='PCA3',
                color=pca_df['Cluster'].astype(str),
                title="Clusters en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # --------------------------------------------------
        # CENTROIDES
        # --------------------------------------------------
        st.subheader("📍 Centroides de los clusters (en espacio PCA)")
        centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
        st.dataframe(centroides_pca)

        # --------------------------------------------------
        # MÉTODO DEL CODO
        # --------------------------------------------------
        st.subheader("📈 Método del Codo (Elbow Method)")
        if st.button("Calcular número óptimo de clusters"):
            inertias = []
            K = range(1, 11)
            for i in K:
                km = KMeans(n_clusters=i, random_state=random_state)
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            plt.plot(K, inertias, 'bo-')
            plt.title('Método del Codo')
            plt.xlabel('Número de Clusters (k)')
            plt.ylabel('Inercia (SSE)')
            plt.grid(True)
            st.pyplot(fig2)

        # --------------------------------------------------
        # DESCARGA DE RESULTADOS
        # --------------------------------------------------
        st.subheader("💾 Descargar resultados")
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="⬇️ Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Carga un archivo CSV en la barra lateral para comenzar.")
    st.write("""
    **Ejemplo de formato:**
    | Ingreso_Anual | Gasto_Tienda | Edad |
    |----------------|--------------|------|
    | 45000 | 350 | 28 |
    | 72000 | 680 | 35 |
    | 28000 | 210 | 22 |
    """)
