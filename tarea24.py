import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------------
st.set_page_config(page_title="K-Means Interactivo", layout="wide")
st.title("🎯 Tarea 2.4 Algoritmos de Busqueda")

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

        # Parámetros adicionales
        init_method = st.sidebar.selectbox("Método de inicialización (init):", ['k-means++', 'random'])
        max_iter = st.sidebar.number_input("Iteraciones máximas (max_iter):", 100, 1000, 300, step=50)
        n_init = st.sidebar.number_input("Número de inicializaciones (n_init):", 1, 50, 10, step=1)
        random_state = st.sidebar.number_input("Semilla aleatoria (random_state):", 0, 999, 42, step=1)

        # --------------------------------------------------
        # ESCALADO DE LOS DATOS
        # --------------------------------------------------
        X = data[selected_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --------------------------------------------------
        # MODELO K-MEANS
        # --------------------------------------------------
        kmeans = KMeans(
            n_clusters=k,
            init=init_method,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state
        )
        kmeans.fit(X_scaled)
        data['Cluster'] = kmeans.labels_

        # --------------------------------------------------
        # PCA PARA VISUALIZAR EN 2D O 3D
        # --------------------------------------------------
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        pca_cols = [f'PCA{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster']

        # --------------------------------------------------
        # VISUALIZACIÓN ANTES DE K-MEANS
        # --------------------------------------------------
        st.subheader("📊 Distribución original (antes de K-Means)")
        if n_components == 2:
            fig_before = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                title="Datos proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        else:
            fig_before = px.scatter_3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                title="Datos proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        st.plotly_chart(fig_before, use_container_width=True)

        # --------------------------------------------------
        # VISUALIZACIÓN DESPUÉS DE K-MEANS
        # --------------------------------------------------
        st.subheader(f"🎨 Clusters obtenidos con K-Means (k = {k})")

        if n_components == 2:
            fig_after = px.scatter(
                pca_df, x='PCA1', y='PCA2',
                color=pca_df['Cluster'].astype(str),
                title="Clusters en 2D con PCA (After K-Means)",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )

            # Agregar centroides
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            fig_after.add_scatter(
                x=centroids_pca[:, 0],
                y=centroids_pca[:, 1],
                mode='markers+text',
                text=[f'C{i}' for i in range(k)],
                textposition='top center',
                marker=dict(symbol='x', size=12, color='black'),
                name='Centroides'
            )

        else:
            fig_after = px.scatter_3d(
                pca_df, x='PCA1', y='PCA2', z='PCA3',
                color=pca_df['Cluster'].astype(str),
                title="Clusters en 3D con PCA (After K-Means)",
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
                km.fit(X_scaled)
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

