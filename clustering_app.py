import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import plotly.graph_objects as go
import plotly.express as px
import io
from typing import Tuple, Optional, Union, List

# --- Helper Functions ---
def normalize_data(data: np.ndarray, method: str) -> Tuple[np.ndarray, Optional[object]]:
    """Normalize data using Min-Max Scaling or Z-Score.

    Args:
        data: Input data as a numpy array.
        method: Normalization method ('Min-Max Scaling' or 'Z-Score').

    Returns:
        Tuple of (normalized data, scaler object).
    """
    if method == "Min-Max Scaling":
        scaler = MinMaxScaler()
        return scaler.fit_transform(data), scaler
    elif method == "Z-Score":
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler
    return data, None

def calculate_silhouette_score(data: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Calculate the average Silhouette Score to evaluate clustering quality.

    Args:
        data: Input data as a numpy array.
        labels: Cluster labels for each data point.

    Returns:
        Silhouette Score (float) or None if computation fails.
    """
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:  # Requires at least 2 clusters
            return silhouette_score(data, labels) # type: ignore
        return None
    except Exception as e:
        st.error(f"[Error] Gagal menghitung Silhouette Score: {str(e)}")  # type: ignore
        return None

def calculate_davies_bouldin_score(data: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Calculate the Davies-Bouldin Index to evaluate clustering quality.

    Args:
        data: Input data as a numpy array.
        labels: Cluster labels for each data point.

    Returns:
        Davies-Bouldin Index (float) or None if computation fails.
    """
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:  # Requires at least 2 clusters
            return davies_bouldin_score(data, labels)
        return None
    except Exception as e:
        st.error(f"[Error] Gagal menghitung Davies-Bouldin Index: {str(e)}")  # type: ignore
        return None

def analisis_kmeans(data: np.ndarray, labels: np.ndarray, model: KMeans):
    """Perform K-Means analysis and provide cluster distribution, evaluation metrics, and outlier details.

    Args:
        data: Input data as a numpy array.
        labels: Cluster labels from K-Means.
        model: Fitted K-Means model.

    Returns:
        None (displays results in Streamlit).
    """
    n_samples = len(data)

    # 1. Distribusi klaster
    cluster_counts = np.bincount(labels, minlength=model.n_clusters) # type: ignore
    st.write(f"\n**Distribusi Klaster dengan {n_samples} sampel:**")
    dist_data = []
    for i, count in enumerate(cluster_counts):
        percentage = (count / n_samples) * 100
        st.write(f"**Klaster {i}:** {count} sampel ({percentage:.2f}%)")
        dist_data.append({"Klaster": f"Klaster {i}", "Jumlah Sampel": count, "Persentase (%)": percentage})
    st.table(pd.DataFrame(dist_data))

    # 2. Metrik evaluasi
    inertia = model.inertia_
    silhouette = silhouette_score(data, labels)
    db_index = davies_bouldin_score(data, labels)
    st.write("\n**Metrik Evaluasi Klaster:**")
    metrics_data = [
        {"Metrik": "Inertia (Within-Cluster Sum of Squares)", "Nilai": f"{inertia:.3f}"},
        {"Metrik": "Silhouette Score", "Nilai": f"{silhouette:.3f}"},
        {"Metrik": "Davies-Bouldin Index", "Nilai": f"{db_index:.3f}"}
    ]
    st.table(pd.DataFrame(metrics_data))

    # 3. Outlier analysis
    distances = np.min(model.transform(data), axis=1)
    threshold = np.percentile(distances, 95)
    outliers = np.where(distances > threshold)[0]
    outlier_percentage = (len(outliers) / n_samples) * 100
    st.write("\n**Outlier Analysis:**")
    outlier_summary = [
        {"Metrik": "Jumlah Outlier", "Nilai": f"{len(outliers)} sampel"},
        {"Metrik": "Persentase Outlier", "Nilai": f"{outlier_percentage:.2f}%"}
    ]
    st.table(pd.DataFrame(outlier_summary))

    # 4. Distribusi outlier per klaster
    outlier_clusters = labels[outliers]
    outlier_counts = np.bincount(outlier_clusters, minlength=model.n_clusters) # type: ignore
    st.write("\n**Distribusi Outlier per Klaster:**")
    outlier_dist_data = []
    for i, count in enumerate(outlier_counts):
        percentage = (count / cluster_counts[i]) * 100 if cluster_counts[i] > 0 else 0
        st.write(f"**Klaster {i}:** {count} outlier ({percentage:.2f}% dari klaster)")
        outlier_dist_data.append({"Klaster": f"Klaster {i}", "Jumlah Outlier": count, "Persentase dari Klaster (%)": percentage})
    
    st.table(pd.DataFrame(outlier_dist_data))

    return outlier_counts, cluster_counts, outliers, distances[outliers]

def perform_clustering(data: np.ndarray, k_value: int) -> Tuple[np.ndarray, Optional[object]]:
    """Perform clustering using K-Means.

    Args:
        data: Input data as a numpy array.
        k_value: Number of clusters for K-Means.

    Returns:
        Tuple of (cluster labels as a numpy array, fitted model).
    """
    model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    labels = model.fit_predict(data)
    return labels, model

def get_clustering_diagnostics(labels: np.ndarray) -> int:
    """Calculate the number of clusters for K-Means.

    Args:
        labels: Cluster labels as a numpy array.

    Returns:
        Number of clusters.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    return n_clusters

def plot_silhouette_analysis(data: np.ndarray, labels: np.ndarray, normalization: str) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create a silhouette plot for clustering analysis.

    Args:
        data: Input data as a numpy array.
        labels: Cluster labels as a numpy array.
        normalization: Normalization method used.

    Returns:
        Tuple of (Plotly figure, error message if any).
    """
    try:
        unique_labels = np.unique(labels)
        n_unique_labels = len(unique_labels)
        if n_unique_labels < 2:
            return None, f"Hanya ditemukan {n_unique_labels} cluster. Dibutuhkan minimal 2 cluster untuk analisis silhouette."
        
        silhouette_vals = silhouette_samples(data, labels)
        silhouette_avg = silhouette_score(data, labels)
        
        # Prepare data for plotting
        y_lower = 10
        cluster_labels = sorted(unique_labels)  # Sort for consistent plotting
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        
        for i, cluster in enumerate(cluster_labels):
            cluster_silhouette_vals = silhouette_vals[labels == cluster] # type: ignore
            cluster_silhouette_vals.sort() # type: ignore
            size_cluster_i = cluster_silhouette_vals.shape[0] # type: ignore
            y_upper = y_lower + size_cluster_i
            y_vals = np.arange(y_lower, y_upper)
            
            # Add scatter trace for silhouette scores
            fig.add_trace(go.Scatter(
                x=cluster_silhouette_vals,
                y=y_vals,
                mode='lines+markers',
                name=f'Cluster {cluster}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
            y_lower = y_upper + 10  # Gap between clusters
        
        # Add average silhouette score line
        fig.add_shape( # type: ignore
            type="line",
            x0=silhouette_avg,
            x1=silhouette_avg,
            y0=0,
            y1=y_lower,
            line=dict(color="red", dash="dash")
        )
        
        # Add annotation for average silhouette score
        fig.add_annotation( # type: ignore
            x=silhouette_avg,
            y=y_lower,
            text=f"Average Silhouette Score: {silhouette_avg:.4f}",
            showarrow=True,
            arrowhead=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Silhouette Analysis (K-Means dengan {normalization})",
            xaxis_title="Silhouette Coefficient",
            yaxis_title="Sample Index",
            showlegend=True,
            xaxis=dict(range=[-0.1, 1.1]),
            yaxis=dict(tickvals=[], ticktext=[]),
            height=600
        )
        
        return fig, None
    except Exception as e:
        return None, f"Error saat membuat plot silhouette: {str(e)}"

def plot_elbow_method(data: np.ndarray, normalization: str) -> go.Figure:
    """Perform Elbow Method to determine optimal K and plot the results.

    Args:
        data: Input data as a numpy array.
        normalization: Normalization method used.

    Returns:
        Plotly figure object.
    """
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Create Plotly plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 11)),
        y=wcss,
        mode='lines+markers',
        name='WCSS',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Update layout
    fig.update_layout(
        title=f"Metode Elbow (Sebelum Seleksi Fitur Otomatis, Normalisasi: {normalization})",
        xaxis_title="Jumlah Kluster",
        yaxis_title="WCSS",
        showlegend=True,
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='white',
        height=500
    )

    return fig

def evaluate_features(data, n_clusters=4):
    """Evaluate features based on Silhouette Score and Davies-Bouldin Index.

    Args:
        data: Input data as a pandas DataFrame.
        n_clusters: Number of clusters for K-Means.

    Returns:
        Tuple of (selected features list, final Silhouette Score, final Davies-Bouldin Index).
    """
    baseline_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    baseline_labels = baseline_kmeans.labels_
    baseline_sil = silhouette_score(data, baseline_labels)
    baseline_db = davies_bouldin_score(data, baseline_labels)

    st.write(f"**Baseline (Fitur Pilihan) -> Silhouette: {baseline_sil:.3f}, Davies-Bouldin: {baseline_db:.3f}**")

    result = []
    for fitur in data.columns:
        subset = data.drop(columns=[fitur])
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(subset)
        sil = silhouette_score(subset, labels)
        db = davies_bouldin_score(subset, labels)
        result.append({
            'fitur_dihapus': fitur,
            'silhouette': sil,
            'davies_bouldin': db,
            'delta_silhouette': sil - baseline_sil,
            'delta_db': db - baseline_db
        })
    df_result = pd.DataFrame(result).sort_values(by='delta_silhouette', ascending=False)
    st.table(df_result)
    
    # Ambil fitur yang jika dihapus meningkatkan Silhouette & menurunkan DB Index
    fitur_dihapus_otomatis = df_result[
        (df_result['delta_silhouette'] > 0) &
        (df_result['delta_db'] < 0)
    ]['fitur_dihapus'].tolist()

    fitur_final = [f for f in data.columns if f not in fitur_dihapus_otomatis]
    
    # Ensure at least 2 features remain
    if len(fitur_final) < 2:
        st.warning("Seleksi fitur menghasilkan kurang dari 2 fitur. Memilih fitur dengan dampak terkecil...")
        # Sort by smallest negative impact (largest delta_silhouette, smallest delta_db increase)
        df_result_sorted = df_result.sort_values(by=['delta_silhouette', 'delta_db'], ascending=[False, True])
        # Keep top 2 features with least negative impact
        fitur_dihapus = df_result_sorted['fitur_dihapus'].iloc[2:].tolist()  # Drop all except top 2
        fitur_final = [f for f in data.columns if f not in fitur_dihapus]
        st.write(f"**Fitur yang dipilih untuk memenuhi minimum (berdasarkan dampak terkecil):** {fitur_final}")

    st.write(f"\n**Fitur yang dihapus otomatis:** {fitur_dihapus_otomatis}")
    st.write(f"**Jumlah fitur tersisa:** {len(fitur_final)}")

    # Final evaluation with selected features
    data_final = data[fitur_final]
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42).fit(data_final)
    labels_final = kmeans_final.labels_
    sil_final = silhouette_score(data_final, labels_final)
    db_final = davies_bouldin_score(data_final, labels_final)

    return fitur_final, sil_final, db_final

def correlation_filtering_auto(data, thresholds=None, n_clusters=4):
    """Perform automated feature selection using correlation filtering with optimal threshold.

    Args:
        data: Input data as a pandas DataFrame.
        thresholds: Range of correlation thresholds to evaluate.
        n_clusters: Number of clusters for K-Means.

    Returns:
        Tuple of (filtered feature list, final Silhouette Score, final Davies-Bouldin Index).
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)

    hasil_evaluasi = []

    st.write("**Mengevaluasi berbagai threshold korelasi...**")
    for threshold in thresholds:
        # Hitung korelasi absolut
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        data_filtered = data.drop(columns=to_drop)

        # Skip if fewer than 2 features remain
        if data_filtered.shape[1] < 2:
            hasil_evaluasi.append({
                'threshold': threshold,
                'n_features': data_filtered.shape[1],
                'n_dropped': len(to_drop),
                'silhouette': np.nan,
                'davies_bouldin': np.nan
            })
            st.write(f"**Threshold:** {threshold:.2f} | **Dropped:** {len(to_drop)} | **Fitur Kurang dari 2**")
            continue

        # Evaluasi klastering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_filtered)

        sil = silhouette_score(data_filtered, labels)
        db = davies_bouldin_score(data_filtered, labels)

        hasil_evaluasi.append({
            'threshold': threshold,
            'n_features': data_filtered.shape[1],
            'n_dropped': len(to_drop),
            'silhouette': sil,
            'davies_bouldin': db
        })

        st.write(f"**Threshold:** {threshold:.2f} | **Dropped:** {len(to_drop)} | **S:** {sil:.3f} | **DBI:** {db:.3f}")

    # Konversi ke DataFrame
    df_eval = pd.DataFrame(hasil_evaluasi)
    st.table(df_eval)

    # Pilih threshold terbaik: Silhouette tinggi, DBI rendah
    df_eval_valid = df_eval[df_eval['n_features'] >= 2].copy()
    if df_eval_valid.empty:
        st.warning("Tidak ada threshold yang menghasilkan minimal 2 fitur. Menggunakan semua fitur...")
        fitur_final = data.columns.tolist()
        data_final = data.copy()
    else:
        df_eval_valid['sil_rank'] = df_eval_valid['silhouette'].rank(ascending=False)
        df_eval_valid['dbi_rank'] = df_eval_valid['davies_bouldin'].rank(ascending=True)
        df_eval_valid['total_rank'] = df_eval_valid['sil_rank'] + df_eval_valid['dbi_rank']
        best_threshold = df_eval_valid.sort_values(by='total_rank').iloc[0]['threshold']

        st.write(f"\n**Threshold terbaik berdasarkan gabungan metrik:** {best_threshold:.2f}")

        # Terapkan threshold terbaik
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > best_threshold)]
        data_final = data.drop(columns=to_drop)

    # Final evaluation with selected features
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42)
    labels_final = kmeans_final.fit_predict(data_final)
    sil_final = silhouette_score(data_final, labels_final)
    db_final = davies_bouldin_score(data_final, labels_final)

    st.write(f"\n**Evaluasi Akhir (Threshold = {best_threshold if 'best_threshold' in locals() else 'N/A'}):**") # type: ignore
    st.write(f"**Jumlah fitur tersisa:** {data_final.shape[1]}")
    st.write(f"**Silhouette Score:** {sil_final:.3f}")
    st.write(f"**Davies-Bouldin Index:** {db_final:.3f}")

    return data_final.columns.tolist(), sil_final, db_final

def generate_recommendations(df, data_final, labels, silhouette, db_index, k_value, outlier_counts, cluster_counts, outliers, outlier_distances, cleaned_indices):
    """Generate recommendations based on clustering results and outlier analysis.

    Args:
        df: Original DataFrame.
        data_final: Final DataFrame after feature selection.
        labels: Cluster labels.
        silhouette: Silhouette Score.
        db_index: Davies-Bouldin Index.
        k_value: Number of clusters.
        outlier_counts: Number of outliers per cluster.
        cluster_counts: Number of samples per cluster.
        outliers: Indices of outlier samples.
        outlier_distances: Distances of outliers to their centroids.
        cleaned_indices: Indices of cleaned data.

    Returns:
        List of recommendation strings.
    """
    recommendations = []
    n_samples = len(labels)
    outlier_percentages = [outlier_counts[i] / cluster_counts[i] * 100 if cluster_counts[i] > 0 else 0 for i in range(k_value)]
    max_outlier_cluster = np.argmax(outlier_percentages) if len(outlier_percentages) > 0 else -1
    max_outlier_percentage = max(outlier_percentages) if len(outlier_percentages) > 0 else 0

    # Recommendation 1: Investigate cluster with most outliers
    if max_outlier_cluster >= 0:
        recommendations.append(
            f"**Rekomendasi 1:** Periksa Klaster {max_outlier_cluster} untuk masalah kualitas data, karena memiliki persentase outlier tertinggi ({max_outlier_percentage:.2f}%). Ini menunjukkan kemungkinan adanya data yang tidak konsisten atau salah label."
        )
    else:
        recommendations.append("**Rekomendasi 1:** Tidak ada klaster dengan outlier yang signifikan. Pastikan data sudah bersih dan representatif.")

    # Recommendation 2: Re-evaluate outliers in the cluster with most outliers
    if len(outliers) > 0 and max_outlier_cluster >= 0:
        outlier_df = pd.DataFrame({
            "Index": outliers,
            "Nama Item": [df.iloc[cleaned_indices[idx]]['Description'] if 'Description' in df.columns and cleaned_indices[idx] < len(df) else f"Item_{cleaned_indices[idx]}" for idx in outliers],
            "Klaster": labels[outliers],
            "Jarak ke Centroid": [f"{d:.3f}" for d in outlier_distances]
        })
        cluster_outliers = outlier_df[outlier_df["Klaster"] == max_outlier_cluster].head(3)
        outlier_items = ", ".join(cluster_outliers["Nama Item"].values)
        recommendations.append(
            f"**Rekomendasi 2:** Evaluasi ulang profil data untuk item outlier di Klaster {max_outlier_cluster} (contoh: {outlier_items}). Item ini memiliki jarak besar ke centroid, kemungkinan karena nilai ekstrem pada fitur tertentu."
        )
    else:
        recommendations.append("**Rekomendasi 2:** Tidak ada outlier yang signifikan untuk dievaluasi ulang.")

    # Recommendation 3: Adjust K value if clustering quality is poor
    if silhouette is not None and silhouette < 0.4:
        recommendations.append(
            f"**Rekomendasi 3:** Silhouette Score ({silhouette:.4f}) di bawah 0.4, menunjukkan kualitas klastering yang kurang optimal. Pertimbangkan untuk menyesuaikan K (saat ini {k_value}) dengan menjalankan kembali Elbow Method untuk K yang lebih besar (misalnya {k_value + 1})."
        )
    else:
        recommendations.append("**Rekomendasi 3:** Kualitas klastering cukup baik berdasarkan Silhouette Score. Tidak perlu menyesuaikan K saat ini.")

    # Recommendation 4: Refine feature selection if DBI is high
    if db_index is not None and db_index > 1.0:
        recommendations.append(
            f"**Rekomendasi 4:** Davies-Bouldin Index ({db_index:.4f}) di atas 1.0, menunjukkan pemisahan klaster yang kurang optimal. Coba gunakan metode seleksi fitur lain (misalnya Threshold Correlation) atau tambahkan fitur yang lebih relevan."
        )
    else:
        recommendations.append("**Rekomendasi 4:** Pemisahan klaster cukup baik berdasarkan Davies-Bouldin Index. Tidak perlu mengubah seleksi fitur saat ini.")

    # Recommendation 5: Handle outliers with domain knowledge
    if len(outliers) > 0:
        example_outlier = outlier_df.iloc[0] if not outlier_df.empty else None # type: ignore
        if example_outlier is not None:
            item_name = example_outlier["Nama Item"]
            cluster = example_outlier["Klaster"]
            recommendations.append(
                f"**Rekomendasi 5:** Gunakan pengetahuan domain untuk menangani outlier seperti '{item_name}' di Klaster {cluster}. Item ini mungkin memiliki karakteristik unik yang tidak sesuai dengan klaster saat ini."
            )
        else:
            recommendations.append("**Rekomendasi 5:** Tidak ada outlier untuk dianalisis dengan pengetahuan domain.")
    else:
        recommendations.append("**Rekomendasi 5:** Tidak ada outlier untuk dianalisis dengan pengetahuan domain.")

    # Recommendation 6: Improve data collection for specific outliers
    if len(outliers) > 0:
        outlier_with_high_distance = outlier_df.iloc[outlier_df["Jarak ke Centroid"].astype(float).idxmax()] if not outlier_df.empty else None # type: ignore
        if outlier_with_high_distance is not None:
            item_name = outlier_with_high_distance["Nama Item"]
            recommendations.append(
                f"**Rekomendasi 6:** Tingkatkan pengumpulan data untuk item seperti '{item_name}' (jarak ke centroid: {outlier_with_high_distance['Jarak ke Centroid']}), mungkin dengan menambahkan fitur tambahan yang relevan."
            )
        else:
            recommendations.append("**Rekomendasi 6:** Tidak ada outlier signifikan untuk meningkatkan pengumpulan data.")
    else:
        recommendations.append("**Rekomendasi 6:** Tidak ada outlier signifikan untuk meningkatkan pengumpulan data.")

    # Recommendation 7: Consider alternative normalization
    total_outlier_percentage = len(outliers) / n_samples * 100
    if total_outlier_percentage > 5:
        recommendations.append(
            f"**Rekomendasi 7:** Persentase outlier total ({total_outlier_percentage:.2f}%) lebih dari 5%. Normalisasi saat ini mungkin memperbesar efek outlier. Coba gunakan metode normalisasi lain seperti Min-Max Scaling atau RobustScaler."
        )
    else:
        recommendations.append("**Rekomendasi 7:** Persentase outlier total cukup rendah. Normalisasi saat ini tampaknya sesuai.")

    # Recommendation 8: Validate cluster interpretability
    if max_outlier_percentage > 5:
        recommendations.append(
            f"**Rekomendasi 8:** Klaster {max_outlier_cluster} memiliki outlier tinggi ({max_outlier_percentage:.2f}%), yang dapat mengurangi interpretabilitas klaster. Analisis centroid klaster untuk memahami fitur dominan dan sesuaikan seleksi fitur."
        )
    else:
        recommendations.append("**Rekomendasi 8:** Interpretabilitas klaster tampak baik berdasarkan distribusi outlier.")

    # Recommendation 9: Educate users on outliers
    if len(outliers) > 0:
        recommendations.append(
            "**Rekomendasi 9:** Outlier mungkin mewakili item unik atau data yang salah label. Pertimbangkan untuk meninjau outlier secara manual untuk memastikan keakuratan data dan relevansi klaster."
        )
    else:
        recommendations.append("**Rekomendasi 9:** Tidak ada outlier yang ditemukan, sehingga tidak perlu tinjauan manual saat ini.")

    # Recommendation 10: Automated suggestion to create a new cluster for high-outlier clusters
    if max_outlier_percentage > 5:
        recommendations.append(
            f"**Rekomendasi 10 (Otomatis):** Klaster {max_outlier_cluster} memiliki persentase outlier tinggi ({max_outlier_percentage:.2f}%). Pertimbangkan untuk meningkatkan K menjadi {k_value + 1} untuk mengelompokkan item outlier ini ke dalam klaster baru, seperti klaster untuk item berkalori tinggi."
        )
    else:
        recommendations.append(
            f"**Rekomendasi 10 (Otomatis):** Persentase outlier di semua klaster di bawah 5%. Tidak perlu menambah jumlah klaster saat ini (K={k_value})."
        )

    return recommendations

# --- Main Application ---
def main():
    """Main function to run the Streamlit clustering app."""
    st.title("Clustering dengan K-Means")  # type: ignore

    # Sidebar untuk input
    with st.sidebar:  # type: ignore
        st.header("Konfigurasi Clustering")  # type: ignore
        normalization = st.selectbox("Pilih Normalisasi", ["Min-Max Scaling", "Z-Score"], help="Pilih metode normalisasi data.")  # type: ignore
        
        # Feature selection method
        feature_method = st.selectbox("Pilih Metode Seleksi Fitur", ["Baseline", "Silhouette + Davies-Bouldin", "Threshold Correlation"], help="Pilih metode seleksi fitur.")  # type: ignore
        
        # File upload at the bottom
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], help="Upload file CSV dengan data numerik.")  # type: ignore

    # Main content
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # -----------------------------------------------
            # 1. DATA YANG DIUNGGAH
            # -----------------------------------------------
            st.subheader("1. Data yang Diunggah")  # type: ignore
            st.write("**Data yang diunggah:**")  # type: ignore
            st.dataframe(df.head())  # type: ignore
            st.write("**Statistik Deskriptif Data:**")  # type: ignore
            st.write(df.describe())  # type: ignore
            
            # Use all numeric columns for initial feature selection
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_columns) < 2:
                st.warning("Dataset harus memiliki setidaknya 2 kolom numerik untuk clustering!")  # type: ignore
                return
            
            # Feature pre-selection for Baseline method
            selected_features = st.multiselect(
                "Pilih fitur yang akan digunakan",
                numeric_columns,
                default=numeric_columns,
                help="Pilih setidaknya 2 fitur numerik untuk clustering."
            ) if feature_method == "Baseline" else numeric_columns
            
            if feature_method == "Baseline" and (not selected_features or len(selected_features) < 2):
                st.error("Pilih setidaknya 2 fitur untuk clustering!")
                st.stop()
            
            # -----------------------------------------------
            # 2. PEMBERSIHAN DATA
            # -----------------------------------------------
            st.subheader("2. Pembersihan Data")  # type: ignore
            data_pilih_fitur = df[selected_features].copy()
            
            # Mengonversi semua kolom ke tipe numerik, nilai non-numerik menjadi NaN
            st.write("Mengonversi kolom ke tipe numerik...")  # type: ignore
            for kolom in data_pilih_fitur.columns:
                data_pilih_fitur[kolom] = pd.to_numeric(data_pilih_fitur[kolom], errors='coerce')
            
            # Menghapus baris dengan nilai NaN (data hilang)
            data_bersih = data_pilih_fitur.dropna()
            st.write(f"Ukuran dataset setelah menghapus NaN: {data_bersih.shape} (baris, kolom)")  # type: ignore
            
            if data_bersih.empty:
                st.error("Dataset kosong setelah menghapus NaN. Harap periksa data Anda.")  # type: ignore
                st.stop()  # type: ignore
            
            # Menghapus baris dengan nilai negatif, karena nutrisi tidak boleh negatif
            data_bersih = data_bersih[(data_bersih >= 0).all(axis=1)]
            st.write(f"Ukuran dataset setelah menghapus nilai negatif: {data_bersih.shape} (baris, kolom)")  # type: ignore
            
            if data_bersih.empty:
                st.error("Dataset kosong setelah menghapus nilai negatif. Harap periksa data Anda.")  # type: ignore
                st.stop()  # type: ignore
            
            # Menghapus baris duplikat untuk menghindari redundansi
            data_bersih = data_bersih.drop_duplicates()
            st.write(f"Ukuran dataset setelah menghapus duplikat: {data_bersih.shape} (baris, kolom)")  # type: ignore
            
            if data_bersih.empty:
                st.error("Dataset kosong setelah menghapus duplikat. Harap periksa data Anda.")  # type: ignore
                st.stop()  # type: ignore
            
            if data_bersih.shape[1] < 2:
                st.error("Dataset harus memiliki setidaknya 2 fitur numerik setelah pembersihan!")  # type: ignore
                st.stop()  # type: ignore
            
            # Track indices of cleaned data
            cleaned_indices = data_bersih.index
            
            # Normalize data for Elbow Method
            X = data_bersih.values
            X_normalized_elbow, _ = normalize_data(X, normalization)
            
            # -----------------------------------------------
            # 3. VALIDASI JUMLAH KLASTER (ELBOW METHOD)
            # -----------------------------------------------
            st.subheader("3. Validasi Jumlah Klaster (Elbow Method)")  # type: ignore
            st.write("**Menghitung WCSS untuk menentukan jumlah klaster optimal...**")
            elbow_fig = plot_elbow_method(X_normalized_elbow, normalization)
            st.plotly_chart(elbow_fig)  # type: ignore
            st.write("**Instruksi:** Lihat plot di atas untuk menentukan 'elbow point' (titik siku) di mana penurunan WCSS mulai melambat. Masukkan jumlah klaster (K) yang Anda pilih berdasarkan plot tersebut.")
            
            # Allow user to input K after viewing the Elbow plot
            k_value = st.number_input("Masukkan jumlah K (minimal 2)", min_value=2, max_value=10, value=4, step=1, help="Pilih jumlah klaster berdasarkan plot Elbow Method.")  # type: ignore
            
            # Button to proceed with clustering
            if not st.button("Lanjutkan Clustering dengan K yang Dipilih"):
                st.stop()  # type: ignore
            
            # -----------------------------------------------
            # 4. SELEKSI FITUR
            # -----------------------------------------------
            st.subheader("4. Seleksi Fitur")  # type: ignore
            data_final = data_bersih.copy()
            fitur_final = data_bersih.columns.tolist()
            
            if feature_method == "Baseline":
                st.write("**Metode: Baseline (Memilih terbaik antara Silhouette + Davies-Bouldin dan Threshold Correlation)**")
                # Add a button to trigger automatic feature selection
                if st.button("Lakukan Seleksi Fitur Baseline"):
                    # Evaluate Silhouette + Davies-Bouldin
                    fitur_sil_db, sil_sil_db, db_sil_db = evaluate_features(data_bersih, n_clusters=k_value)
                    
                    # Evaluate Threshold Correlation
                    fitur_corr, sil_corr, db_corr = correlation_filtering_auto(data_bersih, n_clusters=k_value)
                    
                    # Compare and select the best method
                    if sil_sil_db > sil_corr or (sil_sil_db == sil_corr and db_sil_db < db_corr):
                        st.write("**Metode Terpilih: Silhouette + Davies-Bouldin**")
                        fitur_final = fitur_sil_db
                        st.write(f"**Silhouette Score:** {sil_sil_db:.3f}")
                        st.write(f"**Davies-Bouldin Index:** {db_sil_db:.3f}")
                    else:
                        st.write("**Metode Terpilih: Threshold Correlation**")
                        fitur_final = fitur_corr
                        st.write(f"**Silhouette Score:** {sil_corr:.3f}")
                        st.write(f"**Davies-Bouldin Index:** {db_corr:.3f}")
            elif feature_method == "Silhouette + Davies-Bouldin":
                st.write("**Metode: Silhouette + Davies-Bouldin Index**")
                fitur_final, _, _ = evaluate_features(data_bersih, n_clusters=k_value)
            elif feature_method == "Threshold Correlation":
                st.write("**Metode: Threshold Correlation**")
                fitur_final, _, _ = correlation_filtering_auto(data_bersih, n_clusters=k_value)
            
            # Apply selected features
            data_final = data_bersih[fitur_final]
            st.write(f"**Jumlah fitur yang digunakan:** {len(fitur_final)}")
            st.write(f"**Fitur yang digunakan:** {fitur_final}")

            # -----------------------------------------------
            # 5. TRANSFORMASI DATA
            # -----------------------------------------------
            st.subheader("5. Transformasi Data")  # type: ignore
            # Menangani outlier dengan membatasi nilai pada persentil ke-99
            st.write("Menangani outlier dengan pembatasan pada persentil ke-99...")  # type: ignore
            for kolom in data_final.columns:
                batas_atas = data_final[kolom].quantile(0.99)  # Batas pada persentil ke-99
                data_final[kolom] = data_final[kolom].clip(upper=batas_atas)
            
            # Prepare data for clustering
            X = data_final.values
            
            # Normalisasi data
            X_normalized, scaler = normalize_data(X, normalization)
            
            # Clustering
            labels, model = perform_clustering(X_normalized, k_value)

            # Add cluster labels to data_final
            data_final['Cluster'] = labels

            # Diagnostik
            n_clusters = get_clustering_diagnostics(labels)

            # Evaluasi
            silhouette = calculate_silhouette_score(X_normalized, labels)
            db_index = calculate_davies_bouldin_score(X_normalized, labels)

            # -----------------------------------------------
            # 6. HASIL CLUSTERING
            # -----------------------------------------------
            st.subheader("6. Hasil Clustering")  # type: ignore
            st.write(f"**Algoritma**: K-Means")  # type: ignore
            st.write(f"**Normalisasi**: {normalization}")  # type: ignore
            st.write(f"**Jumlah Cluster (K)**: {k_value}")  # type: ignore
            st.write(f"**Jumlah Cluster Terbentuk**: {n_clusters}")  # type: ignore
            outlier_counts, cluster_counts, outliers, outlier_distances = analisis_kmeans(X_normalized, labels, model) # type: ignore
            
            # Akurasi menggunakan Silhouette Score dan Davies-Bouldin Index
            st.write("**Akurasi (Evaluasi Clustering):**")  # type: ignore
            if silhouette is not None:
                st.write(f"- Silhouette Score: {silhouette:.4f} (mendekati 1 = clustering baik)")  # type: ignore
            else:
                st.write("- Silhouette Score: Tidak dapat dihitung (mungkin hanya 1 cluster)")  # type: ignore
            if db_index is not None:
                st.write(f"- Davies-Bouldin Index: {db_index:.4f} (mendekati 0 = clustering baik)")  # type: ignore
            else:
                st.write("- Davies-Bouldin Index: Tidak dapat dihitung (mungkin hanya 1 cluster)")  # type: ignore

            # -----------------------------------------------
            # 7. SILHOUETTE ANALYSIS
            # -----------------------------------------------
            st.subheader("7. Silhouette Analysis")  # type: ignore
            fig, error_msg = plot_silhouette_analysis(X_normalized, labels, normalization)
            if fig is not None:
                st.plotly_chart(fig)  # type: ignore
            else:
                st.warning(f"Silhouette Analysis tidak dapat ditampilkan: {error_msg}")  # type: ignore
                st.info("Untuk K-Means, pastikan data memiliki variasi yang cukup untuk membentuk beberapa cluster.")  # type: ignore

            # -----------------------------------------------
            # 8. INSTANCE ERROR (OUTLIER) ANALYSIS
            # -----------------------------------------------
            st.subheader("8. Instance Error (Outlier) Analysis")  # type: ignore
            st.write("\n**Detail Instance Error (Outlier):**")
            distances = np.zeros(len(labels))
            cluster_centers = model.cluster_centers_ # type: ignore
            for i in range(len(labels)):
                cluster = labels[i]
                distances[i] = np.linalg.norm(X_normalized[i] - cluster_centers[cluster])
            threshold = np.percentile(distances, 95)  # Top 5% farthest points
            outliers = np.where(distances > threshold)[0]

            if len(outliers) > 0:
                st.write("**Daftar item yang dianggap outlier:**")
                outlier_details = []
                unique_clusters = range(n_clusters)
                for idx in outliers:
                    # Map index back to original DataFrame
                    orig_idx = cleaned_indices[idx]
                    # Cari klaster terdekat dari titik outlier
                    dists = []
                    for cluster in unique_clusters:
                        cluster_points = X_normalized[labels == cluster]
                        if len(cluster_points) > 0:
                            center = np.mean(cluster_points, axis=0)
                            dist = np.linalg.norm(X_normalized[idx] - center)
                            dists.append(dist)
                    nearest_cluster = unique_clusters[np.argmin(dists)] if dists else -1
                    distance = min(dists) if dists else float('nan')
                    # Gunakan data_bersih untuk mendapatkan 'Description', fallback ke 'Item_{idx}' jika tidak ada
                    item_name = df.iloc[orig_idx]['Description'] if 'Description' in df.columns and orig_idx < len(df) else f"Item_{orig_idx}"
                    outlier_details.append({
                        "Index": idx,
                        "Nama Item": item_name,
                        "Klaster": nearest_cluster,
                        "Jarak ke Centroid": f"{distance:.3f}"
                    })
                st.table(pd.DataFrame(outlier_details))
            else:
                st.write("**Tidak ada outlier yang ditemukan.**")

            # -----------------------------------------------
            # 9. INTERPRETASI KUALITATIF: TAMPILAN CONTOH MAKANAN PER KLASTER
            # -----------------------------------------------
            st.subheader("9. Interpretasi Kualitatif: Tampilan Contoh Makanan per Klaster")  # type: ignore
            kolom_fitur = data_final.columns.tolist()
            if 'Cluster' in kolom_fitur:
                kolom_fitur.remove('Cluster')  # Remove 'Cluster' from features for display
            
            st.write("\n**Contoh Makanan per Klaster:**")
            unique_clusters = range(k_value)

            for cluster in unique_clusters:
                st.write(f"\n**Cluster {cluster}:**")
                cluster_data = data_final[data_final['Cluster'] == cluster]
                display_columns = ['Description'] + kolom_fitur if 'Description' in df.columns else kolom_fitur
                if 'Description' in df.columns:
                    # Merge Description from original df using indices
                    cluster_data_with_desc = cluster_data.copy()
                    cluster_data_with_desc['Description'] = df.loc[cluster_data.index, 'Description']
                    st.table(cluster_data_with_desc[display_columns].head(5))
                else:
                    st.table(cluster_data[display_columns].head(5))
                
                # Download button for this cluster's data
                cluster_output = io.StringIO()
                cluster_data.to_csv(cluster_output, index=False)
                st.download_button(
                    label=f"Download data Cluster {cluster} sebagai CSV",
                    data=cluster_output.getvalue(),
                    file_name=f"cluster_{cluster}_data.csv",
                    mime="text/csv"
                )  # type: ignore

            # -----------------------------------------------
            # 10. REKOMENDASI BERDASARKAN HASIL CLUSTERING DAN OUTLIER
            # -----------------------------------------------
            st.subheader("10. Rekomendasi Berdasarkan Hasil Clustering dan Outlier")  # type: ignore
            recommendations = generate_recommendations(
                df, data_final, labels, silhouette, db_index, k_value, outlier_counts, cluster_counts, outliers, outlier_distances, cleaned_indices
            )
            for rec in recommendations:
                st.write(rec)

            # -----------------------------------------------
            # 11. DOWNLOAD HASIL CLUSTERING KESELURUHAN
            # -----------------------------------------------
            st.subheader("11. Download Hasil Clustering Keseluruhan")  # type: ignore
            result_df = df.loc[cleaned_indices].copy()
            result_df['Cluster'] = labels
            output = io.StringIO()
            result_df.to_csv(output, index=False)
            st.download_button(
                label="Download hasil clustering keseluruhan",
                data=output.getvalue(),
                file_name="clustering_result.csv",
                mime="text/csv"
            )  # type: ignore
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {str(e)}")  # type: ignore
    else:
        st.info("Silakan upload file CSV untuk memulai clustering.")  # type: ignore

if __name__ == "__main__":
    main()