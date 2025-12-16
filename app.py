import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    mean_absolute_error, 
    r2_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Mengatur konfigurasi halaman
st.set_page_config(
    page_title="Analisis Transaksi Penjualan (Klasifikasi & Regresi)",
    layout="wide"
)

# ===============================
# 1. FUNGSI PREPROCESSING DATA
# ===============================
@st.cache_data
def preprocess_data(df):
    """Melakukan pembersihan dan transformasi data."""
    
    # Cleaning
    df = df.dropna().drop_duplicates()

    # Konversi Tanggal
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Tanggal"])
    
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    df["Hari"]  = df["Tanggal"].dt.day
    df = df.drop(["Tanggal"], axis=1)

    # Konversi Numerik
    kolom_numerik = ["Dibuat", "Terjual", "Harga", "Modal Satuan", "Pemasukan", "Pengeluaran"]
    for col in kolom_numerik:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Encoding Data Kategorik
    encoder = LabelEncoder()
    if "Nama Produk" in df.columns:
        df["Nama Produk"] = encoder.fit_transform(df["Nama Produk"].astype(str))
    
    # Transaksi target A
    if "Jenis Transaksi" in df.columns:
        df["Jenis Transaksi"] = encoder.fit_transform(df["Jenis Transaksi"].astype(str))
    
    return df

# ===============================
# 2. FUNGSI MODEL KLASIFIKASI (Random Forest)
# ===============================
def run_classification(df):
    """Menjalankan Random Forest Classification."""
    st.header("Bagian A: Random Forest Classifier (Klasifikasi)")
    st.markdown("Memprediksi `Jenis Transaksi` (0 atau 1).")

    try:
        X_A = df.drop(["Jenis Transaksi"], axis=1)
        y_A = df["Jenis Transaksi"]
    except KeyError:
        st.error("Kolom 'Jenis Transaksi' tidak ditemukan untuk klasifikasi.")
        return

    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_A, y_A, test_size=0.3, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_A = scaler.fit_transform(X_train_A)
    X_test_A = scaler.transform(X_test_A)

    # Pelatihan Model
    with st.spinner("Melatih Random Forest Classifier..."):
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_rf.fit(X_train_A, y_train_A)
        y_pred_A = model_rf.predict(X_test_A)

    st.success("Klasifikasi Selesai!")

    # Evaluasi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metrik Kinerja")
        accuracy = accuracy_score(y_test_A, y_pred_A)
        st.metric(label="Akurasi", value=f"{accuracy*100:.2f}%")
        
        st.text("Classification Report:")
        report = classification_report(y_test_A, y_pred_A, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df.drop('support', axis=1))

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_A, y_pred_A)
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        # Menambahkan label pada matrix
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', color='red')
            
        ax.set_title('Confusion Matrix - Random Forest')
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        st.pyplot(fig) # 

# ===============================
# 3. FUNGSI MODEL REGRESI (Linear Regression)
# ===============================
def run_regression(df):
    """Menjalankan Linear Regression."""
    st.header("Bagian B: Linear Regression (Regresi)")
    st.markdown("Memprediksi `Pemasukan` berdasarkan fitur lainnya.")

    try:
        X_B = df.drop(["Pemasukan"], axis=1)
        y_B = df["Pemasukan"]
    except KeyError:
        st.error("Kolom 'Pemasukan' tidak ditemukan untuk regresi.")
        return

    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
        X_B, y_B, test_size=0.3, random_state=42
    )

    # Pelatihan Model
    with st.spinner("Melatih Linear Regression..."):
        linreg = LinearRegression()
        linreg.fit(X_train_B, y_train_B)
        y_pred_B = linreg.predict(X_test_B)

    st.success("Regresi Selesai!")

    # Evaluasi
    r2 = r2_score(y_test_B, y_pred_B)
    mae = mean_absolute_error(y_test_B, y_pred_B)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metrik Kinerja")
        st.metric(label="R² Score", value=f"{r2:.3f}")
        st.metric(label="Mean Absolute Error (MAE)", value=f"Rp {mae:,.2f}")
        
    with col2:
        st.subheader("Aktual vs Prediksi")
        fig, ax = plt.subplots()
        ax.scatter(y_test_B, y_pred_B, alpha=0.6)
        # Garis y=x (ideal)
        min_val = min(y_test_B.min(), y_pred_B.min())
        max_val = max(y_test_B.max(), y_pred_B.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal') 
        ax.set_xlabel("Pemasukan Aktual")
        ax.set_ylabel("Pemasukan Prediksi")
        ax.set_title("Linear Regression: Aktual vs Prediksi")
        st.pyplot(fig) # 

# ===============================
# 4. FUNGSI SEGMENTASI (Visualisasi Tambahan)
# ===============================
def run_segmentation(df):
    """Menjalankan dan memvisualisasikan segmentasi pelanggan."""
    st.header("Visualisasi Segmentasi Pelanggan")

    if "Pemasukan" not in df.columns:
        st.error("Kolom 'Pemasukan' tidak ditemukan untuk segmentasi.")
        return

    # Segmentasi berdasarkan quantile pemasukan
    df["Segment_Pelanggan"] = pd.qcut(
        df["Pemasukan"],
        q=3,
        labels=["Low Value", "Medium Value", "High Value"]
    )
    
    st.subheader("Distribusi Segmen Pelanggan")
    segment_count = df["Segment_Pelanggan"].value_counts().sort_index()
    st.dataframe(segment_count.to_frame(name="Jumlah Transaksi"))

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Jumlah Transaksi per Segmen")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(segment_count.index, segment_count.values, color=['#7FFFD4', '#FFA07A', '#FFD700'])
        ax.set_xlabel("Segmen Pelanggan")
        ax.set_ylabel("Jumlah Transaksi")
        ax.set_title("Segmentasi Pelanggan Berdasarkan Pemasukan")
        st.pyplot(fig)

    with col2:
        st.subheader("Pemasukan vs Terjual berdasarkan Segmen")
        fig, ax = plt.subplots(figsize=(6,4))
        scatter = ax.scatter(
            df["Terjual"],
            df["Pemasukan"],
            c=df["Segment_Pelanggan"].cat.codes,
            cmap='viridis',
            alpha=0.6
        )
        ax.set_xlabel("Jumlah Terjual")
        ax.set_ylabel("Pemasukan")
        ax.set_title("Scatter Segmentasi Pelanggan")
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper left", title="Segmen")
        ax.add_artist(legend1)
        st.pyplot(fig)
        
# ===============================
# 5. STRUKTUR UTAMA APLIKASI
# ===============================
def main():
    st.title("🚀 Aplikasi Analisis Transaksi Penjualan")
    st.caption("Klasifikasi (Random Forest) dan Regresi (Linear Regression)")
    st.sidebar.header("1. Upload Data")

    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV Anda (Contoh: catatan_sangat_realistis.csv)", 
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.sidebar.success("File berhasil diunggah!")
            
            # Tampilkan Raw Data
            st.subheader("2. Data Mentah (Preview)")
            st.dataframe(df_raw.head())
            
            # Preprocessing
            st.subheader("3. Preprocessing Data")
            with st.spinner("Membersihkan dan mengubah data..."):
                df_clean = preprocess_data(df_raw.copy())
            st.success("Preprocessing Selesai!")
            st.dataframe(df_clean.head())
            
            st.divider()
            
            # --- MENJALANKAN MODEL ---
            
            # Pastikan kolom yang diperlukan ada sebelum menjalankan model
            if "Jenis Transaksi" in df_clean.columns:
                run_classification(df_clean.copy())
                st.divider()
            else:
                st.warning("Klasifikasi dilewati: Kolom 'Jenis Transaksi' tidak ada setelah preprocessing.")

            if "Pemasukan" in df_clean.columns:
                run_regression(df_clean.copy())
                st.divider()
                
                # Visualisasi Tambahan (Segmentasi)
                run_segmentation(df_clean.copy())
            else:
                st.warning("Regresi dan Segmentasi dilewati: Kolom 'Pemasukan' tidak ada setelah preprocessing.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.info("Pastikan file CSV Anda memiliki struktur kolom yang benar seperti yang didefinisikan dalam kode (Tanggal, Nama Produk, Jenis Transaksi, Dibuat, Terjual, Harga, Modal Satuan, Pemasukan, Pengeluaran).")

    else:
        st.info("Silakan unggah file CSV Anda di sidebar kiri untuk memulai analisis.")

if __name__ == "__main__":
    main()
