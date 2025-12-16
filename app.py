import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date # Tambahkan import date

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

# Inisialisasi Scaler dan Encoders di luar fungsi (agar bisa diakses oleh prediksi)
scaler_A = StandardScaler()
label_encoder_product = LabelEncoder()
label_encoder_transaksi = LabelEncoder()

# ===============================
# 1. FUNGSI PREPROCESSING DATA (Diperbarui untuk menyimpan state encoder)
# ===============================
@st.cache_data
def preprocess_and_fit(df):
    """Melakukan pembersihan dan transformasi data, serta melatih encoders/scalers."""
    global scaler_A, label_encoder_product, label_encoder_transaksi
    
    # Cleaning
    df = df.dropna().drop_duplicates()

    # Konversi Tanggal
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Tanggal"])
    
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    df["Hari"] = df["Tanggal"].dt.day
    df = df.drop(["Tanggal"], axis=1)

    # Konversi Numerik
    kolom_numerik = ["Dibuat", "Terjual", "Harga", "Modal Satuan", "Pemasukan", "Pengeluaran"]
    for col in kolom_numerik:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Encoding Data Kategorik (Fit Encoders)
    if "Nama Produk" in df.columns:
        label_encoder_product.fit(df["Nama Produk"].astype(str))
        df["Nama Produk"] = label_encoder_product.transform(df["Nama Produk"].astype(str))
    
    if "Jenis Transaksi" in df.columns:
        label_encoder_transaksi.fit(df["Jenis Transaksi"].astype(str))
        df["Jenis Transaksi"] = label_encoder_transaksi.transform(df["Jenis Transaksi"].astype(str))
    
    return df

# ===============================
# 2. FUNGSI PELATIHAN MODEL KLASIFIKASI (Random Forest)
# ===============================
@st.cache_resource
def train_classification_model(X_train_A, y_train_A):
    """Melatih dan mengembalikan model Random Forest."""
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_A, y_train_A)
    return model_rf

def run_classification(df):
    """Menjalankan Random Forest Classification."""
    st.header("Bagian A: Random Forest Classifier (Klasifikasi)")
    st.markdown("Memprediksi `Jenis Transaksi` (0 atau 1).")

    try:
        X_A = df.drop(["Jenis Transaksi"], axis=1)
        y_A = df["Jenis Transaksi"]
    except KeyError:
        st.error("Kolom 'Jenis Transaksi' tidak ditemukan untuk klasifikasi.")
        return None, None, None

    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
        X_A, y_A, test_size=0.3, random_state=42
    )

    # Scaling (Fit dan Transform)
    global scaler_A
    X_train_A_scaled = scaler_A.fit_transform(X_train_A)
    X_test_A_scaled = scaler_A.transform(X_test_A)

    # Pelatihan Model
    model_rf = train_classification_model(X_train_A_scaled, y_train_A)
    y_pred_A = model_rf.predict(X_test_A_scaled)

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
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', color='red')
        ax.set_title('Confusion Matrix - Random Forest')
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        st.pyplot(fig) # 
        
    # Mengembalikan model dan data pelatihan untuk prediksi
    return model_rf, X_A.columns, label_encoder_transaksi.classes_

# ===============================
# 3. FUNGSI PELATIHAN MODEL REGRESI (Linear Regression)
# ===============================
@st.cache_resource
def train_regression_model(X_train_B, y_train_B):
    """Melatih dan mengembalikan model Linear Regression."""
    linreg = LinearRegression()
    linreg.fit(X_train_B, y_train_B)
    return linreg

def run_regression(df):
    """Menjalankan Linear Regression."""
    st.header("Bagian B: Linear Regression (Regresi)")
    st.markdown("Memprediksi `Pemasukan` berdasarkan fitur lainnya.")

    try:
        X_B = df.drop(["Pemasukan"], axis=1)
        y_B = df["Pemasukan"]
    except KeyError:
        st.error("Kolom 'Pemasukan' tidak ditemukan untuk regresi.")
        return None

    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
        X_B, y_B, test_size=0.3, random_state=42
    )

    # Pelatihan Model
    linreg = train_regression_model(X_train_B, y_train_B)
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
        min_val = min(y_test_B.min(), y_pred_B.min())
        max_val = max(y_test_B.max(), y_pred_B.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal') 
        ax.set_xlabel("Pemasukan Aktual")
        ax.set_ylabel("Pemasukan Prediksi")
        ax.set_title("Linear Regression: Aktual vs Prediksi")
        st.pyplot(fig) # 
        
    return linreg

# ===============================
# 4. FUNGSI INPUT PENGGUNA DAN PREDIKSI
# ===============================

def user_input_features(product_classes):
    """Mengumpulkan input fitur dari pengguna di sidebar."""
    st.sidebar.header("3. Input Data Baru")
    
    with st.sidebar.form("input_form"):
        st.markdown("**Detail Transaksi**")
        
        # Kolom Kategorik
        nama_produk_str = st.selectbox(
            "Nama Produk", 
            options=product_classes,
            index=0
        )
        # Mengubah input string menjadi label numerik yang dipelajari (handle_unknown='ignore' diimplementasikan secara manual)
        try:
            nama_produk = label_encoder_product.transform([nama_produk_str])[0]
        except ValueError:
            nama_produk = 0 # Default jika produk baru atau tidak dikenal
            
        # Kolom Numerik
        dibuat = st.number_input("Dibuat (Stok Awal)", min_value=1, value=100)
        terjual = st.number_input("Terjual (Kuantitas)", min_value=1, value=5)
        harga = st.number_input("Harga Jual Satuan (Rp)", min_value=1.0, value=50000.0, step=1000.0)
        modal_satuan = st.number_input("Modal Satuan (Rp)", min_value=1.0, value=30000.0, step=1000.0)
        pengeluaran = st.number_input("Pengeluaran Lain (Rp)", min_value=0.0, value=10000.0, step=1000.0)
        
        st.markdown("**Detail Tanggal**")
        # Tanggal (untuk mendapatkan Tahun, Bulan, Hari)
        tanggal_input = st.date_input("Tanggal Transaksi", date.today())
        
        tahun = tanggal_input.year
        bulan = tanggal_input.month
        hari = tanggal_input.day

        submitted = st.form_submit_button("Lakukan Prediksi")

    data = {
        'Nama Produk': nama_produk,
        'Dibuat': dibuat,
        'Terjual': terjual,
        'Harga': harga,
        'Modal Satuan': modal_satuan,
        'Pengeluaran': pengeluaran,
        'Tahun': tahun,
        'Bulan': bulan,
        'Hari': hari
    }
    
    features = pd.DataFrame(data, index=[0])
    return features, submitted

def display_predictions(df_input, model_rf, model_linreg, cols_model, target_classes):
    """Melakukan prediksi pada data input pengguna dan menampilkan hasilnya."""
    st.header("4. Hasil Prediksi untuk Data Baru")

    # Reindex kolom input agar sesuai dengan urutan kolom model (penting!)
    X_input = df_input.reindex(columns=cols_model, fill_value=0)
    
    # --- Prediksi Klasifikasi (Jenis Transaksi) ---
    st.subheader("Prediksi Klasifikasi (Jenis Transaksi)")
    
    # Scaling data input
    X_input_scaled = scaler_A.transform(X_input)
    
    pred_klasifikasi_num = model_rf.predict(X_input_scaled)[0]
    
    # Mengubah prediksi numerik kembali ke label string
    pred_klasifikasi_label = target_classes[pred_klasifikasi_num]
    
    st.metric(
        label="Hasil Prediksi Jenis Transaksi", 
        value=f"Transaksi: {pred_klasifikasi_label}"
    )
    
    # --- Prediksi Regresi (Pemasukan) ---
    st.subheader("Prediksi Regresi (Pemasukan)")
    
    pred_regresi = model_linreg.predict(X_input)[0]
    
    st.metric(
        label="Hasil Prediksi Pemasukan", 
        value=f"Rp {pred_regresi:,.2f}"
    )


# ===============================
# 5. STRUKTUR UTAMA APLIKASI
# ===============================
def main():
    st.title("🚀 Aplikasi Analisis Transaksi Penjualan")
    st.caption("Klasifikasi (Random Forest) dan Regresi (Linear Regression)")
    st.sidebar.header("1. Upload Data Training")
    
    # Inisialisasi variabel model dan kolom
    model_rf, model_linreg = None, None
    cols_model = None
    target_classes = None
    product_classes = []

    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV Anda untuk training model", 
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.sidebar.success("File training berhasil diunggah!")
            
            st.subheader("2. Data Mentah dan Training Model")
            st.dataframe(df_raw.head())
            
            # 1. Preprocessing dan Fit Encoders
            with st.spinner("Membersihkan, mengubah, dan melatih encoders..."):
                df_clean = preprocess_and_fit(df_raw.copy())
            
            # Ambil daftar nama produk unik untuk input
            product_classes = label_encoder_product.classes_
            
            st.success("Preprocessing dan Fit Encoders Selesai!")
            st.dataframe(df_clean.head())
            
            st.divider()
            
            # 2. Training Model Klasifikasi
            if "Jenis Transaksi" in df_clean.columns:
                model_rf, cols_model, target_classes = run_classification(df_clean.copy())
                st.divider()
            else:
                st.warning("Klasifikasi dilewati: Kolom 'Jenis Transaksi' tidak ada.")

            # 3. Training Model Regresi
            if "Pemasukan" in df_clean.columns:
                model_linreg = run_regression(df_clean.copy())
                st.divider()
                
                # Visualisasi Tambahan (Segmentasi)
                # run_segmentation(df_clean.copy()) # Dihilangkan agar fokus pada prediksi

            else:
                st.warning("Regresi dilewati: Kolom 'Pemasukan' tidak ada.")
            
            # --- Bagian Prediksi Input Pengguna ---
            
            if model_rf and model_linreg:
                df_input, submitted = user_input_features(product_classes)
                
                if submitted:
                    display_predictions(df_input, model_rf, model_linreg, cols_model, target_classes)
            else:
                st.error("Model tidak dapat dilatih. Tidak dapat menerima input pengguna.")
                

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.info("Pastikan file CSV Anda memiliki struktur kolom yang benar dan cobalah lagi.")

    else:
        st.info("Silakan unggah file CSV Anda di sidebar kiri untuk melatih model dan memulai prediksi.")

if __name__ == "__main__":
    main()
