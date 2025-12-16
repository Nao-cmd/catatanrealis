import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date 

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

# Inisialisasi Scaler dan Encoders di luar fungsi
scaler_A = StandardScaler()
label_encoder_product = LabelEncoder()
label_encoder_transaksi = LabelEncoder()

# ===============================
# 1. FUNGSI PREPROCESSING DATA (Diperkuat)
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

    
    product_classes = np.array([]) # Inisialisasi sebagai numpy array kosong
    
    # Encoding Data Kategorik (Fit Encoders)
    if "Nama Produk" in df.columns and len(df["Nama Produk"].unique()) > 0:
        label_encoder_product.fit(df["Nama Produk"].astype(str))
        df["Nama Produk"] = label_encoder_product.transform(df["Nama Produk"].astype(str))
        product_classes = label_encoder_product.classes_ # Ambil classes di sini

    if "Jenis Transaksi" in df.columns and len(df["Jenis Transaksi"].unique()) > 0:
        label_encoder_transaksi.fit(df["Jenis Transaksi"].astype(str))
        df["Jenis Transaksi"] = label_encoder_transaksi.transform(df["Jenis Transaksi"].astype(str))
    
    # Mengembalikan data yang sudah diproses dan product_classes
    return df, product_classes

# ===============================
# 2. FUNGSI PELATIHAN MODEL KLASIFIKASI (Random Forest)
# ===============================
@st.cache_resource
def train_classification_model(X_train_A_scaled, y_train_A):
    """Melatih dan mengembalikan model Random Forest."""
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_A_scaled, y_train_A)
    return model_rf

def run_classification(df):
    """Menjalankan Random Forest Classification."""
    st.header("Bagian A: Random Forest Classifier (Klasifikasi)")
    st.markdown("Memprediksi `Jenis Transaksi` (0 atau 1).")

    try:
        # Pengecualian kolom Pemasukan, karena kolom ini akan diprediksi di bagian B
        kolom_drop_A = ["Jenis Transaksi"]
        if "Pemasukan" in df.columns:
            kolom_drop_A.append("Pemasukan")
            
        X_A = df.drop(kolom_drop_A, axis=1, errors='ignore')
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
        report = classification_report(y_test_A, y_pred_A, output_dict=True, zero_division=0)
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
        st.pyplot(fig) 
        
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
        # Pengecualian kolom Jenis Transaksi, karena kolom ini sudah di-encode
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
        st.pyplot(fig)
        
    return linreg

# ===============================
# 4. FUNGSI INPUT PENGGUNA DAN PREDIKSI (Diperkuat)
# ===============================

def user_input_features(product_classes):
    """Mengumpulkan input fitur dari pengguna di sidebar."""
    st.sidebar.header("3. Input Data Baru")
    
    # Ambil nilai default produk untuk menghindari error jika list kosong
    product_options = product_classes if len(product_classes) > 0 else ["Produk A (Isi Data Dulu)"]

    with st.sidebar.form("input_form"):
        st.markdown("**Detail Transaksi**")
        
        # Kolom Kategorik
        nama_produk_str = st.selectbox(
            "Nama Produk", 
            options=product_options,
            index=0
        )
        
        # Mengubah input string menjadi label numerik yang dipelajari
        nama_produk = 0 
        if nama_produk_str not in ["Produk A (Isi Data Dulu)", "N/A (Unggah Data Dulu)"]:
            try:
                # Menggunakan transform yang aman
                nama_produk = label_encoder_product.transform([nama_produk_str])[0]
            except ValueError:
                # Jika produk yang dipilih tidak dikenal oleh encoder yang di-fit, gunakan 0
                nama_produk = 0 
                
        # Kolom Numerik
        dibuat = st.number_input("Dibuat (Stok Awal)", min_value=1, value=100)
        terjual = st.number_input("Terjual (Kuantitas)", min_value=1, value=5)
        harga = st.number_input("Harga Jual Satuan (Rp)", min_value=1.0, value=50000.0, step=1000.0)
        modal_satuan = st.number_input("Modal Satuan (Rp)", min_value=1.0, value=30000.0, step=1000.0)
        pengeluaran = st.number_input("Pengeluaran Lain (Rp)", min_value=0.0, value=10000.0, step=1000.0)
        
        st.markdown("**Detail Tanggal**")
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

def display_predictions(df_input, model_rf, model_linreg, cols_model_rf, target_classes):
    """Melakukan prediksi pada data input pengguna dan menampilkan hasilnya."""
    st.header("4. Hasil Prediksi untuk Data Baru")
    
    # Kolom untuk regresi (semua fitur RF kecuali Jenis Transaksi)
    cols_model_linreg = [col for col in cols_model_rf if col != 'Jenis Transaksi'] 
    
    # --- Prediksi Klasifikasi (Jenis Transaksi) ---
    st.subheader("Prediksi Klasifikasi (Jenis Transaksi)")

    # 1. Klasifikasi: Input harus diskalakan
    X_input_rf = df_input.reindex(columns=cols_model_rf, fill_value=0)
    X_input_scaled = scaler_A.transform(X_input_rf)
    
    pred_klasifikasi_num = model_rf.predict(X_input_scaled)[0]
    
    try:
        # Mengubah prediksi numerik kembali ke label string
        pred_klasifikasi_label = target_classes[int(pred_klasifikasi_num)]
    except IndexError:
        pred_klasifikasi_label = "Label Tidak Dikenal"
    
    st.metric(
        label="Hasil Prediksi Jenis Transaksi", 
        value=f"Transaksi: **{pred_klasifikasi_label}**"
    )
    
    # --- Prediksi Regresi (Pemasukan) ---
    st.subheader("Prediksi Regresi (Pemasukan)")
    
    # 2. Regresi: Input tidak perlu diskalakan
    X_input_linreg = df_input.reindex(columns=cols_model_linreg, fill_value=0)
    
    pred_regresi = model_linreg.predict(X_input_linreg)[0]
    
    st.metric(
        label="Hasil Prediksi Pemasukan", 
        value=f"Rp **{pred_regresi:,.2f}**"
    )


# ===============================
# 5. STRUKTUR UTAMA APLIKASI (Diperkuat)
# ===============================
def main():
    st.title("🚀 Aplikasi Analisis Transaksi Penjualan")
    st.caption("Klasifikasi (Random Forest) dan Regresi (Linear Regression)")
    st.sidebar.header("1. Upload Data Training")
    
    model_rf, model_linreg = None, None
    cols_model_rf = None 
    target_classes = None
    product_classes = np.array([]) # Gunakan numpy array kosong sebagai default

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
                df_clean, product_classes = preprocess_and_fit(df_raw.copy()) 
            
            st.success("Preprocessing dan Fit Encoders Selesai!")
            st.dataframe(df_clean.head())
            
            st.divider()
            
            # 2. Training Model Klasifikasi
            if "Jenis Transaksi" in df_clean.columns and len(df_clean["Jenis Transaksi"].unique()) > 1:
                model_rf, cols_model_rf, target_classes = run_classification(df_clean.copy())
                st.divider()
            else:
                st.warning("Klasifikasi dilewati: Kolom 'Jenis Transaksi' tidak ada atau hanya memiliki satu kelas unik.")

            # 3. Training Model Regresi
            if "Pemasukan" in df_clean.columns:
                model_linreg = run_regression(df_clean.copy())
                st.divider()

            else:
                st.warning("Regresi dilewati: Kolom 'Pemasukan' tidak ada.")
            
            # --- Bagian Prediksi Input Pengguna ---
            
            if model_rf and model_linreg: 
                st.sidebar.caption(f"Jumlah produk unik dipelajari: {len(product_classes)}")
                
                df_input, submitted = user_input_features(product_classes)
                
                if submitted:
                    if cols_model_rf is not None and target_classes is not None:
                         display_predictions(df_input, model_rf, model_linreg, cols_model_rf, target_classes)
                    else:
                        st.error("Model Klasifikasi belum selesai diinisialisasi. Tidak dapat memprediksi.")
            else:
                st.sidebar.header("3. Input Data Baru")
                st.warning("Model tidak dapat dilatih sepenuhnya. Prediksi tidak dapat dilakukan.")
                

        except Exception as e:
            # Penanganan error umum
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.info("Pastikan file CSV Anda memiliki struktur kolom yang benar (Tanggal, Nama Produk, Jenis Transaksi, Pemasukan, dll.) dan cobalah lagi.")

    else:
        st.info("Silakan unggah file CSV Anda di sidebar kiri untuk melatih model dan memulai prediksi.")

if __name__ == "__main__":
    main()
