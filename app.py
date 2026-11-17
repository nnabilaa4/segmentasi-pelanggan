import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Memuat model dan scaler
package = joblib.load(
    "clustering_deployment.joblib"
)

model = package["model"]
scaler = package["scaler"]
feature_columns = package["feature_columns"]

# Judul aplikasi
st.title("Segmentasi Pelanggan Online Retail")
st.write("Prediksi cluster pelanggan menggunakan K-Means")

# Input data pelanggan
recency = st.number_input(
    "Recency (hari)", min_value=0, value=15
)

frequency = st.number_input(
    "Frequency (transaksi)", min_value=1, value=8
)

monetary = st.number_input(
    "Monetary (pound)", min_value=1, value=2500
)

# Tombol prediksi
if st.button("Prediksi Cluster"):

    # Membentuk data pelanggan
    data = pd.DataFrame({
        "Recency": [recency],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log": [np.log1p(monetary)]
    })

    # Scaling menggunakan scaler lama
    data_scaled = scaler.transform(
        data[feature_columns]
    )

    # Prediksi cluster
    cluster = model.predict(data_scaled)

    # Menampilkan hasil
    st.success(
        f"Pelanggan masuk ke Cluster {cluster[0]}"
    )
