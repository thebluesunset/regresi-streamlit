## Aplikasi Penyelesaian SPL (Regresi Linear - Metode Eliminasi Gauss) dengan Streamlit
## Jalankan dengan: streamlit run app_spl_gauss.py

import streamlit as st
import pandas as pd
import numpy as np

def baca_file(file):
    """Membaca file CSV atau Excel menjadi DataFrame."""
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_excel(file)

def bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep):
    """Membentuk matriks desain X dan vektor target y dari DataFrame."""
    data = df[kolom_fitur + [kolom_target]].apply(pd.to_numeric, errors="raise")
    if data.isnull().any().any():
        raise ValueError("Terdapat sel kosong/NaN pada kolom yang dipilih.")
    
    X = data[kolom_fitur].to_numpy(dtype=float)
    y = data[kolom_target].to_numpy(dtype=float)
    nama_w = [f"w{i+1}" for i in range(len(kolom_fitur))]
    
    if pakai_intersep:
        X = np.column_stack([np.ones(len(X)), X])  # Tambah kolom 1 di kiri
        nama_w = ["w0"] + nama_w
        
    return X, y, nama_w

def selesaikan_gauss(X, y):
    """
    Menyelesaikan (X^T X) w = X^T y menggunakan Metode Eliminasi Gauss.
    """
    A = (X.T @ X).copy()          # Matriks koefisien (X^T X)
    b = (X.T @ y).copy()          # Vektor ruas kanan (X^T y)
    
    n = len(b)
    # Membentuk Matriks Augmentasi [A | b]
    aug = np.column_stack([A, b])
    
    # 1. Eliminasi Maju (Forward Elimination)
    for i in range(n):
        # Partial Pivoting untuk stabilitas numerik
        max_row = i + np.argmax(np.abs(aug[i:, i]))
        if np.abs(aug[max_row, i]) < 1e-12:
            raise ValueError(
                "Matriks XᵀX singular atau tidak memiliki solusi unik. "
                "Eliminasi Gauss tidak dapat dilanjutkan karena elemen pivot bernilai nol."
            )
        # Tukar baris jika diperlukan
        if max_row != i:
            aug[[i, max_row]] = aug[[max_row, i]]
            
        # Proses eliminasi elemen di bawah pivot
        for j in range(i + 1, n):
            faktor = aug[j, i] / aug[i, i]
            aug[j, i:] -= faktor * aug[i, i:]

    matriks_segitiga_atas = aug[:, :n].copy()
    b_tertransformasi = aug[:, n].copy()

    #  2. Substitusi Mundur (Back Substitution)
    w = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suku_diketahui = np.sum(aug[i, i + 1:n] * w[i + 1:n])
        w[i] = (aug[i, n] - suku_diketahui) / aug[i, i]

    return A, b, aug, matriks_segitiga_atas, b_tertransformasi, w

#  Tampilan Utama Streamlit

st.set_page_config(page_title="SPL Regresi Linear - Gauss", layout="wide")
st.title("Penyelesaian SPL Regresi Linear dengan Eliminasi Gauss")
st.caption("Menyelesaikan persamaan (XᵀX) w = Xᵀy menggunakan Metode Eliminasi Gauss & Substitusi Mundur")

file = st.file_uploader("Unggah file data (CSV / Excel)", type=["csv", "xlsx", "xls"])

if file is not None:
    try:
        df = baca_file(file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    st.subheader("1. Data yang Diunggah")
    st.dataframe(df)

    # Pilih Kolom
    kolom_numerik = df.select_dtypes(include="number").columns.tolist()
    if len(kolom_numerik) < 2:
        st.error("Data harus memiliki minimal 2 kolom numerik (fitur dan target).")
        st.stop()

    st.subheader("2. Pilih Variabel")
    col1, col2 = st.columns(2)
    with col1:
        kolom_target = st.selectbox(
            "Kolom target (y)", kolom_numerik, index=len(kolom_numerik) - 1
        )
    with col2:
        abaikan = ("sampel", "no", "id", "nomor")
        default_fitur = [
            k for k in kolom_numerik
            if k != kolom_target and k.lower() not in abaikan
        ]
        kolom_fitur = st.multiselect(
            "Kolom fitur (x)", [k for k in kolom_numerik if k != kolom_target],
            default=default_fitur,
        )

    pakai_intersep = st.checkbox("Tambahkan intersep (w0)", value=True)

    # Tombol Hitung
    if st.button("Selesaikan (Eliminasi Gauss)", type="primary"):
        try:
            if not kolom_fitur:
                raise ValueError("Pilih minimal satu kolom fitur.")
            
            X, y, nama_w = bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep)
            A, b, aug, U, b_mod, w = selesaikan_gauss(X, y)

            st.session_state["hasil"] = dict(
                file=file.name, fitur=kolom_fitur, target=kolom_target,
                X=X, y=y, nama_w=nama_w, A=A, b=b, aug=aug, U=U,
                b_mod=b_mod, w=w, intersep=pakai_intersep
            )
        except Exception as e:
            st.session_state.pop("hasil", None)
            st.error(f"Terjadi kesalahan: {e}")

    # Tampilkan Hasil
    h = st.session_state.get("hasil")
    if h and h["file"] == file.name:
        X, y, w, nama_w = h["X"], h["y"], h["w"], h["nama_w"]
        w = w + 0.0  # Menghindari tampilan -0.0

        st.subheader("3. Langkah Perhitungan")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matriks Awal XᵀX (A)**")
            st.dataframe(pd.DataFrame(h["A"], index=nama_w, columns=nama_w))
        with c2:
            st.markdown("**Vektor Awal Xᵀy (b)**")
            st.dataframe(pd.DataFrame(h["b"], index=nama_w, columns=["Xᵀy"]))

        st.subheader("4. Hasil Eliminasi Maju (Matriks Segitiga Atas)")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Matriks Segitiga Atas (U)**")
            st.dataframe(pd.DataFrame(h["U"], index=nama_w, columns=nama_w))
        with c4:
            st.markdown("**Vektor b setelah Eliminasi**")
            st.dataframe(pd.DataFrame(h["b_mod"], index=nama_w, columns=["b ter-transformasi"]))

        st.subheader("5. Solusi w (Hasil Substitusi Mundur)")
        df_gauss = pd.DataFrame({
            "Parameter": nama_w,
            "Nilai Solusi (w_i)": np.round(w, 4)
        })
        st.dataframe(df_gauss, hide_index=True)

        # Persamaan Model
        suku = []
        for nama, nilai, fitur in zip(nama_w, w, ([None] + h["fitur"]) if h["intersep"] else h["fitur"]):
            suku.append(f"{nilai:.4f}" if fitur is None else f"{nilai:.4f}·{fitur}")
        st.info(f"Model: {h['target']} = " + " + ".join(suku))

        # Evaluasi Model
        y_hat = X @ w
        galat = y - y_hat
        sse = float(np.sum(galat ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / sst if sst > 0 else float("nan")

        m1, m2 = st.columns(2)
        m1.metric("RMSE", f"{np.sqrt(np.mean(galat ** 2)):.4f}")
        m2.metric("R²", f"{r2:.4f}")

        st.markdown("**Perbandingan Nilai Aktual dan Prediksi**")
        df_pred = pd.DataFrame({
            "Aktual": y, 
            "Prediksi": np.round(y_hat, 4),
            "Galat": np.round(galat, 4)
        })
        st.dataframe(df_pred)

        # Prediksi Data Baru
        st.subheader("6. Prediksi Data Baru")
        nilai_baru = []
        kolom_in = st.columns(len(h["fitur"]))
        for kol, nama in zip(kolom_in, h["fitur"]):
            nilai_baru.append(kol.number_input(nama, value=float(df[nama].mean())))
        
        x_baru = np.array(([1.0] if h["intersep"] else []) + nilai_baru)
        st.success(f"Prediksi {h['target']} = {float(x_baru @ w):.2f}")
else: 
    st.info("Silakan unggah file CSV/Excel terlebih dahulu.")