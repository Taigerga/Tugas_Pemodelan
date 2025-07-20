import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec 
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import seaborn as sns
import warnings
import os
import streamlit as st
from io import BytesIO
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Simulasi ISPU", layout="wide", page_icon="🌫️")

# Title
st.title("Laporan Simulasi ISPU dan Visualisasi")

# ====== Fungsi untuk Dokumen Word ======
def save_to_word(document, title, content=None, image=None, table=None):
    document.add_heading(title, level=2)
    if content:
        for line in content.split('\n'):
            if line.strip():
                p = document.add_paragraph(line.strip())
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if table is not None:
        df = table
        t = document.add_table(rows=df.shape[0]+1, cols=df.shape[1])
        t.style = 'Table Grid'
        for j in range(df.shape[1]):
            t.cell(0, j).text = df.columns[j]
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                t.cell(i+1, j).text = str(df.values[i, j])
    if image:
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        plt.close()
        document.add_picture(image_stream, width=Inches(5))
    document.add_paragraph()

def create_word_document(df, params, n_samples):
    doc = Document()
    doc.add_heading("Laporan Simulasi ISPU dan Visualisasi", 0)
    
    # ====== BAGIAN BARU: STATISTIK & PARAMETER DISTRIBUSI ======
    doc.add_heading("Statistik dan Parameter Distribusi", level=2)
    
    # 1. Statistik Deskriptif
    p = doc.add_paragraph()
    p.add_run("Statistik Deskriptif Simulasi (µg/m³)").bold = True
    p.add_run("\nBerikut ringkasan statistik data simulasi:")
    
    # Buat tabel statistik
    stats_table = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].describe().round(2)
    t = doc.add_table(stats_table.shape[0]+1, stats_table.shape[1]+1)
    t.style = 'Table Grid'
    
    # Header kolom
    for j in range(stats_table.shape[1]):
        t.cell(0,j+1).text = stats_table.columns[j]
    
    # Header baris dan data
    for i in range(stats_table.shape[0]):
        t.cell(i+1,0).text = stats_table.index[i]
        for j in range(stats_table.shape[1]):
            t.cell(i+1,j+1).text = str(stats_table.iloc[i,j])
    
    doc.add_paragraph("Catatan: CO ditampilkan dalam µg/m³ (tanpa konversi)")
    
    # 2. Parameter Distribusi
    doc.add_paragraph().add_run("\nParameter Distribusi Normal (Data Excel)").bold = True
    
    # Hitung parameter dari data asli
    param_data = []
    for col in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']:
        mu, std = norm.fit(df[col])
        param_data.append([col, f"{mu:.2f}", f"{std:.2f}"])
    
    # Buat tabel parameter
    param_table = doc.add_table(rows=len(param_data)+1, cols=3)
    param_table.style = 'Table Grid'
    
    # Header
    param_table.cell(0,0).text = "Polutan"
    param_table.cell(0,1).text = "Mean (µg/m³)"
    param_table.cell(0,2).text = "Std Dev (µg/m³)"
    
    # Isi data
    for i, row in enumerate(param_data, start=1):
        param_table.cell(i,0).text = row[0]  # Polutan
        param_table.cell(i,1).text = row[1]  # Mean
        param_table.cell(i,2).text = row[2]  # Std Dev
    
    doc.add_paragraph("Parameter ini digunakan sebagai dasar simulasi data")
    
    # Distribusi Parameter
    plt.figure(figsize=(15, 10))
    plt.suptitle('Distribusi Parameter Kualitas Udara (Normal)', fontsize=14)
    for i, col in enumerate(['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], 1):
        if col in params:
            dist_type, mu, std = params[col]
            plt.subplot(2, 3, i)
            sns.histplot(df[col], bins=20, kde=False, stat="density", alpha=0.6, label='Histogram')
            x = np.linspace(0, df[col].max(), 100)
            pdf = norm.pdf(x, mu, std)
            plt.plot(x, pdf, 'g-', lw=2, label='Normal Fit')
            plt.title(f'{col} (Normal)')
            plt.xlabel('Konsentrasi')
            plt.ylabel('Density')
            plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_to_word(doc, "Distribusi Parameter dengan Kurva Normal", 
                content="""**Histogram menunjukkan distribusi konsentrasi polutan. Garis hijau menunjukkan kurva distribusi normal.**
                           **Penjelasan:**
                           - Grafik membandingkan distribusi aktual (histogram) dengan kurva normal teoritis
                           - Garis hijau: Distribusi Normal yang difitkan ke data
                           - Semua polutan sekarang dimodelkan dengan distribusi normal
                           - PM2.5 dan PM10 cenderung memiliki skewness positif
                           - SO2 dan CO memiliki distribusi yang lebih simetris
                """,
                image=True)
    
    # Tren Harian ISPU
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['ISPU_max'], label='ISPU Harian', alpha=0.5)
    plt.plot(df['ISPU_max'].rolling(7).mean(), label='Rata-rata 7 Hari', color='red')
    plt.title('Tren Harian ISPU dengan Smoothing')
    plt.xlabel('Hari')
    plt.ylabel('ISPU')
    plt.grid(True)
    plt.legend()
    save_to_word(doc, "Tren Harian ISPU", 
                content="""**Plot menunjukkan tren ISPU harian selama simulasi dengan rata-rata 7 hari terakhir.**
                           **Penjelasan Tren ISPU:**
                           - Garis biru: Nilai ISPU harian dengan fluktuasi alami
                           - Garis merah: Rata-rata 7 hari untuk melihat tren
                           - Lonjakan ISPU bisa disebabkan oleh:
                           - Peningkatan aktivitas kendaraan
                           - Pembakaran biomassa
                           - Kondisi meteorologi yang tidak mendispersi polutan
                """,
                image=True)
    
    # Kategori ISPU
    df['Kategori_ISPU'] = df['ISPU_max'].apply(lambda x: "Baik" if x <= 50 else 
                                              "Sedang" if x <= 100 else 
                                              "Tidak Sehat" if x <= 200 else 
                                              "Sangat Tidak Sehat" if x <= 300 else 
                                              "Berbahaya")
    
    kategori_counts = df['Kategori_ISPU'].value_counts()
    warna_kategori = {
        'Baik': 'green',
        'Sedang': 'gold',
        'Tidak Sehat': 'orange',
        'Sangat Tidak Sehat': 'red',
        'Berbahaya': 'darkred'
    }
    colors = [warna_kategori[k] for k in kategori_counts.index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.pie(kategori_counts, labels=kategori_counts.index,
            autopct='%1.1f%%', startangle=90, colors=colors,
            shadow=True, textprops={'color': 'black', 'fontsize': 10})
    ax1.set_title('Distribusi Persentase Kategori ISPU')
    ax1.axis('equal')
    
    bars = ax2.bar(kategori_counts.index, kategori_counts.values, color=colors, edgecolor='black')
    ax2.set_title('Jumlah Hari per Kategori ISPU')
    ax2.set_xlabel('Kategori ISPU')
    ax2.set_ylabel('Jumlah Hari')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}\n({height/len(df):.1%})',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_to_word(doc, "Distribusi Kategori ISPU", 
                content="""**Diagram lingkaran dan batang menunjukkan distribusi kategori ISPU selama periode simulasi.**
                           **Distribusi Kategori Kualitas Udara:**
                           - **Baik (0-50)**: Tidak ada risiko kesehatan
                           - **Sedang (51-100)**: Kelompok sensitif mungkin terpengaruh
                           - **Tidak Sehat (101-200)**: Seluruh populasi mulai terpengaruh
                           - **Sangat Tidak Sehat (201-300)**: Peringatan kesehatan serius
                           - **Berbahaya (>300)**: Darurat kesehatan publik
                """,
                image=True)

    # Polutan Dominan
    polutan_dominan = df['Parameter_Dominan'].value_counts()
    plt.figure(figsize=(10, 6))
    polutan_dominan.plot(kind='bar', color='skyblue')
    plt.title('Frekuensi Polutan Dominan Penyebab ISPU Tertinggi')
    plt.xlabel('Polutan')
    plt.ylabel('Jumlah Hari')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_to_word(doc, "Polutan Dominan Penyebab ISPU Tertinggi",
                content="""**Diagram batang menunjukkan polutan mana yang paling sering menjadi penyebab ISPU tertinggi.**
                           **Penjelasan Diagram:**
                           - **Sumbu X**: Nama polutan (PM2.5, PM10, SO2, NO2, CO, O3)
                           - **Sumbu Y**: Jumlah hari polutan tersebut menjadi penyebab ISPU tertinggi
                           - Warna biru muda: Frekuensi dominansi masing-masing polutan
                           - Garis grid: Membantu membaca nilai secara akurat
                """,
                image=True)
    
    # Line Chart Polutan Utama
    plt.figure(figsize=(14, 6))
    for polutan in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']:
        plt.plot(df.index, df[polutan], label=polutan, alpha=0.7)
    plt.title('Tren Polutan Utama')
    plt.xlabel('Hari')
    plt.ylabel('Konsentrasi')
    plt.legend()
    plt.grid(True)
    save_to_word(doc, "Tren Polutan Utama",
                content="""**Line chart menunjukkan tren beberapa polutan utama selama periode simulasi.**
                           **Penjelasan Diagram:**
                           - **Sumbu X**: Hari ke- (dikonversi dari hari simulasi)
                           - **Sumbu Y**: Konsentrasi harian (µg/m³)
                           - Garis berwarna: Mewakili polutan berbeda
                           - Area terarsir: Variasi harian dalam Hari tersebut
                           - Pola mingguan membantu identifikasi pengaruh aktivitas manusia
                """,
                image=True)

    #Histogram PM2.5
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PM2.5'], bins=30, kde=True, color='teal')
    plt.axvline(x=15.5, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi PM2.5')
    plt.xlabel('PM2.5 (µg/m³)')
    plt.ylabel('Frekuensi (Jumlah Hari)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_to_word(doc, "Distribusi Konsentrasi PM2.5",
                content="""**Histogram menunjukkan Distribusi PM2.5.**
                           **Penjelasan Diagram:**
                           - **Sumbu X**: Rentang konsentrasi PM2.5 (µg/m³)
                           - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
                           - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
                           - Garis KDE (Kernel Density Estimate): Estimasi kurva distribusi
                           - Garis Merah batas aman ISPU
                """,
                image=True)

    #Histogram PM10 
    plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['PM10'], bins=30, kde=True, color='teal')
    plt.axvline(x=50, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi PM10')
    plt.xlabel('PM10 (µg/m³)')
    plt.ylabel('Frekuensi (Jumlah Hari)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_to_word(doc, "Distribusi Konsentrasi PM10",
                content="""**Histogram menunjukkan Distribusi PM10.**
                        **Penjelasan Diagram:**
                        - **Sumbu X**: Rentang konsentrasi PM10 (µg/m³)
                        - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
                        - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
                        - Garis KDE (Kernel Density Estimate): Estimasi kurva distribusi
                        - Garis Merah batas aman ISPU
                """,
                image=True)

    #Histogram SO2
    plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['SO2'], bins=30, kde=True, color='teal')
    plt.axvline(x=52, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi SO2')
    plt.xlabel('SO2 (µg/m³)')
    plt.ylabel('Frekuensi (Jumlah Hari)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_to_word(doc, "Distribusi Konsentrasi SO2",
                content="""**Histogram menunjukkan Distribusi SO2.**
                        **Penjelasan Diagram:**
                        - **Sumbu X**: Rentang konsentrasi SO2 (µg/m³)
                        - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
                        - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
                        - Garis KDE: Estimasi kurva distribusi
                        - Garis Merah batas aman ISPU
                """,
                image=True)

    #Histogram NO2
    plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['NO2'], bins=30, kde=True, color='teal')
    plt.axvline(x=80, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi NO2')
    plt.xlabel('NO2 (µg/m³)')
    plt.ylabel('Frekuensi (Jumlah Hari)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_to_word(doc, "Distribusi Konsentrasi NO2",
                content="""**Histogram menunjukkan Distribusi NO2.**
                        **Penjelasan Diagram:**
                        - **Sumbu X**: Rentang konsentrasi NO2 (µg/m³)
                        - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
                        - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
                        - Garis KDE: Estimasi kurva distribusi
                        - Garis Merah batas aman ISPU
                """,
                image=True)

    #Histogram CO
    plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['CO'], bins=30, kde=True, color='teal')
    plt.axvline(x=4, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi CO')
    plt.xlabel('CO (µg/m³)')
    plt.ylabel('Frekuensi (Jumlah Hari)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_to_word(doc, "Distribusi Konsentrasi CO",
                content="""**Histogram menunjukkan Distribusi CO.**
                        **Penjelasan Diagram:**
                        - **Sumbu X**: Rentang konsentrasi CO (mg/m³)
                        - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
                        - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
                        - Garis KDE: Estimasi kurva distribusi
                        - Garis Merah batas aman ISPU
                """,
                image=True)

    #Histogram O3
    plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['O3'], bins=30, kde=True, color='teal')
    plt.axvline(x=120, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi O3')
    plt.xlabel('O3 (µg/m³)')
    plt.ylabel('Frekuensi (Jumlah Hari)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_to_word(doc, "Distribusi Konsentrasi O3",
                content="""**Histogram menunjukkan Distribusi O3.**
                        **Penjelasan Diagram:**
                        - **Sumbu X**: Rentang konsentrasi O3 (µg/m³)
                        - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
                        - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
                        - Garis KDE: Estimasi kurva distribusi
                        - Garis Merah batas aman ISPU
                """,
                image=True)

    # Kesimpulan
    rata2_ispu = df['ISPU_max'].mean()
    max_ispu = df['ISPU_max'].max()
    hari_tidak_sehat = len(df[df['Kategori_ISPU'].isin(['Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya'])])
    polusi_tertinggi = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].max().idxmax()
    hari_dengan_polusi_tinggi = len(df[df['Polusi_Total'] > df['Polusi_Total'].quantile(0.75)])


    pm25_over_limit = len(st.session_state.df[st.session_state.df['PM2.5'] > 15.5])
    pm10_over_limit = len(st.session_state.df[st.session_state.df['PM10'] > 50])
    o3_over_limit = len(st.session_state.df[st.session_state.df['O3'] > 120])
    co_over_limit = len(st.session_state.df[st.session_state.df['CO'] > 5])
    so2_over_limit = len(st.session_state.df[st.session_state.df['SO2'] > 20])
    no2_over_limit = len(st.session_state.df[st.session_state.df['NO2'] > 30])
    max_pollutant = st.session_state.df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].max().idxmax()
    min_pollutant = st.session_state.df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].min().idxmax()
    kategori_urut = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    kategori_counts = st.session_state.df['Kategori_ISPU'].value_counts().reindex(kategori_urut, fill_value=0)

    kesimpulan = f"""
    1. Rata-rata nilai ISPU selama {st.session_state.n_samples} hari adalah {rata2_ispu:.2f}, menunjukkan bahwa kualitas udara cenderung berada pada kategori {'Baik/Sedang' if rata2_ispu < 100 else 'Tidak Sehat'}.
    2. Nilai ISPU tertinggi adalah {max_ispu:.2f}, masuk dalam kategori '{st.session_state.df.loc[st.session_state.df['ISPU_max'].idxmax(), 'Kategori_ISPU']}'. Ini merupakan kondisi kritis yang memerlukan antisipasi lebih lanjut.
    3. Terdapat {hari_tidak_sehat} hari dengan kategori ISPU tidak sehat/sangat tidak sehat/berbahaya. Kelompok rentan seperti lansia dan penderita penyakit paru-paru perlu waspada.
    4. Polutan dominan berdasarkan konsentrasi tertinggi adalah '{polusi_tertinggi}', menunjukkan perlunya fokus mitigasi pada parameter tersebut.
    5. Terdapat {hari_dengan_polusi_tinggi} hari dengan polusi total sangat tinggi (>75% kuartil), yang mungkin dipengaruhi oleh peningkatan beberapa polutan secara bersamaan.
    6. PM2.5 melebihi batas aman WHO (15.5 µg/m³) sebanyak {pm25_over_limit} hari, menunjukkan perlunya pengendalian emisi kendaraan dan industri.
    7. PM10 melebihi batas aman WHO (50 µg/m³) sebanyak {pm10_over_limit} hari, menunjukkan polusi udara yang cukup signifikan di wilayah simulasi.
    8. Ozon (O3) melebihi ambang batas aman sebanyak {o3_over_limit} hari, menunjukkan aktivitas fotokimia yang tinggi terutama di siang hari.
    9. CO melebihi ambang batas sebanyak {co_over_limit} hari, mengindikasikan adanya peningkatan emisi kendaraan bermotor.
    10. SO2 melebihi ambang batas sebanyak {so2_over_limit} hari, menunjukkan kontribusi aktivitas industri atau pembangkit listrik dalam memengaruhi kualitas udara.
    11. NO2 melebihi ambang batas sebanyak {no2_over_limit} hari, mengindikasikan adanya peningkatan aktivitas kendaraan bermotor dan pembakaran bahan bakar fosil.
    12. Polutan {max_pollutant} memiliki konsentrasi tertinggi di seluruh simulasi, menunjukkan bahwa polutan tersebut paling berkontribusi terhadap penurunan kualitas udara.
    13. Polutan {min_pollutant} memiliki konsentrasi terendah secara rata-rata, menunjukkan bahwa sumber emisi polutan tersebut relatif terkendali.
    """
    
    rekomendasi = """
    Rekomendasi:
    - Pengurangan emisi kendaraan bermotor dan aktivitas industri
    - Edukasi publik tentang cara melindungi diri dari paparan polusi udara
    - Peningkatan pemantauan kualitas udara di area dengan risiko tinggi
    """
    
    save_to_word(doc, "Kesimpulan Simulasi", content=kesimpulan)
    save_to_word(doc, "Rekomendasi", content=rekomendasi)
    
    # Simpan ke buffer
    doc_buffer = BytesIO()
    doc.save(doc_buffer)
    doc_buffer.seek(0)
    
    return doc_buffer

# ====== Inisialisasi Session State ======
if 'df' not in st.session_state:
    st.session_state.df = None
if 'params' not in st.session_state:
    st.session_state.params = None
if 'df_excel' not in st.session_state:
    st.session_state.df_excel = None
if 'n_samples' not in st.session_state:
    st.session_state.n_samples = 365

# ====== STEP 1: BACA FILE EXCEL DAN FIT PARAMETER NORMAL ======
@st.cache_data
def load_data():
    file_excel = "TugasBesar.xlsx"  # Update with your file path
    df_excel = pd.read_excel(file_excel, sheet_name="Sort")
    
    # Filter hanya untuk baris dengan stasiun 'DKI1 Bunderan HI'
    df_excel = df_excel[df_excel['stasiun'] == 'DKI1 Bunderan HI']
    
    # Ambil dan ganti nama kolom sesuai format standar
    df_excel = df_excel[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].copy()
    df_excel.columns = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']
    
    # Konversi ke numerik dan hapus baris NaN
    df_excel = df_excel.apply(pd.to_numeric, errors='coerce').dropna()
    return df_excel

if st.session_state.df_excel is None:
    st.session_state.df_excel = load_data()

# Estimasi parameter distribusi normal untuk semua polutan
if st.session_state.params is None:
    st.session_state.params = {}
    for col in st.session_state.df_excel.columns:
        mu, std = norm.fit(st.session_state.df_excel[col])
        st.session_state.params[col] = ('Normal', mu, std)

# ====== STEP 2: GENERATE DATA SIMULASI ======
def generate_simulation(n_samples=365):
    rng = np.random.default_rng()
    df = pd.DataFrame()
    
    for col in st.session_state.df_excel.columns:
        dist_type, mu, std = st.session_state.params[col]
        
        # 1. Generate data normal
        data = rng.normal(mu, std, n_samples)
        
        # 2. Handle nilai negatif dengan lebih elegan
        neg_indices = data < 0
        
        if np.any(neg_indices):
            # Hitung probabilitas kumulatif untuk nilai negatif
            cdf_neg = norm.cdf(0, loc=mu, scale=std)
            
            # Generate nilai baru dari truncated normal (0 sampai infinity)
            truncated_values = mu + std * rng.standard_normal(size=np.sum(neg_indices))
            truncated_values = np.abs(truncated_values)  # Refleksi nilai
            
            # Gabungkan dengan data positif
            data[neg_indices] = truncated_values
        
        df[col] = np.round(data, 2)
    
    return df



# User input for number of samples
st.session_state.n_samples = st.sidebar.number_input("Jumlah Sampel Hari", min_value=30, max_value=1000, value=365)

if st.session_state.df is None:
    st.session_state.df = generate_simulation(st.session_state.n_samples)

# Fungsi untuk menghitung ulang data jika n_samples berubah
def update_simulation():
    st.session_state.df = generate_simulation(st.session_state.n_samples)

if st.sidebar.button("Generate Ulang Data"):
    update_simulation()

# ====== STEP 3: HITUNG ISPU ======
def calculate_ispu_pm25(pm25):
    if pm25 <= 15.5:
        return pm25 * 50 / 15.5
    elif pm25 <= 55.4:
        return 50 + (pm25 - 15.5) * 50 / (55.4 - 15.5)
    elif pm25 <= 150.4:
        return 100 + (pm25 - 55.4) * 100 / (150.4 - 55.4)
    elif pm25 <= 250.4:
        return 200 + (pm25 - 150.4) * 100 / (250.4 - 150.4)
    else:
        return 300 + (pm25 - 250.4) * 100 / (500 - 250.4)

def calculate_ispu_pm10(pm10):
    if pm10 <= 50:
        return pm10 * 50 / 50
    elif pm10 <= 150:
        return 50 + (pm10 - 50) * 50 / (150 - 50)
    elif pm10 <= 350:
        return 100 + (pm10 - 150) * 100 / (350 - 150)
    elif pm10 <= 420:
        return 200 + (pm10 - 350) * 100 / (420 - 350)
    else:
        return 300 + (pm10 - 420) * 100 / (500 - 420)

def calculate_ispu_so2(so2):
    if so2 <= 52:
        return so2 * 50 / 52
    elif so2 <= 180:
        return 50 + (so2 - 52) * 50 / (180 - 52)
    elif so2 <= 365:
        return 100 + (so2 - 180) * 100 / (365 - 180)
    elif so2 <= 800:
        return 200 + (so2 - 365) * 100 / (800 - 365)
    else:
        return 300 + (so2 - 800) * 100 / (1200 - 800)

def calculate_ispu_no2(no2):
    if no2 <= 80:
        return no2 * 50 / 80
    elif no2 <= 200:
        return 50 + (no2 - 80) * 50 / (200 - 80)
    elif no2 <= 1130:
        return 100 + (no2 - 200) * 100 / (1130 - 200)
    elif no2 <= 2000:
        return 200 + (no2 - 1130) * 100 / (2000 - 1130)
    else:
        return 300 + (no2 - 2000) * 100 / (3000 - 2000)


def calculate_ispu_co(co):
    if co <= 4.0:
        return co * 50 / 4.0
    elif co <= 8.0:
        return 50 + (co - 4.0) * 50 / (8.0 - 4.0)
    elif co <= 15.0:
        return 100 + (co - 8.0) * 100 / (15.0 - 8.0)
    elif co <= 17.0:
        return 200 + (co - 15.0) * 100 / (17.0 - 15.0)
    else:
        return 300 + (co - 17.0) * 100 / (30.0 - 17.0)

def calculate_ispu_o3(o3):
    if o3 <= 120:
        return o3 * 50 / 120
    elif o3 <= 235:
        return 50 + (o3 - 120) * 50 / (235 - 120)
    elif o3 <= 400:
        return 100 + (o3 - 235) * 100 / (400 - 235)
    elif o3 <= 800:
        return 200 + (o3 - 400) * 100 / (800 - 400)
    else:
        return 300 + (o3 - 800) * 100 / (1000 - 800)


# Hitung semua ISPU
if 'ISPU_PM2_5' not in st.session_state.df.columns:
    st.session_state.df['ISPU_PM2_5'] = st.session_state.df['PM2.5'].apply(calculate_ispu_pm25).round(2)
    st.session_state.df['ISPU_PM10'] = st.session_state.df['PM10'].apply(calculate_ispu_pm10).round(2)
    st.session_state.df['ISPU_SO2'] = st.session_state.df['SO2'].apply(calculate_ispu_so2).round(2)
    st.session_state.df['ISPU_NO2'] = st.session_state.df['NO2'].apply(calculate_ispu_no2).round(2)
    st.session_state.df['ISPU_CO'] = st.session_state.df['CO'].apply(calculate_ispu_co).round(2)
    st.session_state.df['ISPU_O3'] = st.session_state.df['O3'].apply(calculate_ispu_o3).round(2)

    # Ambil ISPU maksimum sebagai indeks utama
    st.session_state.df['ISPU_max'] = st.session_state.df[['ISPU_PM2_5', 'ISPU_PM10', 'ISPU_SO2', 'ISPU_NO2', 'ISPU_CO', 'ISPU_O3']].max(axis=1)
    st.session_state.df['Parameter_Dominan'] = st.session_state.df[['ISPU_PM2_5', 'ISPU_PM10', 'ISPU_SO2', 'ISPU_NO2', 'ISPU_CO', 'ISPU_O3']].idxmax(axis=1).str[5:]

    st.session_state.df['Polusi_Total'] = st.session_state.df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].sum(axis=1)

# ====== TAMPILAN STREAMLIT ======
st.sidebar.header("Parameter Simulasi")
show_raw_data = st.sidebar.checkbox("Tampilkan Data Mentah")

if show_raw_data:
    st.subheader("Data Mentah Simulasi")
    st.dataframe(st.session_state.df)

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Parameter", "Visualisasi ISPU", "Kesimpulan", "Parameter Dari File Excel"])

with tab1:
    st.header("Distribusi Parameter Polutan Simulasi")
    
    # ====== STATISTIK SUMMARY DAN PARAMETER ======
    st.subheader("Ringkasan Statistik Simulasi")
    
    # 1. Statistik Deskriptif (dari data simulasi)
    st.markdown("**Statistik Deskriptif (µg/m³):**")
    summary_stats = st.session_state.df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].describe().round(2)
    st.dataframe(summary_stats)
    
    # 2. Parameter Distribusi (dari data Simulasi)
    st.markdown("**Parameter Distribusi Normal (Dihitung dari Data Simulasi):**")
    
    # Hitung parameter dari data Simulasi
    original_params = []
    for col in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']:
        mu, std = norm.fit(st.session_state.df[col])
        original_params.append({
            'Polutan': col,
            'Mean (µg/m³)': mu,
            'Std Dev (µg/m³)': std
        })
    
    params_df = pd.DataFrame(original_params)
    st.dataframe(params_df.round(2))
    
    st.markdown("""
    **Keterangan:**
    - **Mean**: Rata-rata konsentrasi polutan dalam data asli
    - **Std Dev**: Standar deviasi konsentrasi polutan
    - Parameter ini digunakan untuk membangkitkan data simulasi
    """)


    # Plot distribusi parameter
    st.subheader("Distribusi Parameter dengan Kurva Normal")
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle('Distribusi Parameter Kualitas Udara (Normal)', fontsize=14)
    axes1 = axes1.ravel()

    st.markdown("""
    **Penjelasan:**
    - Grafik membandingkan distribusi aktual (histogram) dengan kurva normal teoritis
    - Garis hijau: Distribusi Normal yang difitkan ke data
    - Semua polutan sekarang dimodelkan dengan distribusi normal
    - PM2.5 dan PM10 cenderung memiliki skewness positif
    - SO2 dan CO memiliki distribusi yang lebih simetris
    """)
    
    for i, col in enumerate(['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']):
        if col in st.session_state.params:
            dist_type, mu, std = st.session_state.params[col]
            ax = axes1[i]
            sns.histplot(st.session_state.df[col], bins=20, kde=False, stat="density", alpha=0.6, label='Histogram', ax=ax)
            x = np.linspace(0, st.session_state.df[col].max(), 100)
            pdf = norm.pdf(x, mu, std)
            ax.plot(x, pdf, 'g-', lw=2, label='Normal Fit')
            ax.set_title(f'{col} (Normal)')
            ax.set_xlabel('Konsentrasi')
            ax.set_ylabel('Density')
            ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig1)
    st.caption("Histogram menunjukkan distribusi konsentrasi polutan. Garis hijau menunjukkan kurva distribusi normal.")
    plt.close()

with tab2:
    st.header("Visualisasi Indeks ISPU")
    
    # Tren Harian ISPU
    st.subheader("Tren Harian ISPU")
    fig3 = plt.figure(figsize=(14, 6))

    st.markdown("""
    **Penjelasan Tren ISPU:**
    - Garis biru: Nilai ISPU harian dengan fluktuasi alami
    - Garis merah: Rata-rata 7 hari untuk melihat tren
    - Lonjakan ISPU bisa disebabkan oleh:
      - Peningkatan aktivitas kendaraan
      - Pembakaran biomassa
      - Kondisi meteorologi yang tidak mendispersi polutan
    """)
    
    plt.plot(st.session_state.df.index, st.session_state.df['ISPU_max'], label='ISPU Harian', alpha=0.5)
    plt.plot(st.session_state.df['ISPU_max'].rolling(7).mean(), label='Rata-rata 7 Hari', color='red')
    plt.title('Tren Harian ISPU dengan Smoothing')
    plt.xlabel('Hari')
    plt.ylabel('ISPU')
    plt.grid(True)
    plt.legend()
    st.pyplot(fig3)
    st.caption("Plot menunjukkan tren ISPU harian dengan rata-rata 7 hari terakhir.")
    
    # Kategori ISPU
    st.subheader("Distribusi Kategori ISPU")
    st.markdown("""
    **Kategori ISPU dan Dampaknya:**
    1. **Baik** (Hijau): Aktivitas luar ruangan aman
    2. **Sedang** (Kuning): Kurangi aktivitas panjang di luar bagi penderita pernapasan
    3. **Tidak Sehat** (Oranye): Hindari aktivitas luar ruangan berkepanjangan
    4. **Sangat Tidak Sehat** (Merah): Hindari semua aktivitas luar ruangan
    5. **Berbahaya** (Merah Tua): Tetap di dalam ruangan dengan pembersih udara
    """)
    # Hitung jumlah hari per kategori
    kategori_urut = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    st.session_state.df['Kategori_ISPU'] = st.session_state.df['ISPU_max'].apply(lambda x: "Baik" if x <= 50 else 
                                              "Sedang" if x <= 100 else 
                                              "Tidak Sehat" if x <= 200 else 
                                              "Sangat Tidak Sehat" if x <= 300 else 
                                              "Berbahaya")
    kategori_counts = st.session_state.df['Kategori_ISPU'].value_counts().reindex(kategori_urut, fill_value=0)
    
    # Warna dan style
    warna_kategori = {
        'Baik': 'green',
        'Sedang': 'gold',
        'Tidak Sehat': 'orange',
        'Sangat Tidak Sehat': 'red',
        'Berbahaya': 'darkred'
    }
    colors = [warna_kategori[k] for k in kategori_counts.index]
    
    # Pie Chart
    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.pie(kategori_counts, labels=kategori_counts.index,
            autopct='%1.1f%%', startangle=90, colors=colors,
            shadow=True, textprops={'color': 'black', 'fontsize': 10})
    ax1.set_title('Distribusi Persentase Kategori ISPU')
    ax1.axis('equal')
    
    # Bar Chart
    bars = ax2.bar(kategori_counts.index, kategori_counts.values, color=colors, edgecolor='black')
    ax2.set_title('Jumlah Hari per Kategori ISPU')
    ax2.set_xlabel('Kategori ISPU')
    ax2.set_ylabel('Jumlah Hari')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}\n({height/len(st.session_state.df):.1%})',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig5)
    st.caption("Diagram lingkaran dan batang menunjukkan distribusi kategori ISPU selama periode simulasi.")
    
    # ====== 4 DIAGRAM BARU ======
    
    # 2. Bar Chart Polutan Dominan (menggunakan variabel utama)
    st.subheader("Polutan Dominan Penyebab ISPU Tertinggi")
    st.markdown("""
    **Penjelasan Diagram:**
    - **Sumbu X**: Nama polutan (PM2.5, PM10, SO2, NO2, CO, O3)
    - **Sumbu Y**: Jumlah hari polutan tersebut menjadi penyebab ISPU tertinggi
    - Warna biru muda: Frekuensi dominansi masing-masing polutan
    - Garis grid: Membantu membaca nilai secara akurat
    """)
    fig_bar = plt.figure(figsize=(10, 6))
    polutan_dominan = st.session_state.df['Parameter_Dominan'].value_counts()
    polutan_dominan.plot(kind='bar', color='skyblue')
    plt.title('Frekuensi Polutan Dominan Penyebab ISPU Tertinggi')
    plt.xlabel('Polutan')
    plt.ylabel('Jumlah Hari')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_bar)

    
    # 3. Line Chart Tren Polutan Utama (diagram bebas)
    st.subheader("Tren Polutan Utama")
    st.markdown("""
    **Penjelasan Diagram:**
    - **Sumbu X**: Hari ke- (dikonversi dari hari simulasi)
    - **Sumbu Y**: Konsentrasi harian (µg/m³)
    - Garis berwarna: Mewakili polutan berbeda
    - Area terarsir: Variasi harian dalam hari tersebut
    - Pola mingguan membantu identifikasi pengaruh aktivitas manusia
    """)
    fig_line = plt.figure(figsize=(14, 6))
    for polutan in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']:
        plt.plot(st.session_state.df.index, st.session_state.df[polutan], label=polutan, alpha=0.7)
    plt.title('Tren Polutan Utama')
    plt.xlabel('Hari')
    plt.ylabel('Konsentrasi')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_line)

    
    # 4. Histogram PM2.5 (diagram bebas)
    st.subheader("Distribusi PM2.5")
    st.markdown("""
    **Penjelasan Diagram:**
    - **Sumbu X**: Rentang konsentrasi PM2.5 (µg/m³)
    - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
    - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
    - Garis KDE (Kernel Density Estimate): Estimasi kurva distribusi
    - Garis Merah batas aman ISPU
    """)
    fig_hist = plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['PM2.5'], bins=30, kde=True, color='teal')
    plt.axvline(x=15.5, color='red', linestyle='--', label='Batas Aman ISPU')    
    plt.title('Distribusi Konsentrasi PM2.5')
    plt.xlabel('PM2.5 (µg/m³)')
    plt.ylabel('Frekuensi')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_hist)

    # 5. Histogram PM10
    st.subheader("Distribusi PM10")
    st.markdown("""
    **Penjelasan Diagram:**
    - **Sumbu X**: Rentang konsentrasi PM10 (µg/m³)
    - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
    - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
    - Garis KDE (Kernel Density Estimate): Estimasi kurva distribusi
    - Garis Merah batas aman ISPU
    """)
    fig_hist = plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['PM10'], bins=30, kde=True, color='teal')
    plt.axvline(x=50, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi PM10')
    plt.xlabel('PM10 (µg/m³)')
    plt.ylabel('Frekuensi')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_hist)

    # 6. Histogram SO2
    st.subheader("Distribusi SO2")
    st.markdown("""
    **Penjelasan Diagram:**
    - **Sumbu X**: Rentang konsentrasi SO2 (µg/m³)
    - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
    - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
    - Garis KDE: Estimasi kurva distribusi
    - Garis Merah batas aman ISPU
    """)
    fig_hist = plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['SO2'], bins=30, kde=True, color='teal')
    plt.axvline(x=35, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi SO2')
    plt.xlabel('SO2 (µg/m³)')
    plt.ylabel('Frekuensi')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_hist)

    # 7. Histogram NO2
    st.subheader("Distribusi NO2")
    st.markdown("""
    **Penjelasan Diagram:**
    - **Sumbu X**: Rentang konsentrasi NO2 (µg/m³)
    - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
    - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
    - Garis KDE: Estimasi kurva distribusi
    - Garis Merah batas aman ISPU
    """)
    fig_hist = plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['NO2'], bins=30, kde=True, color='teal')
    plt.axvline(x=80, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi NO2')
    plt.xlabel('NO2 (µg/m³)')
    plt.ylabel('Frekuensi')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_hist)

    # 8. Histogram CO
    st.subheader("Distribusi CO")
    st.markdown("""
    **Penjelasan Diagram:**
    - **Sumbu X**: Rentang konsentrasi CO (µg/m³)
    - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
    - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
    - Garis KDE: Estimasi kurva distribusi
    - Garis Merah batas aman ISPU
    """)
    fig_hist = plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['CO'], bins=30, kde=True, color='teal')
    plt.axvline(x=4, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi CO')
    plt.xlabel('CO (mg/m³)')
    plt.ylabel('Frekuensi')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_hist)

    # 9. Histogram O3
    st.subheader("Distribusi O3")
    st.markdown("""
    **Penjelasan Diagram:**
    - **Sumbu X**: Rentang konsentrasi O3 (µg/m³)
    - **Sumbu Y**: Frekuensi kemunculan (jumlah hari)
    - Batang histogram: Menunjukkan seberapa sering suatu konsentrasi terjadi
    - Garis KDE: Estimasi kurva distribusi
    - Garis Merah batas aman ISPU
    """)
    fig_hist = plt.figure(figsize=(10, 6))
    sns.histplot(st.session_state.df['O3'], bins=30, kde=True, color='teal')
    plt.axvline(x=120, color='red', linestyle='--', label='Batas Aman ISPU')
    plt.title('Distribusi Konsentrasi O3')
    plt.xlabel('O3 (µg/m³)')
    plt.ylabel('Frekuensi')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_hist)

with tab3:
    st.header("Kesimpulan Simulasi")
    
    # Hitung statistik penting
    kategori_ispu_counts = st.session_state.df['Kategori_ISPU'].value_counts()
    rata2_ispu = st.session_state.df['ISPU_max'].mean()
    max_ispu = st.session_state.df['ISPU_max'].max()
    hari_tidak_sehat = len(st.session_state.df[st.session_state.df['Kategori_ISPU'].isin(['Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya'])])
    polusi_tertinggi = st.session_state.df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].max().idxmax()
    hari_dengan_polusi_tinggi = len(st.session_state.df[st.session_state.df['Polusi_Total'] > st.session_state.df['Polusi_Total'].quantile(0.75)])


    pm25_over_limit = len(st.session_state.df[st.session_state.df['PM2.5'] > 15.5])
    pm10_over_limit = len(st.session_state.df[st.session_state.df['PM10'] > 50])
    o3_over_limit = len(st.session_state.df[st.session_state.df['O3'] > 120])
    co_over_limit = len(st.session_state.df[st.session_state.df['CO'] > 5])
    so2_over_limit = len(st.session_state.df[st.session_state.df['SO2'] > 20])
    no2_over_limit = len(st.session_state.df[st.session_state.df['NO2'] > 30])
    max_pollutant = st.session_state.df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].max().idxmax()
    min_pollutant = st.session_state.df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].min().idxmax()
    kategori_urut = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    kategori_counts = st.session_state.df['Kategori_ISPU'].value_counts().reindex(kategori_urut, fill_value=0)

    # Tampilkan kesimpulan
    st.write(f"""
    1. Rata-rata nilai ISPU selama {st.session_state.n_samples} hari adalah {rata2_ispu:.2f}, menunjukkan bahwa kualitas udara cenderung berada pada kategori {'Baik/Sedang' if rata2_ispu < 100 else 'Tidak Sehat'}.
    2. Nilai ISPU tertinggi adalah {max_ispu:.2f}, masuk dalam kategori '{st.session_state.df.loc[st.session_state.df['ISPU_max'].idxmax(), 'Kategori_ISPU']}'. Ini merupakan kondisi kritis yang memerlukan antisipasi lebih lanjut.
    3. Terdapat {hari_tidak_sehat} hari dengan kategori ISPU tidak sehat/sangat tidak sehat/berbahaya. Kelompok rentan seperti lansia dan penderita penyakit paru-paru perlu waspada.
    4. Polutan dominan berdasarkan konsentrasi tertinggi adalah '{polusi_tertinggi}', menunjukkan perlunya fokus mitigasi pada parameter tersebut.
    5. Terdapat {hari_dengan_polusi_tinggi} hari dengan polusi total sangat tinggi (>75% kuartil), yang mungkin dipengaruhi oleh peningkatan beberapa polutan secara bersamaan.
    6. PM2.5 melebihi batas aman WHO (15.5 µg/m³) sebanyak {pm25_over_limit} hari, menunjukkan perlunya pengendalian emisi kendaraan dan industri.
    7. PM10 melebihi batas aman WHO (50 µg/m³) sebanyak {pm10_over_limit} hari, menunjukkan polusi udara yang cukup signifikan di wilayah simulasi.
    8. Ozon (O3) melebihi ambang batas aman sebanyak {o3_over_limit} hari, menunjukkan aktivitas fotokimia yang tinggi terutama di siang hari.
    9. CO melebihi ambang batas sebanyak {co_over_limit} hari, mengindikasikan adanya peningkatan emisi kendaraan bermotor.
    10. SO2 melebihi ambang batas sebanyak {so2_over_limit} hari, menunjukkan kontribusi aktivitas industri atau pembangkit listrik dalam memengaruhi kualitas udara.
    11. NO2 melebihi ambang batas sebanyak {no2_over_limit} hari, mengindikasikan adanya peningkatan aktivitas kendaraan bermotor dan pembakaran bahan bakar fosil.
    12. Polutan {max_pollutant} memiliki konsentrasi tertinggi di seluruh simulasi, menunjukkan bahwa polutan tersebut paling berkontribusi terhadap penurunan kualitas udara.
    13. Polutan {min_pollutant} memiliki konsentrasi terendah secara rata-rata, menunjukkan bahwa sumber emisi polutan tersebut relatif terkendali.
    """)
    
    st.write("""
    **Rekomendasi:**
    - Pengurangan emisi kendaraan bermotor dan aktivitas industri
    - Edukasi publik tentang cara melindungi diri dari paparan polusi udara
    - Peningkatan pemantauan kualitas udara di area dengan risiko tinggi
    - Fokus pada pengendalian polutan dominan yang teridentifikasi
    - Implementasi sistem peringatan dini ketika konsentrasi polutan mencapai level berbahaya
    """)

with tab4:
    st.header("Distribusi Parameter Polutan Excel")
    
    # ====== STATISTIK SUMMARY DAN PARAMETER ======
    st.subheader("Ringkasan Statistik Simulasi Excel")
    
    # 1. Statistik Deskriptif (dari data Excel)
    st.markdown("**Statistik Deskriptif (µg/m³):**")
    summary_stats = st.session_state.df_excel[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].describe().round(2)
    st.dataframe(summary_stats)
    
    # 2. Parameter Distribusi (dari data Excel asli)
    st.markdown("**Parameter Distribusi Normal (Dihitung dari Data Asli):**")
    
    # Hitung parameter dari data Excel
    original_params = []
    for col in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']:
        mu, std = norm.fit(st.session_state.df_excel[col])
        original_params.append({
            'Polutan': col,
            'Mean (µg/m³)': mu,
            'Std Dev (µg/m³)': std
        })
    
    params_df = pd.DataFrame(original_params)
    st.dataframe(params_df.round(2))
    
    st.markdown("""
    **Keterangan:**
    - **Mean**: Rata-rata konsentrasi polutan dalam data asli
    - **Std Dev**: Standar deviasi konsentrasi polutan
    - Parameter ini digunakan untuk membangkitkan data simulasi
    """)

# Download buttons
st.sidebar.header("Unduh Hasil")
if st.sidebar.button("Buat Laporan Word"):
    with st.spinner('Membuat dokumen Word...'):
        doc_buffer = create_word_document(st.session_state.df, st.session_state.params, st.session_state.n_samples)
        st.sidebar.download_button(
            label="Unduh Laporan Word",
            data=doc_buffer,
            file_name=f"Laporan_ISPU_{st.session_state.n_samples}_hari.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(st.session_state.df)

st.sidebar.download_button(
    label="Unduh Data Simulasi (CSV)",
    data=csv,
    file_name=f"simulasi_ispu_{st.session_state.n_samples}_hari.csv",
    mime="text/csv"
)
