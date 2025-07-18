import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec 
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import weibull_min
import seaborn as sns
from scipy.stats import norm
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
    
    # Parameter Distribusi
    weibull_table = []
    for col in df.columns:
        if col in params:
            dist_type, *params_values = params[col]
            weibull_table.append([col, dist_type, *params_values])
    save_to_word(doc, "Parameter Distribusi", 
                content="Parameter distribusi untuk setiap polutan:",
                table=pd.DataFrame(weibull_table, columns=["Polutan", "Distribusi", "Param1", "Param2"]))
    
    # Ringkasan Statistik
    save_to_word(doc, "Ringkasan Statistik Simulasi", 
                content=f"Ringkasan statistik untuk {n_samples} hari simulasi:",
                table=df.describe().round(2))
    
    # Distribusi Parameter
    plt.figure(figsize=(15, 10))
    plt.suptitle('Distribusi Parameter Kualitas Udara dan Kurva Teoritis', fontsize=14)
    for i, col in enumerate(['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], 1):
        if col in params:
            dist_type, *params_values = params[col]
            plt.subplot(2, 3, i)
            sns.histplot(df[col], bins=20, kde=False, stat="density", alpha=0.6, label='Histogram')
            x = np.linspace(0, df[col].max(), 100)
            if dist_type == 'Weibull':
                k, lam = params_values
                pdf = (k / lam) * (x / lam)**(k - 1) * np.exp(-(x / lam)**k)
                plt.plot(x, pdf, 'r-', lw=2, label='Weibull Fit')
                plt.title(f'{col} (Weibull)')
            elif dist_type == 'Normal':
                mu, std = params_values
                pdf = norm.pdf(x, mu, std)
                plt.plot(x, pdf, 'g-', lw=2, label='Normal Fit')
                plt.title(f'{col} (Normal)')
            plt.xlabel('Konsentrasi')
            plt.ylabel('Density')
            plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_to_word(doc, "Distribusi Parameter dengan Kurva Distribusi Teoritis", 
                content="Histogram menunjukkan distribusi konsentrasi polutan. Garis merah/hijau menunjukkan kurva distribusi teoritis.",
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
                content="Plot menunjukkan tren ISPU harian selama simulasi dengan rata-rata 7 hari terakhir.",
                image=True)
    
    # Korelasi Antar Parameter
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].corr(), annot=True, cmap='coolwarm')
    plt.title('Korelasi Antar Parameter Polutan')
    save_to_word(doc, "Korelasi Antar Parameter Polutan", 
                content="Heatmap menunjukkan tingkat korelasi antar parameter polutan.",
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
                content="Diagram lingkaran dan batang menunjukkan distribusi kategori ISPU selama periode simulasi.",
                image=True)
    
    # Skor Risiko
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Skor_Risiko'], bins=range(6, 19), kde=True, color='skyblue')
    plt.title('Distribusi Skor Risiko Harian')
    plt.xlabel('Skor Risiko (Total)')
    plt.ylabel('Jumlah Hari')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_to_word(doc, "Distribusi Skor Risiko Harian", 
                content="Histogram menunjukkan distribusi skor risiko harian.",
                image=True)
    
    # Kategori Risiko
    df['Kategori_Risiko'] = df['Skor_Risiko'].apply(lambda x: "RENDAH" if x <= 9 else 
                                                   "SEDANG" if x <= 14 else 
                                                   "TINGGI")
    kategori_counts = df['Kategori_Risiko'].value_counts()
    colors = {'RENDAH': 'green', 'SEDANG': 'gold', 'TINGGI': 'red'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.pie(kategori_counts, labels=kategori_counts.index, autopct='%1.1f%%',
            startangle=90, colors=[colors[c] for c in kategori_counts.index],
            shadow=True, textprops={'color': 'black'})
    ax1.set_title('Distribusi Persentase Kategori Risiko')
    
    bars = ax2.bar(kategori_counts.index, kategori_counts.values,
                color=[colors[c] for c in kategori_counts.index])
    ax2.set_title('Jumlah Hari per Kategori Risiko')
    ax2.set_xlabel('Kategori Risiko')
    ax2.set_ylabel('Jumlah Hari')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}\n({height/len(df):.1%})', ha='center', va='bottom')
    
    plt.tight_layout()
    save_to_word(doc, "Distribusi Kategori Risiko", 
                content="Diagram lingkaran dan batang menunjukkan distribusi kategori risiko kesehatan.",
                image=True)
    
    # Kesimpulan
    rata2_ispu = df['ISPU_max'].mean()
    max_ispu = df['ISPU_max'].max()
    hari_tidak_sehat = len(df[df['Kategori_ISPU'].isin(['Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya'])])
    polusi_tertinggi = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].max().idxmax()
    hari_dengan_polusi_tinggi = len(df[df['Polusi_Total'] > df['Polusi_Total'].quantile(0.75)])
    
    kesimpulan = f"""
    1. Rata-rata nilai ISPU selama {n_samples} hari adalah {rata2_ispu:.2f}, menunjukkan bahwa kualitas udara cenderung berada pada kategori {'Baik/Sedang' if rata2_ispu < 100 else 'Tidak Sehat'}.
    2. Nilai ISPU tertinggi adalah {max_ispu:.2f}, masuk dalam kategori '{df.loc[df['ISPU_max'].idxmax(), 'Kategori_ISPU']}'. Ini merupakan kondisi kritis yang memerlukan antisipasi lebih lanjut.
    3. Terdapat {hari_tidak_sehat} hari dengan kategori ISPU tidak sehat/sangat tidak sehat/berbahaya. Kelompok rentan seperti lansia dan penderita penyakit paru-paru perlu waspada.
    4. Polutan dominan berdasarkan konsentrasi tertinggi adalah '{polusi_tertinggi}', menunjukkan perlunya fokus mitigasi pada parameter tersebut.
    5. Dari {n_samples} hari simulasi, terdapat {len(df[df['Kategori_Risiko'] == 'TINGGI'])} hari dengan risiko tinggi, {len(df[df['Kategori_Risiko'] == 'SEDANG'])} hari risiko sedang, dan {len(df[df['Kategori_Risiko'] == 'RENDAH'])} hari risiko rendah.
    6. Terdapat {hari_dengan_polusi_tinggi} hari dengan polusi total sangat tinggi (>75% kuartil), yang mungkin dipengaruhi oleh peningkatan beberapa polutan secara bersamaan.
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

# ====== STEP 1: BACA FILE EXCEL DAN FIT PARAMETER WEIBULL ======
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

df_excel = load_data()

# Estimasi parameter distribusi
params = {}
for col in df_excel.columns:
    if col in ['CO', 'SO2']:
        mu, std = norm.fit(df_excel[col])
        params[col] = ('Normal', mu, std)
    else:
        k, loc, lam = weibull_min.fit(df_excel[col], floc=0)
        params[col] = ('Weibull', k, lam)

# ====== STEP 2: GENERATE DATA SIMULASI ======
def generate_simulation(n_samples=365):
    rng = np.random.default_rng()
    df = pd.DataFrame()
    for col in df_excel.columns:
        dist_type, p1, p2 = params[col]
        if dist_type == 'Normal':
            df[col] = np.round(rng.normal(p1, p2, n_samples), 2)
        else:
            df[col] = np.round(p2 * rng.weibull(p1, n_samples), 2)
    return df

# User input for number of samples
n_samples = st.sidebar.number_input("Jumlah Sampel Hari", min_value=30, max_value=1000, value=365)
df = generate_simulation(n_samples)

# ====== STEP 3: HITUNG ISPU DAN SKOR RISIKO ======
def calculate_ispu_pm25(pm25):
    if pm25 <= 15.5: return pm25 * 50 / 15.5
    elif pm25 <= 55.4: return 50 + (pm25 - 15.5) * 50 / (55.4 - 15.5)
    elif pm25 <= 150.4: return 100 + (pm25 - 55.4) * 100 / (150.4 - 55.4)
    elif pm25 <= 250.4: return 200 + (pm25 - 150.4) * 100 / (250.4 - 150.4)
    elif pm25 <= 350.4: return 300 + (pm25 - 250.4) * 100 / (350.4 - 250.4)
    else: return 400 + (pm25 - 350.4) * 100 / (500.4 - 350.4)

def calculate_ispu_pm10(pm10):
    if pm10 <= 50: return pm10
    elif pm10 <= 150: return 50 + (pm10 - 50) * 50 / 100
    elif pm10 <= 350: return 100 + (pm10 - 150) * 100 / 200
    elif pm10 <= 420: return 200 + (pm10 - 350) * 100 / 70
    elif pm10 <= 500: return 300 + (pm10 - 420) * 100 / 80
    else: return 400 + (pm10 - 500) * 100 / 100

def calculate_ispu_so2(so2):
    if so2 <= 20: return so2 * 50 / 20
    elif so2 <= 80: return 50 + (so2 - 20) * 50 / 60
    elif so2 <= 365: return 100 + (so2 - 80) * 100 / 285
    elif so2 <= 800: return 200 + (so2 - 365) * 100 / 435
    elif so2 <= 1000: return 300 + (so2 - 800) * 100 / 200
    else: return 400 + (so2 - 1000) * 100 / 1000

def calculate_ispu_no2(no2):
    if no2 <= 30: return no2 * 50 / 30
    elif no2 <= 60: return 50 + (no2 - 30) * 50 / 30
    elif no2 <= 150: return 100 + (no2 - 60) * 100 / 90
    elif no2 <= 200: return 200 + (no2 - 150) * 100 / 50
    elif no2 <= 400: return 300 + (no2 - 200) * 100 / 200
    else: return 400 + (no2 - 400) * 100 / 200

def calculate_ispu_co(co):
    if co <= 5: return co * 50 / 5
    elif co <= 10: return 50 + (co - 5) * 50 / 5
    elif co <= 17: return 100 + (co - 10) * 100 / 7
    elif co <= 34: return 200 + (co - 17) * 100 / 17
    elif co <= 46: return 300 + (co - 34) * 100 / 12
    else: return 400 + (co - 46) * 100 / 11.5

def calculate_ispu_o3(o3):
    if o3 <= 120: return o3 * 50 / 120
    elif o3 <= 235: return 50 + (o3 - 120) * 50 / 115
    elif o3 <= 400: return 100 + (o3 - 235) * 100 / 165
    elif o3 <= 800: return 200 + (o3 - 400) * 100 / 400
    elif o3 <= 1000: return 300 + (o3 - 800) * 100 / 200
    else: return 400 + (o3 - 1000) * 100 / 200

def hitung_skor_risiko(pm25, pm10, so2, no2, co, o3):
    skor = 0
    
    # PM2.5
    if pm25 > 35: skor += 3
    elif pm25 > 12: skor += 2
    else: skor += 1

    # PM10
    if pm10 > 50: skor += 3
    elif pm10 > 20: skor += 2
    else: skor += 1

    # SO2
    if so2 > 80: skor += 3
    elif so2 > 20: skor += 2
    else: skor += 1

    # NO2
    if no2 > 60: skor += 3
    elif no2 > 30: skor += 2
    else: skor += 1

    # CO
    if co > 10: skor += 3
    elif co > 5: skor += 2
    else: skor += 1

    # O3
    if o3 > 50: skor += 3
    elif o3 > 35: skor += 2
    else: skor += 1

    return skor

# Hitung semua ISPU
df['ISPU_PM2_5'] = df['PM2.5'].apply(calculate_ispu_pm25).round(2)
df['ISPU_PM10'] = df['PM10'].apply(calculate_ispu_pm10).round(2)
df['ISPU_SO2'] = df['SO2'].apply(calculate_ispu_so2).round(2)
df['ISPU_NO2'] = df['NO2'].apply(calculate_ispu_no2).round(2)
df['ISPU_CO'] = df['CO'].apply(calculate_ispu_co).round(2)
df['ISPU_O3'] = df['O3'].apply(calculate_ispu_o3).round(2)
df['Skor_Risiko'] = df.apply(lambda row: hitung_skor_risiko(
    row['PM2.5'], row['PM10'], row['SO2'], row['NO2'], row['CO'], row['O3']
), axis=1)

# Ambil ISPU maksimum sebagai indeks utama
df['ISPU_max'] = df[['ISPU_PM2_5', 'ISPU_PM10', 'ISPU_SO2', 'ISPU_NO2', 'ISPU_CO', 'ISPU_O3']].max(axis=1)
df['Parameter_Dominan'] = df[['ISPU_PM2_5', 'ISPU_PM10', 'ISPU_SO2', 'ISPU_NO2', 'ISPU_CO', 'ISPU_O3']].idxmax(axis=1).str[5:]

df['Polusi_Total'] = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].sum(axis=1)

# ====== TAMPILAN STREAMLIT ======
st.sidebar.header("Parameter Simulasi")
show_raw_data = st.sidebar.checkbox("Tampilkan Data Mentah")

if show_raw_data:
    st.subheader("Data Mentah Simulasi")
    st.dataframe(df)

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Parameter", "Visualisasi ISPU", "Analisis Risiko", "Kesimpulan"])

with tab1:
    st.header("Distribusi Parameter Polutan")
    
    # Plot distribusi parameter - PERBAIKAN DI SINI JUGA
    st.subheader("Distribusi Parameter dengan Kurva Distribusi Teoritis")
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle('Distribusi Parameter Kualitas Udara dan Kurva Teoritis', fontsize=14)
    axes1 = axes1.ravel()
    
    for i, col in enumerate(['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']):
        if col in params:
            dist_type, *params_values = params[col]
            ax = axes1[i]
            sns.histplot(df[col], bins=20, kde=False, stat="density", alpha=0.6, label='Histogram', ax=ax)
            x = np.linspace(0, df[col].max(), 100)
            if dist_type == 'Weibull':
                k, lam = params_values
                pdf = (k / lam) * (x / lam)**(k - 1) * np.exp(-(x / lam)**k)
                ax.plot(x, pdf, 'r-', lw=2, label='Weibull Fit')
                ax.set_title(f'{col} (Weibull)')
            elif dist_type == 'Normal':
                mu, std = params_values
                pdf = norm.pdf(x, mu, std)
                ax.plot(x, pdf, 'g-', lw=2, label='Normal Fit')
                ax.set_title(f'{col} (Normal)')
            ax.set_xlabel('Konsentrasi')
            ax.set_ylabel('Density')
            ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig1)
    st.caption("Histogram menunjukkan distribusi konsentrasi polutan. Garis merah/hijau menunjukkan kurva distribusi teoritis.")
    plt.close()
    
    # Boxplot Ternormalisasi
    st.subheader("Boxplot Ternormalisasi")
    fig2 = plt.figure(figsize=(12, 6))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']])
    sns.boxplot(data=pd.DataFrame(scaled, columns=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']).melt(), 
                x='variable', y='value')
    plt.title('Distribusi Parameter Polutan (Normalized)')
    plt.xlabel('Parameter')
    plt.ylabel('Nilai Normalized')
    st.pyplot(fig2)
    st.caption("Boxplot ini menunjukkan distribusi nilai polutan yang telah dinormalisasi ke rentang [0,1].")

with tab2:
    st.header("Visualisasi Indeks ISPU")
    
    # Tren Harian ISPU
    st.subheader("Tren Harian ISPU")
    fig3 = plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['ISPU_max'], label='ISPU Harian', alpha=0.5)
    plt.plot(df['ISPU_max'].rolling(7).mean(), label='Rata-rata 7 Hari', color='red')
    plt.title('Tren Harian ISPU dengan Smoothing')
    plt.xlabel('Hari')
    plt.ylabel('ISPU')
    plt.grid(True)
    plt.legend()
    st.pyplot(fig3)
    st.caption("Plot menunjukkan tren ISPU harian dengan rata-rata 7 hari terakhir.")
    
    # Korelasi Antar Parameter
    st.subheader("Korelasi Antar Parameter Polutan")
    fig4 = plt.figure(figsize=(10, 8))
    sns.heatmap(df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].corr(), annot=True, cmap='coolwarm')
    plt.title('Korelasi Antar Parameter Polutan')
    st.pyplot(fig4)
    st.caption("Heatmap ini menunjukkan tingkat korelasi antar parameter polutan.")
    
    # Kategori ISPU
    st.subheader("Distribusi Kategori ISPU")
    
    # Hitung jumlah hari per kategori
    kategori_urut = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    df['Kategori_ISPU'] = df['ISPU_max'].apply(lambda x: "Baik" if x <= 50 else 
                                              "Sedang" if x <= 100 else 
                                              "Tidak Sehat" if x <= 200 else 
                                              "Sangat Tidak Sehat" if x <= 300 else 
                                              "Berbahaya")
    kategori_counts = df['Kategori_ISPU'].value_counts().reindex(kategori_urut, fill_value=0)
    
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
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}\n({height/len(df):.1%})',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig5)
    st.caption("Diagram lingkaran dan batang menunjukkan distribusi kategori ISPU selama periode simulasi.")

with tab3:
    st.header("Analisis Risiko Kesehatan")
    
    # Skor Risiko
    st.subheader("Distribusi Skor Risiko Harian")
    fig6 = plt.figure(figsize=(10, 6))
    sns.histplot(df['Skor_Risiko'], bins=range(6, 19), kde=True, color='skyblue')
    plt.title('Distribusi Skor Risiko Harian')
    plt.xlabel('Skor Risiko (Total)')
    plt.ylabel('Jumlah Hari')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig6)
    st.caption("Histogram ini menunjukkan distribusi skor risiko harian.")
    
    # Kategori Risiko
    st.subheader("Distribusi Kategori Risiko")
    
    # Hitung jumlah dan persentase
    df['Kategori_Risiko'] = df['Skor_Risiko'].apply(lambda x: "RENDAH" if x <= 9 else 
                                                   "SEDANG" if x <= 14 else 
                                                   "TINGGI")
    kategori_counts = df['Kategori_Risiko'].value_counts().reindex(["RENDAH", "SEDANG", "TINGGI"], fill_value=0)
    colors = {'RENDAH': 'green', 'SEDANG': 'gold', 'TINGGI': 'red'}
    
    # Plot
    fig7, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.pie(kategori_counts, labels=kategori_counts.index, autopct='%1.1f%%',
            startangle=90, colors=[colors[c] for c in kategori_counts.index],
            shadow=True, textprops={'color': 'black'})
    ax1.set_title('Distribusi Persentase Kategori Risiko')
    
    bars = ax2.bar(kategori_counts.index, kategori_counts.values,
                color=[colors[c] for c in kategori_counts.index])
    ax2.set_title('Jumlah Hari per Kategori Risiko')
    ax2.set_xlabel('Kategori Risiko')
    ax2.set_ylabel('Jumlah Hari')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}\n({height/len(df):.1%})', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig7)
    st.caption("Diagram lingkaran dan batang menunjukkan distribusi kategori risiko kesehatan.")

with tab4:
    st.header("Kesimpulan Simulasi")
    
    # Hitung statistik penting
    kategori_ispu_counts = df['Kategori_ISPU'].value_counts()
    kategori_risiko_counts = df['Kategori_Risiko'].value_counts()
    rata2_ispu = df['ISPU_max'].mean()
    max_ispu = df['ISPU_max'].max()
    hari_tidak_sehat = len(df[df['Kategori_ISPU'].isin(['Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya'])])
    polusi_tertinggi = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].max().idxmax()
    hari_dengan_polusi_tinggi = len(df[df['Polusi_Total'] > df['Polusi_Total'].quantile(0.75)])
    
    # Tampilkan kesimpulan
    st.write(f"""
    1. Rata-rata nilai ISPU selama {n_samples} hari adalah {rata2_ispu:.2f}, menunjukkan bahwa kualitas udara cenderung berada pada kategori {'Baik/Sedang' if rata2_ispu < 100 else 'Tidak Sehat'}.
    2. Nilai ISPU tertinggi adalah {max_ispu:.2f}, masuk dalam kategori '{df.loc[df['ISPU_max'].idxmax(), 'Kategori_ISPU']}'. Ini merupakan kondisi kritis yang memerlukan antisipasi lebih lanjut.
    3. Terdapat {hari_tidak_sehat} hari dengan kategori ISPU tidak sehat/sangat tidak sehat/berbahaya. Kelompok rentan seperti lansia dan penderita penyakit paru-paru perlu waspada.
    4. Polutan dominan berdasarkan konsentrasi tertinggi adalah '{polusi_tertinggi}', menunjukkan perlunya fokus mitigasi pada parameter tersebut.
    5. Dari {n_samples} hari simulasi, terdapat {len(df[df['Kategori_Risiko'] == 'TINGGI'])} hari dengan risiko tinggi, {len(df[df['Kategori_Risiko'] == 'SEDANG'])} hari risiko sedang, dan {len(df[df['Kategori_Risiko'] == 'RENDAH'])} hari risiko rendah.
    6. Terdapat {hari_dengan_polusi_tinggi} hari dengan polusi total sangat tinggi (>75% kuartil), yang mungkin dipengaruhi oleh peningkatan beberapa polutan secara bersamaan.
    """)
    
    st.write("""
    **Rekomendasi:**
    - Pengurangan emisi kendaraan bermotor dan aktivitas industri
    - Edukasi publik tentang cara melindungi diri dari paparan polusi udara
    - Peningkatan pemantauan kualitas udara di area dengan risiko tinggi
    """)

# Download buttons
st.sidebar.header("Unduh Hasil")
if st.sidebar.button("Buat Laporan Word"):
    with st.spinner('Membuat dokumen Word...'):
        doc_buffer = create_word_document(df, params, n_samples)
        st.sidebar.download_button(
            label="Unduh Laporan Word",
            data=doc_buffer,
            file_name=f"Laporan_ISPU_{n_samples}_hari.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df)

st.sidebar.download_button(
    label="Unduh Data Simulasi (CSV)",
    data=csv,
    file_name=f"simulasi_ispu_{n_samples}_hari.csv",
    mime="text/csv"
)
