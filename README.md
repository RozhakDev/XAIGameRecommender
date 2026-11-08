# XAIGameRecommender

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Sentence-Transformers](https://img.shields.io/badge/NLP-all--MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

XAIGameRecommender adalah sistem rekomendasi game Steam berbasis PyTorch multimodal attention yang memprediksi preferensi pemain sekaligus menyajikan atribusi faktor pemicu rekomendasi (Explainable AI).

## Overview

Sistem ini memprediksi kecocokan game berdasarkan tiga sumber informasi: pola interaksi pemain, metadata teknis game, dan representasi teks judul game. Mekanisme atensi menghitung bobot kontribusi relatif dari tiap modalitas sehingga setiap hasil rekomendasi memiliki transparansi penilaian.

### Alur Kerja Sistem

Sistem terdiri atas dua bagian utama:

1. **Pelatihan Model (`DeepLearning/`)**:
   Notebook PyTorch untuk memproses data ulasan mentah, melakukan penyaringan ulasan minimal 5 ulasan (K-Core), mengekstrak embedding teks judul game, melatih model `MultiModalAttentionRecSys`, dan menyimpan artefak model ke disk.
2. **Aplikasi Web (`Visualization/`)**:
   Antarmuka web interaktif berbasis Streamlit yang memuat model terlatih dan data fitur untuk menampilkan rekomendasi game beserta penjelasan atensi secara langsung.

### Dataset & Penyaringan K-Core

Model dilatih menggunakan dataset ulasan Steam dari Kaggle (`antonkozyriev/game-recommendations-on-steam`):

- **Data Mentah**: 50.872 metadata game (`games.csv`) dan 41.154.794 log ulasan interaksi (`recommendations.csv`).
- **Penyaringan K-Core**: Menyaring pengguna aktif dengan ambang batas minimal 5 ulasan untuk mereduksi sparsitas data dan mengatasi kendala *cold-start*.
- **Katalog Akhir**: Menghasilkan 22.444.312 interaksi ulasan tervalidasi, 1.910.831 profil pengguna unik, dan 37.478 judul game terindeks.

## Features

### 1. Ekstraksi Fitur Multimodal

Model menggabungkan tiga kelompok fitur ke dalam ruang laten 64 dimensi:

- **Fitur Kolaboratif**: Embedding ID pengguna dan ID game untuk menangkap pola interaksi historis pemain.
- **Fitur Metadata Tabular**: 8 fitur terstandarisasi yang mencakup kompatibilitas OS (`win`, `mac`, `linux`), status `steam_deck`, label `rating`, ulasan positif (`positive_ratio`), harga (`price_final`), dan diskon (`discount`).
- **Fitur Teks Semantik**: Vektor dense 384 dimensi dari judul game menggunakan model `sentence-transformers/all-MiniLM-L6-v2`.

### 2. Mekanisme Atensi Terbobot Pemain

Fusi fitur menggunakan mekanisme atensi terkondisi profil pengguna:

- **Query Proyeksi**: Vektor pengguna diproyeksikan sebagai Query (lapisan linear 64 dimensi).
- **Key Proyeksi**: Ketiga modalitas game diproyeksikan sebagai Key representasi.
- **Bobot Atensi**: Perkalian matriks antara Query dan Key dinormalisasi dengan Softmax untuk menghasilkan distribusi bobot ketiga modalitas.
- **Klasifikasi Rekomendasi**: Konteks representasi terbobot dialirkan ke feed-forward network (Dropout 0.2) dengan aktivasi Sigmoid untuk menghasilkan estimasi probabilitas kecocokan.

### 3. Penjelasan Keputusan Model (Explainable AI)

Modul `ReasoningExplainer` menerjemahkan bobot atensi dominan menjadi penjelasan yang mudah dipahami:

- **Faktor Kolaboratif**: Muncul saat atensi kolaboratif dominan, menandakan rekomendasi dipicu oleh kesamaan pola bermain dengan pengguna lain yang menyukai game tersebut.
- **Faktor Metadata**: Muncul saat atensi tabular dominan, menandakan rekomendasi didorong oleh kesesuaian atribut objektif game (kategori rating dan rentang harga).
- **Faktor Semantik**: Muncul saat atensi teks dominan, menandakan rekomendasi dipicu oleh kemiripan tema atau kata kunci pada judul game.

### 4. Antarmuka Web Interaktif

Aplikasi inferensi Streamlit menyediakan fitur pengujian model:

- **Profil Benchmark**: 5 profil pemain sampel dengan riwayat ulasan aktif untuk pengujian instan tanpa memasukkan User ID secara manual.
- **Pengaturan Top-K**: Slider penentuan jumlah rekomendasi dari 1 hingga 50 judul game (default: 5).
- **Validasi User ID**: Verifikasi otomatis ID pengguna terhadap katalog latih untuk mencegah eror runtime.
- **Hasil Rekomendasi Terperinci**: Menampilkan peringkat, judul game, App ID Steam, badge platform, skor kecocokan, grafik sebaran bobot atensi, dan teks alasan rekomendasi.

## Project Structure

```text
XAIGameRecommender/
├── DeepLearning/            # Pipeline data science & notebook pelatihan
│   ├── notebook/            # Jupyter notebook pemodelan multimodal
│   └── requirements.txt     # Dependensi notebook pelatihan
├── Visualization/           # Aplikasi web inferensi Streamlit
│   ├── artifacts/           # Checkpoint model PyTorch & tensor fitur
│   ├── src/                 # Modul logika inferensi & antarmuka web
│   ├── main.py              # Titik masuk eksekusi Streamlit
│   └── requirements.txt     # Dependensi aplikasi web Streamlit
├── LICENSE                  # Lisensi proyek MIT
└── README.md                # Dokumentasi sistem
```

## Installation

### Prerequisites

- Python versi 3.10 atau yang lebih baru
- Akselerator GPU CUDA (opsional; otomatis menggunakan CPU jika GPU tidak tersedia)
- Git

### Langkah Pemasangan

1. **Clone Repositori**:
   ```bash
   git clone https://github.com/RozhakDev/XAIGameRecommender.git
   cd XAIGameRecommender
   ```

2. **Konfigurasi Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
   ```

3. **Instalasi Paket Dependensi**:
   ```bash
   pip install --upgrade pip
   pip install -r Visualization/requirements.txt
   ```
   Untuk menyiapkan dependensi notebook pelatihan model:
   ```bash
   pip install -r DeepLearning/requirements.txt
   ```

## Usage

### 1. Pelatihan Model (Notebook)

Jika ingin melatih ulang atau mereproduksi eksperimen pemodelan:

1. Buka notebook `DeepLearning/notebook/01_Explainable_Game_Recommendation.ipynb` pada Google Colab atau Jupyter berakselerator GPU.
2. Konfigurasi kredensial `kaggle.json` untuk mengunduh dataset ulasan Steam.
3. Jalankan seluruh sel hingga selesai untuk preprocessing, pelatihan model, dan ekspor artefak.
4. Ekstrak file `artifacts.zip` hasil pelatihan ke dalam folder `Visualization/artifacts/`.

### 2. Menjalankan Aplikasi Web (Streamlit)

Seluruh artefak inferensi sudah tersedia di dalam direktori `Visualization/artifacts/`. Jalankan aplikasi Streamlit:

```bash
cd Visualization
python -m streamlit run main.py --server.port 8501
```

Buka peramban web dan akses antarmuka pada `http://localhost:8501`.

### 3. Alur Pengujian Rekomendasi

1. Pilih salah satu profil pengguna dari daftar Benchmark Persona pada panel samping, atau ketikkan Steam User ID terdaftar secara manual.
2. Atur slider Top-K untuk menentukan jumlah game yang ingin ditampilkan (rentang 1 hingga 50, nilai default: 5).
3. Klik tombol jalankan rekomendasi untuk memicu proses forward pass model.
4. Telaah hasil rekomendasi yang memuat skor probabilitas, pembagian atensi modalitas, dan narasi penjelasan faktor pemicu.

## Technology Stack

| Komponen | Pustaka / Alat | Versi Minimum | Peran Teknis Sistem |
|---|---|---|---|
| Deep Learning Engine | PyTorch | 2.0+ | Arsitektur `MultiModalAttentionRecSys`, komputasi tensor, loss BCELoss, dan optimizer AdamW. |
| NLP & Embedding | Sentence Transformers | 2.2+ | Ekstraksi dense embedding 384 dimensi dari judul game via `all-MiniLM-L6-v2`. |
| Web Application | Streamlit | 1.30+ | Antarmuka pengguna interaktif, state management, dan caching artefak via `@st.cache_resource`. |
| Data Processing | Pandas & NumPy | 2.0+ / 1.24+ | Pemrosesan tabel ulasan Steam, filtering K-Core, dan serialisasi tensor `.npy`. |
| Preprocessing & Evaluasi | Scikit-Learn | 1.2+ | Penskalaan `StandardScaler`, transformasi `LabelEncoder`, dan metrik validasi ROC-AUC/Akurasi. |
| Visualisasi Data | Seaborn & Matplotlib | 3.7+ | Visualisasi konvergensi training loss dan dinamika ROC-AUC per epoch pada notebook. |

## License

Proyek ini dilisensikan di bawah [MIT License](LICENSE). Dokumentasi sumber dan atribusi dataset publik Steam dapat dilihat pada [DeepLearning/DATASET.md](DeepLearning/DATASET.md).
