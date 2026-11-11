# IF3170_G15_MessiLearning
## Tugas Besar 2 IF3170 Inteligensi Artifisial (Semester I 2026/2027)

Implementasi algoritma **Supervised Learning *from Scratch*** (KNN, Decision Tree Learning, Logistic Regression, Support Vector Machine, dan Naive Bayes / TAN) beserta pipeline pemrosesan data bebas *leakage* dan evaluasi *Cost-Sensitive Classification*.

---

### Anggota Kelompok G15 — MessiLearning

| NIM | Peran & Modul Tanggung Jawab | Branch Kerja |
| :---: | :--- | :--- |
| **13524042** | Anggota 1 — Data Engineering Pipeline (`src/preprocessing.py`) & KNN (`src/knn.py`) | `feat/anggota1-knn-pipeline` |
| **13524053** | Anggota 2 — Decision Tree Learning (`src/dtl.py`), Top-$N$ Tree Visualizer & Post-Pruning | `feat/anggota2-dtl` |
| **13524057** | Anggota 3 — Logistic Regression (`src/logistic_regression.py`), Adam Optimizer & Loss Contour Video | `feat/anggota3-logreg` |
| **13524060** | Anggota 4 — Support Vector Machine (`src/svm.py`), SMO/Pegasos & 2D Margin Plotter | `feat/anggota4-svm` |
| **13524063** | Anggota 5 — Naive Bayes & TAN (`src/naive_bayes.py`), BN DAG Visualizer & Cost-Sensitive Classification | `feat/anggota5-nb-cost` |

> *(Catatan: Silakan sesuaikan urutan NIM dengan pembagian Anggota 1–5 yang disepakati kelompok).*

---

### Struktur Repositori

```text
IF3170_G15_MessiLearning/
├── data/
│   ├── raw/                      # Dataset mentah dari Kaggle (train.csv, test.csv)
│   └── processed/                # Output pipeline Anggota 1 (clean & khusus DTL)
├── docs/
│   └── Laporan_Tubes2_G15.pdf    # Laporan akhir kelompok (format PDF)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb # Notebook EDA & Data Preparation (Anggota 1)
│   ├── 02_model_experiments.ipynb # Notebook pelatihan & validasi 5 model (Kolektif)
│   ├── 03_cost_sensitive.ipynb    # Notebook analisis cost-sensitive (Anggota 5)
│   └── main_reproducible.ipynb    # Notebook final reproducible untuk asisten & Kaggle
├── output/
│   ├── models/                   # Serialisasi model tersimpan (.pkl)
│   ├── plots/                    # Visualisasi bonus DTL, SVM 2D, dan Bayesian Network (.png)
│   ├── videos/                   # Visualisasi video kontur loss Logistic Regression (.mp4/.gif)
│   └── submissions/              # Berkas submisi prediksi Kaggle (submission_final.csv)
├── src/
│   ├── __init__.py
│   ├── base.py                   # [FONDASI] Abstract Base Class (BaseModel)
│   ├── utils.py                  # [FONDASI] Metrik evaluasi, K-Fold CV, & fungsi biaya custom
│   ├── dummy_generator.py        # [FONDASI] Generator dataset sintetis untuk kerja paralel
│   ├── preprocessing.py          # Modul pipeline pemrosesan data & PCA (Anggota 1)
│   ├── knn.py                    # Modul KNN from scratch (Anggota 1)
│   ├── dtl.py                    # Modul Decision Tree from scratch & visualizer (Anggota 2)
│   ├── logistic_regression.py    # Modul Logistic Regression & video maker (Anggota 3)
│   ├── svm.py                    # Modul SVM with Kernels & 2D plotter (Anggota 4)
│   └── naive_bayes.py            # Modul Naive Bayes / TAN & graph visualizer (Anggota 5)
├── FOUNDATION.md                 # Dokumen kontrak teknis & spesifikasi lengkap tim
├── requirements.txt              # Daftar dependensi pustaka Python
└── README.md                     # Dokumentasi utama repositori
```

---

### Petunjuk Instalasi & Penggunaan Paralel

1. **Buat dan aktifkan *virtual environment*:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # atau: venv\Scripts\activate  (Windows)
   ```

2. **Instal seluruh dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verifikasi fondasi & generator data sintetis (untuk pengujian mandiri tanpa menunggu data bersih):**
   ```bash
   python3 -m src.dummy_generator
   ```

4. **Menjalankan Notebook Reproducible:**
   Buka `notebooks/main_reproducible.ipynb` dan pilih **Kernel -> Restart Kernel and Run All Cells**.
