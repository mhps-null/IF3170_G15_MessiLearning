# IF3170_G15_MessiLearning
## Tugas Besar 2 IF3170 Inteligensi Artifisial (Semester I 2026/2027)

Implementasi algoritma **Supervised Learning *from Scratch*** (KNN, Decision Tree Learning, Logistic Regression, Support Vector Machine, dan Naive Bayes / TAN) beserta pipeline pemrosesan data bebas *leakage* dan evaluasi *Cost-Sensitive Classification* untuk prediksi pembatalan pemesanan penginapan (`is_canceled`, metrik evaluasi resmi kompetisi Kaggle: **Macro F1-Score**).

---

### Anggota Kelompok G15 — MessiLearning

| NIM | Peran & Modul Tanggung Jawab | Branch Kerja |
| :---: | :--- | :--- |
| **13524042** | Data Engineering Pipeline (`src/preprocessing.py`) & KNN (`src/knn.py`) | `feat/anggota1-knn-pipeline` |
| **13524053** | Decision Tree Learning — **C4.5 *from Scratch*** (`src/dtl.py`), Bonus Visualisasi Top-$N$ Tree (`export_tree_visualization`), & Bonus Optimasi DTL (**PSO + Histogram Quantile Split $B=64$ + Minimal Cost-Complexity Weakest-Link Pruning Path**, ablasi Surrogate Splits) | `feat/anggota2-dtl-integration` |
| **13524060** | Logistic Regression (`src/logistic_regression.py`), Adam Optimizer & Loss Contour Video | `feat/anggota3-logreg` |
| **13524057** | Support Vector Machine (`src/svm.py`), SMO/Pegasos & 2D Margin Plotter | `feat/anggota4-svm` |
| **13524063** | Naive Bayes & TAN (`src/naive_bayes.py`), BN DAG Visualizer & Cost-Sensitive Classification | `feat/anggota5-nb-cost` |

---

### Struktur Repositori

```text
IF3170_G15_MessiLearning/
├── data/
│   ├── raw/                                  # Dataset mentah Kaggle (train.csv, test.csv, sample_submission.csv)
│   └── processed/                            # Output pipeline Anggota 1 (clean, clean_pca, & khusus DTL)
├── docs/
│   └── Laporan_Tubes2_G15.pdf                # Laporan akhir kelompok (format PDF)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb            # Notebook EDA & Data Preparation (Anggota 1)
│   ├── 02_model_experiments.ipynb            # Notebook eksperimen & komparasi 5 model (Kolektif)
│   ├── 03_cost_sensitive.ipynb               # Notebook analisis cost-sensitive (Anggota 5)
│   ├── main_reproducible.ipynb               # Ringkasan skrip notebook reproducible
│   └── Copy_of_Notebook_Template_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
│                                             # Notebook template resmi terintegrasi (Anggota 1 + Anggota 2)
├── output/
│   ├── models/
│   │   └── dtl_model.pkl                     # Model C4.5 pemenang tersimpan (Opsi A1: 415 nodes, depth 18)
│   ├── plots/
│   │   └── dtl_tree_top_n.png                # Bonus #2: Visualisasi Top-3 percabangan pohon C4.5
│   ├── videos/                               # Video lintasan kontur loss Logistic Regression (.mp4/.gif)
│   └── submissions/
│       └── submission_dtl_opsi_a1.csv        # Submisi reproducible DTL C4.5 (Kaggle Public LB: 0.83701)
├── src/
│   ├── __init__.py
│   ├── base.py                               # [FONDASI] Abstract Base Class (BaseModel: fit/predict/save/load)
│   ├── utils.py                              # [FONDASI] Metrik evaluasi, K-Fold CV manual, & fungsi biaya custom
│   ├── dummy_generator.py                    # [FONDASI] Generator dataset sintetis untuk pengujian paralel
│   ├── preprocessing.py                      # Pipeline pemrosesan data bebas leakage & dual-export (Anggota 1)
│   ├── knn.py                                # KNN from scratch (Anggota 1)
│   ├── dtl.py                                # DTL C4.5 from scratch, PSO + Hist + Minimal CCP, & Top-N visualizer (Anggota 2)
│   ├── logistic_regression.py                # Logistic Regression & video maker (Anggota 3)
│   ├── svm.py                                # SVM with Kernels & 2D margin plotter (Anggota 4)
│   └── naive_bayes.py                        # Naive Bayes / TAN & DAG visualizer (Anggota 5)
├── FOUNDATION.md                             # Dokumen kontrak teknis & arsitektur tim
├── requirements.txt                          # Daftar dependensi pustaka Python
└── README.md                                 # Dokumentasi utama repositori
```

---

### Ringkasan Hasil & Reproduksibilitas Modul DTL — Anggota 2 (`13524053`)

Modul [`src/dtl.py`](src/dtl.py) diimplementasikan **100% pure NumPy** (tanpa impor `scikit-learn` maupun `scipy`) dalam bentuk **1 pohon keputusan tunggal (non-ensemble)** berbasis varian **C4.5** (*Gain Ratio* + *above-average gain*, *multi-way categorical split*, penanganan `NaN` asli Quinlan):

- **Validasi Holdout Resmi (`val_dtl.csv`, 5.462 baris, bebas *leakage*)**:
  - *Baseline Kuliah* C4.5 Exact + REP (Materi 10c): Val Macro F1 **`0.8186`**, Cost `1.136`, `364` nodes.
  - **Model Pemenang Bonus #6 (Opsi A1 — PSO + Histogram Quantile $B=64$ + Minimal Cost-Complexity Weakest-Link Pruning Path)**: **Val Macro F1 `0.8259`**, Accuracy **`85.34%`**, Precision **`0.8285`**, Cost **`1.059`**, `415` nodes, Depth `18`.
  - **Uji McNemar vs Baseline REP**: $b=123, c=162, \chi^2=5.0667, p=0.02439 < 0.05$ (peningkatan signifikan secara statistik).
  - **Komparasi `scikit-learn` (`val_clean.csv`, 5.462 ID identik)**: Mengungguli `DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.000302)` (`Val Macro F1 = 0.8221`, Cost `1.159`, `643` nodes) dan `ccp_alpha=0.01` (`Val Macro F1 = 0.7688`).
  - **Stabilitas 5-Fold CV (`train_dtl.csv`)**: Train `0.8444 ± 0.0021` | Val `0.8234 ± 0.0047` | Gap `0.0210`.
- **Submisi Kaggle Reproducible (`output/submissions/submission_dtl_opsi_a1.csv`)**:
  - Dilatih pada 100% `train.csv` terdeduplikasi (`27.309` baris) menggunakan 1 pohon C4.5 tunggal (361 nodes, depth 18) dengan kalibrasi *threshold* jujur berbasis 5-Fold Out-Of-Fold CV (`th = 0.415`, OOF Macro F1 meningkat di 5/5 fold: `0.82127` → `0.82954`, *Train-Val gap* `0.0141`).
  - Skor **Public Leaderboard Kaggle: `0.83701`** (SHA-256 terverifikasi *byte-for-byte*: `c5565c01cc742184e25b2aa0cc4c406cc4accfc4a0dbe385513dbf0eb8adac7c`).
- **Bonus #2 Visualisasi Top-$N$ Tree**: Dihasilkan melalui `export_tree_visualization(model, top_n=3, ...)` ke [`output/plots/dtl_tree_top_n.png`](output/plots/dtl_tree_top_n.png).

---

### Petunjuk Instalasi & Cara Menjalankan Program

1. **Buat dan aktifkan *virtual environment* (Python 3.12 direkomendasikan):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux / macOS
   # atau: venv\Scripts\activate  # Windows
   ```

2. **Instal seluruh dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verifikasi modul mandiri (opsional):**
   ```bash
   python3 -m src.dummy_generator
   python3 -m src.dtl
   ```

4. **Menjalankan Notebook Reproducible (Lokal):**
   - Pastikan kernel Jupyter / VS Code mengarah ke `venv` di atas dan direktori kerja berada di dalam repositori `IF3170_G15_MessiLearning/`.
   - Buka [`notebooks/Copy_of_Notebook_Template_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb`](notebooks/Copy_of_Notebook_Template_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb) (atau [`notebooks/02_model_experiments.ipynb`](notebooks/02_model_experiments.ipynb) untuk eksperimen DTL mandiri) lalu pilih **Kernel → Restart Kernel and Run All Cells**.

5. **Menjalankan Notebook di Google Colab:**
   - Buka sel **Environment Setup (`#14`)** pada notebook template, *uncomment* perintah `!git clone https://github.com/mhps-null/IF3170_G15_MessiLearning.git` dan `%cd IF3170_G15_MessiLearning` (atau *mount* Google Drive yang berisi repositori di `/content/drive/MyDrive/IF3170_G15_MessiLearning`), lalu pilih **Runtime → Run all**.
