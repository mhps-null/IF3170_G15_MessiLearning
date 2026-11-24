# IF3170_G15_MessiLearning
## Tugas Besar 2 IF3170 Inteligensi Artifisial (Semester I 2026/2027)

Implementasi terpadu lima algoritma **Supervised Learning *from Scratch*** (**KNN**, **Decision Tree Learning C4.5**, **Logistic Regression FISTA**, **Support Vector Machine SMO**, dan **Bayesian Network / Weighted Gaussian Naive Bayes**) beserta *pipeline* pemrosesan data bebas *data leakage*, komparasi `scikit-learn`, optimasi model lanjutan, dan evaluasi *Cost-Sensitive Classification* untuk memprediksi pembatalan pemesanan penginapan (`is_canceled`, metrik evaluasi resmi kompetisi Kaggle: **Macro F1-Score**).

---

### Anggota Kelompok G15 — MessiLearning

| NIM | Nama | Peran & Modul Tanggung Jawab |
| :---: | :--- | :--- |
| **13524042** | Suryani Mulia Utami | EDA, *Data Engineering Pipeline* bebas *leakage* (`src/preprocessing.py`), implementasi **KNN *from Scratch*** (`src/knn.py`), komparasi `KNeighborsClassifier`, dan submisi Kaggle KNN (`submission_knn.csv`) |
| **13524053** | Muhammad Haris Putra Sulastianto | Implementasi **DTL C4.5 *from Scratch*** (`src/dtl.py`), Bonus Visualisasi Top-$N$ Tree (`export_tree_visualization`), Bonus Optimasi DTL (**PSO + Histogram Quantile Split $B=64$ + Minimal Cost-Complexity Weakest-Link Pruning Path**, ablasi *Surrogate Splits*), konsolidasi notebook terpusat, dan submisi final Kaggle DTL (`submission_dtl_opsi_a1.csv`) |
| **13524057** | Benedict Darrel Setiawan | Implementasi **SVM *from Scratch*** (solver dual SMO orde-dua, kernel Linear/Poly/Gaussian/Laplacian, *Platt scaling*, `src/svm.py` & `src/svm_tuning.py`), Bonus Visualisasi Margin 2D (`plot_svm_margin_2d`), Bonus Optimasi SVM (Kernel Laplacian + 5-Fold Stratified CV), dan submisi final Kaggle SVM (`submission_svm_improved.csv`) |
| **13524060** | Steven Tan | Implementasi **Logistic Regression *from Scratch*** (regularisasi L1, L2, Elastic Net & `BookingFeatureTransformer` pada `src/logistic_regression.py`), Bonus Optimasi **FISTA dengan *backtracking***, Bonus Video Kontur Loss (`src/logistic_contour.py`), dan submisi final Kaggle (`submission_logistic_regression.csv`) |
| **13524063** | Marcel Luther Sitorus | Implementasi **Gaussian Naive Bayes *from Scratch*** (`src/naive_bayes.py`), Bonus Optimasi Naive Bayes (**Weighted Gaussian NB** via *Projected Adam* + **Adaptive Out-of-Fold Smoothed Target Encoding**), Bonus Visualisasi DAG (`export_bayesian_network_graph`), evaluasi **Bonus Cost-Sensitive Classification**, dan submisi Kaggle (`submission_nb.csv`) |

---

### Struktur Repositori

```text
IF3170_G15_MessiLearning/
├── data/
│   ├── raw/                                  # Dataset mentah Kaggle (train.csv, test.csv, sample_submission.csv)
│   └── processed/                            # Output pipeline bebas leakage (clean, clean_pca, & khusus DTL/NB)
├── docs/
│   └── Laporan_Tubes2_G15.pdf                # Laporan akhir kelompok (format PDF)
├── notebooks/
│   └── IF3170_G15_MessiLearning.ipynb        # Satu-satunya notebook eksperimen terpusat & reproducible (#1 s.d. #12)
├── output/
│   ├── models/                               # Model terlatih (.pkl): knn, dtl, logistic_regression, svm, naive_bayes
│   ├── plots/                                # Visualisasi evaluasi & bonus (DTL Top-3, SVM 2D, BN DAG, Cost-Sensitive)
│   ├── videos/
│   │   └── logreg_loss_contour.mp4           # Bonus #3: Video lintasan kontur loss Logistic Regression
│   └── submissions/                          # Berkas submisi reproducible Kaggle (.csv)
├── src/
│   ├── __init__.py
│   ├── base.py                               # Abstract Base Class (BaseModel: fit/predict/predict_proba/save/load)
│   ├── utils.py                              # Metrik evaluasi from scratch, K-Fold CV manual, & fungsi biaya 2*FP + FN
│   ├── dummy_generator.py                    # Generator dataset sintetis untuk uji unit mandiri
│   ├── preprocessing.py                      # DataPipeline bebas leakage (standar, PCA, & jalur khusus DTL)
│   ├── knn.py                                # KNN from scratch — Euclidean & Manhattan (Anggota 1)
│   ├── dtl.py                                # DTL C4.5 from scratch, PSO + Hist + CCP, & Top-N visualizer (Anggota 2)
│   ├── logistic_regression.py                # Logistic Regression FISTA (L1/L2/Elastic Net) & Transformer (Anggota 3)
│   ├── logistic_contour.py                   # Generator animasi kontur loss & lintasan FISTA (Anggota 3)
│   ├── svm.py                                # SVM SMO orde-dua, Platt scaling, 4 kernel & 2D margin plotter (Anggota 4)
│   ├── svm_tuning.py                         # Pencarian hyperparameter SVM 5-Fold Stratified CV murni NumPy (Anggota 4)
│   └── naive_bayes.py                        # Weighted Gaussian NB, Adaptive OOF Encoding, DAG & Cost-Sensitive (Anggota 5)
├── requirements.txt                          # Daftar dependensi pustaka Python
└── README.md                                 # Dokumentasi utama repositori
```

---

### Ringkasan Performa Validasi Resmi (`5.462` Baris Holdout, `SEED = 42`) & Submisi Final Kaggle

| Algoritma (*From Scratch* — Non-Ensemble) | Konfigurasi Terbaik | Val Macro F1 | Val Accuracy | Custom Cost (`2·FP + FN`) | Berkas Submisi & Status Kaggle |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **SVM (SMO)** | Kernel Laplacian ($C=6.9344, \gamma=0.0718$, 5-Fold CV) | **`0.8464`** | **`86.85%`** | **`983`** | `submission_svm_improved.csv` (**Final Selected #1**) |
| **Logistic Regression** | FISTA L2 ($C=0.05, \tau=0.475$, CSR Pairwise Indicators) | `0.8297` | `85.23%` | `1,159` | `submission_logistic_regression.csv` (**Final Selected #2**) |
| **DTL (C4.5)** | PSO + Histogram ($B=64$) + Minimal CCP (`415` nodes) | `0.8259` | `85.34%` | `1,059` | `submission_dtl_opsi_a1.csv` (**Final Selected #3**, Public LB `0.83701`) |
| **Naive Bayes** | Stage E3: Weighted Gaussian NB (Adam L2) + Adaptive OOF | `0.8236` | `85.12%` | `1,082` | `submission_nb.csv` |
| **KNN** | Manhattan / Euclidean ($k=10$) | `0.7915` | `81.86%` | `1,445` | `submission_knn.csv` (Public LB `0.81104`) |

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

3. **Menjalankan Notebook Terpusat Secara Reproducible (Lokal):**
   - Pastikan kernel Jupyter / VS Code mengarah ke `venv` di atas.
   - Buka [`notebooks/IF3170_G15_MessiLearning.ipynb`](notebooks/IF3170_G15_MessiLearning.ipynb) lalu pilih **Kernel → Restart Kernel and Run All Cells**.

4. **Menjalankan di Google Colab:**
   - Buka sel **Environment Setup (`#14`)** pada [`notebooks/IF3170_G15_MessiLearning.ipynb`](notebooks/IF3170_G15_MessiLearning.ipynb), *uncomment* baris `!git clone https://github.com/mhps-null/IF3170_G15_MessiLearning.git` dan `%cd IF3170_G15_MessiLearning` (atau *mount* Google Drive), lalu jalankan **Runtime → Run all**.
