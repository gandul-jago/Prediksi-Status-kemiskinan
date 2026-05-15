# 🏠 Prediksi Status Kemiskinan dengan XGBoost & Data NHANES

> Mengklasifikasikan status kemiskinan rumah tangga menggunakan model machine learning XGBoost berbasis data survei kesehatan nasional Amerika Serikat (NHANES).

---

## 📌 Deskripsi

Penelitian ini bertujuan untuk memprediksi apakah suatu individu termasuk dalam kategori **miskin atau tidak miskin** berdasarkan rasio pendapatan terhadap garis kemiskinan (`INDFMPIR < 1`). Model dilatih menggunakan fitur-fitur dari data survei NHANES yang mencakup aspek pendapatan, demografi, status kesehatan, pola makan, dan aktivitas fisik.

---

## 📊 Dataset

Data bersumber dari survei **NHANES (National Health and Nutrition Examination Survey)** yang terdiri dari 7 file `.xpt` (format SAS):

| File | Isi |
|---|---|
| `DEMO_J.xpt` | Data demografis (usia, jenis kelamin, etnis, pendidikan) |
| `BMX_J.xpt` | Indeks massa tubuh (BMI) |
| `BPX_J.xpt` | Tekanan darah (sistolik & diastolik) |
| `DIQ_J.xpt` | Status diabetes |
| `SMQ_J.xpt` | Riwayat merokok |
| `PAQ_J.xpt` | Aktivitas fisik |
| `DR1TOT_J.xpt` | Asupan gizi harian |

📁 **Link Dataset:** [Google Drive](https://drive.google.com/drive/folders/1-Zxe4Kt7Uq17MGepZ914aq22CV_nzUSr?usp=sharing)

---

## ⚙️ Alur Pipeline

```
Load Data → Cleaning → Feature Engineering → Preprocessing → Training → Evaluasi
```

### 1. 🔗 Load & Gabungkan Data
Ketujuh file digabungkan berdasarkan `SEQN` (ID unik responden) menggunakan left merge.

### 2. 🧹 Cleaning
- Seleksi 19 kolom relevan
- Drop kolom dengan missing value > 50%
- Hapus baris tanpa nilai `INDFMPIR` (kolom label)

### 3. 🛠️ Feature Engineering
Dibuat 5 fitur gabungan baru dari kolom mentah:

| Fitur Baru | Sumber | Keterangan |
|---|---|---|
| `BP_STATUS` | BPXSY1 + BPXDI1 | Status tekanan darah (1=Normal, 2=Pre-hipertensi, 3=Hipertensi) |
| `Ratio_Carb/Prot/Fat` | DR1TCARB + DR1TPROT + DR1TTFAT | Proporsi makronutrien terhadap total asupan |
| `PA_SCORE` | PAQ605 + PAQ620 | Skor aktivitas fisik (0–2) |
| `Health_Risk_Count` | DIQ010 + SMQ020 | Jumlah faktor risiko kesehatan (0–2) |
| `Income_Per_Capita` | INDHHIN2 / DMDHHSIZ | Estimasi pendapatan per anggota keluarga |

### 4. 🔄 Preprocessing
- Split data: **80% train / 20% test** dengan `stratify`
- Imputasi missing value menggunakan `SimpleImputer(strategy='median')` — hanya di-fit pada train set untuk menghindari data leakage
- Penanganan class imbalance menggunakan `scale_pos_weight` pada XGBoost

### 5. 🤖 Training — XGBoost + Hyperparameter Tuning
- Model: `XGBClassifier` dengan `scale_pos_weight=ratio`
- Tuning: `RandomizedSearchCV` (20 iterasi, 5-fold CV, scoring=F1)
- Threshold optimal dicari via **Precision-Recall Curve** (bukan default 0.5)

---

## 📈 Hasil Evaluasi

### Confusion Matrix
|  | Prediksi Tidak Miskin | Prediksi Miskin |
|---|---|---|
| **Aktual Tidak Miskin** | 712 ✅ | 21 ❌ |
| **Aktual Miskin** | 16 ❌ | 147 ✅ |

### Metrik Utama
| Metrik | Nilai |
|---|---|
| **AUC-ROC** | **0.973** |
| Optimal Threshold | 0.5745 |
| False Negative (miskin salah prediksi) | 16 |

---

## 🔍 SHAP — Faktor Penentu Kemiskinan

Berdasarkan analisis SHAP, fitur yang paling berpengaruh terhadap prediksi:

1. 🥇 `Income_Per_Capita` — Pendapatan per kapita (dominan)
2. 🥈 `DMDHHSIZ` — Jumlah anggota keluarga
3. 🥉 `RIDAGEYR` — Usia
4. `BMXBMI` — Indeks massa tubuh
5. `DMDEDUC2` — Tingkat pendidikan

> Pendapatan per kapita **rendah** (biru) secara konsisten mendorong model memprediksi **Miskin**, dan sebaliknya.

---

## 🗂️ Struktur File

```
├── data/
│   ├── DEMO_J.xpt
│   ├── BMX_J.xpt
│   ├── BPX_J.xpt
│   ├── DIQ_J.xpt
│   ├── SMQ_J.xpt
│   ├── PAQ_J.xpt
│   └── DR1TOT_J.xpt
├── Prediksi_Status_Kemiskinan_XGBoost_NHANES.ipynb
└── README.md
```

---

## 🚀 Cara Menjalankan

1. Upload semua file `.xpt` ke Google Drive
2. Buka notebook di Google Colab
3. Sesuaikan `folder_path` dengan lokasi file di Drive kamu
4. Jalankan semua cell secara berurutan (`Runtime > Run all`)

---

## 🛠️ Library yang Digunakan

```
pandas, numpy, scikit-learn, xgboost, imbalanced-learn, shap, matplotlib, seaborn
```

---

## 👤 Author

> Dibuat sebagai proyek analisis prediksi kemiskinan berbasis data kesehatan NHANES.
