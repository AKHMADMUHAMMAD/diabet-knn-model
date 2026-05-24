# Diabet Tashxisi — K-NN Model

Pima Indians Diabetes dataseti asosida diabet kasalligini tashxis qo'yuvchi ML loyihasi.

---

## Loyiha haqida

Bu loyihada K-Nearest Neighbors (K-NN) algoritmi yordamida diabet tashxisi qo'yuvchi model qurilgan. Pipeline quyidagi bosqichlarni o'z ichiga oladi: ma'lumotlarni tozalash, SMOTE bilan klasslar balansini tenglashtirish, feature scaling va cross-validation yordamida eng yaxshi K ni topish.

---

## Model natijalari

Model 20% test to'plami asosida baholandi:

| Metrika | 0-klass (Sog'lom) | 1-klass (Diabet) |
|---------|-------------------|------------------|
| Precision | 0.85 | 0.60 |
| Recall | 0.73 | 0.76 |
| F1-score | 0.78 | 0.67 |
| Accuracy | 0.74 | — |

Confusion Matrix:
```
              Bashorat: 0    Bashorat: 1
Haqiqiy: 0        73              27
Haqiqiy: 1        13              41
```

> Diabetli bemorlarning 76% to'g'ri aniqlandi (Recall)

---

## Loyiha tuzilmasi

```
diabetes-knn/
├── data/
│   └── diabetes.csv        
├── models/
│   └── pipeline.pkl        
├── notebooks/
│   └── diabetes_knn.ipynb
|   ⌊__ diabet_kasalligini_aniqlaydigan_k_nn_model.ipynb
├── src/
│   ├── train.py              
│   └── predict.py             
├── requirements.txt
└── README.md
```

---

## Pipeline bosqichlari

```
Ma'lumot yuklash → EDA → Shovqinli ustunlarni tashlash → Train/Test Split
→ 0 larni NaN ga o'tkazish → Imputer (Median) → SMOTE
→ StandardScaler → Cross-Val K topish → K-NN Model → Baholash
```

---

## Ishlatiladigan ustunlar

| Ustun | Tavsif |
|-------|--------|
| `Pregnancies` | Homiladorlik soni |
| `Glucose` | Qon shakari darajasi |
| `Insulin` | Insulin darajasi |
| `BMI` | Tana massasi indeksi |
| `DiabetesPedigreeFunction` | Oilaviy diabet tarixi |
| `Age` | Yosh |

> `BloodPressure` va `SkinThickness` — target bilan korrelyatsiyasi past bo'lgani uchun tashlab yuborildi.

---

##  Ishga tushirish

### 1. Kutubxonalarni o'rnatish


pip install -r requirements.txt


### 2. Modelni o'qitish


python src/train.py


### 3. Tashxis qo'yish


python src/predict.py


---

## Bashorat namunasi

predict.py` ishga tushirilganda bemor ma'lumotlari so'raladi:


========================================
  DIABET TASHXIS TIZIMI
========================================
Bemor ma'lumotlarini kiriting:

Homiladorlik soni               : 2
Qondagi shakar miqdori(Glucose) : 138
Insulin darajasi                : 0
Tana massasi (BMI)              : 33.6
Oilaviy tarix (0.0 - 2.5)       : 0.627
Yoshi                           : 50

========================================
 TASHXIS NATIJASI
========================================
 Tashxis   : DIABET BOR
 Ehtimollik : Sog'lom 34.2%  |  Diabet 65.8%
========================================



## Texnologiyalar

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue)



## Muallif

**AKHMADMUHAMMAD** — Junior ML Developer
Data Science & Machine Learning
