import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ── 1. Ma'lumotlarni yuklash ──────────────────────────────────────────────────
URL = 'https://raw.githubusercontent.com/anvarnarz/praktikum_datasets/main/diabetes.csv'
df = pd.read_csv(URL)
print(f"Dataset yuklandi: {df.shape}")

# ── 2. Shovqinli ustunlarni tashlaymiz ───────────────────────────────────────
df = df.drop(['BloodPressure', 'SkinThickness'], axis=1)

# ── 3. X va y ajratamiz ──────────────────────────────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# ── 4. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=77, stratify=y
)

# ── 5. 0 larni NaN ga o'tkazamiz ─────────────────────────────────────────────
cols = ['Glucose', 'Insulin', 'BMI']
X_train[cols] = X_train[cols].replace(0, np.nan)
X_test[cols]  = X_test[cols].replace(0, np.nan)

# ── 6. SMOTE — faqat train ga (NaN bilan ishlamaydi, shuning uchun avval impute) ──
imputer_smote = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer_smote.fit_transform(X_train), columns=X_train.columns)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_imp, y_train)
print(f"SMOTE dan keyin train hajmi: {X_train_res.shape}")

# ── 7. Eng yaxshi K ni topish ────────────────────────────────────────────────
scores = []
for k in range(1, 25):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  KNeighborsClassifier(n_neighbors=k))
    ])
    score = cross_val_score(pipe, X_train_res, y_train_res, cv=5, scoring='f1')
    scores.append(score.mean())

best_k = np.argmax(scores) + 1

# ── 8. Final Pipeline — Imputer + Scaler + Model ─────────────────────────────
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   KNeighborsClassifier(n_neighbors=best_k, weights='distance'))
])

pipeline.fit(X_train_res, y_train_res)

# ── 9. Pipeline ni saqlash — bitta fayl! ─────────────────────────────────────
joblib.dump(pipeline, 'pipeline.pkl')
print(" Pipeline saqlandi: models/pipeline.pkl")
