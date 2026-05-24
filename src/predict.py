import pandas as pd
import joblib

# ── Pipeline ni yuklash — bitta fayl! ────────────────────────────────────────
pipeline = joblib.load('pipeline.pkl')

# ── Foydalanuvchidan ma'lumot olish ──────────────────────────────────────────
print("=" * 40)
print("🩺 DIABET TASHXIS TIZIMI")
print("=" * 40)
print("Bemor ma'lumotlarini kiriting:\n")

pregnancies = float(input("Homiladorlik soni         : "))
glucose     = float(input("Qondagi shakar miqdori(Glucose)     : "))
insulin     = float(input("Insulin darajasi          : "))
bmi         = float(input("Tana vazni (BMI)        : "))
dpf         = float(input("Oilaviy dabet tarixi (0.0 - 2.5) : "))
age         = float(input("Yoshi                     : "))

# ── DataFrame ga o'tkazish ────────────────────────────────────────────────────
new_patient = pd.DataFrame({
    "Pregnancies"              : [pregnancies],
    "Glucose"                  : [glucose],
    "Insulin"                  : [insulin],
    "BMI"                      : [bmi],
    "DiabetesPedigreeFunction" : [dpf],
    "Age"                      : [age]
})

# ── Bashorat — pipeline ichida hamma narsa! ───────────────────────────────────
natija     = pipeline.predict(new_patient)
ehtimollik = pipeline.predict_proba(new_patient)

# ── Natija ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
print(" TASHXIS NATIJASI")
print("=" * 40)

if natija[0] == 1:
    print("  Tashxis   : DIABET BOR")
else:
    print(" Tashxis   : SOG'LOM")

print(f" Ehtimollik : Sog'lom {ehtimollik[0][0]*100:.1f}%  |  Diabet {ehtimollik[0][1]*100:.1f}%")
print("=" * 40)
