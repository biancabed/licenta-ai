import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# 1. Citește datele
df = pd.read_csv("date_reale.csv")
df = df.dropna()

# 2. Conversie inteligentă a vitezei vântului (doar dacă pare în km/h)
df.loc[df["WindSpeed"] > 30, "WindSpeed"] = df["WindSpeed"] / 3.6

# 3. Extragem ora
df["Hour"] = pd.to_datetime(df["Datetime"]).dt.hour

# 4. Definim variabilele de intrare
X = df[[
    "Temperature", "Humidity", "WindSpeed", "Hour",
    "GeneralDiffuseFlows", "DiffuseFlows"
]]
y = df["PowerConsumption_Zone1"]

# 5. Împărțim în antrenare și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Antrenăm modelul
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluăm
score = model.score(X_test, y_test)
print(f"Acuratețea modelului (R²): {score:.2f}")

# 8. Salvăm modelul
joblib.dump(model, "model_consum.joblib")
print("✅ Modelul a fost salvat cu succes.")

