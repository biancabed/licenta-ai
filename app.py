import streamlit as st
import joblib
import numpy as np

# Încarcă modelul
model = joblib.load("model_consum.joblib")

# Titlu și descriere
st.title("Simularea producției de energie cu Machine Learning")
st.markdown("""
Aplicație AI care estimează producția de  energie în funcție de condițiile meteorologice și ora din zi.
Modelul a fost antrenat pe date reale și include și efectele radiației solare.
""")

# Formulare
temp = st.number_input("🌡 Temperatura [°C]", value=20.0)
umid = st.number_input("💧 Umiditate [%]", value=50.0)
vant = st.number_input("🌬️ Viteza vântului [m/s]", value=2.)
ora  = st.selectbox("⏰ Ora din zi", options=[
    ("00", 0),
    ("01", 1), ("02", 2), ("03", 3), ("04", 4), ("05", 5),
    ("06", 6), ("07", 7), ("08", 8), ("09", 9), ("10", 10), ("11", 11),
    ("12", 12), ("13", 13), ("14", 14), ("15", 15),
    ("16", 16), ("17", 17), ("18", 18), ("19", 19), ("20", 20),
    ("21", 21), ("22", 22), ("23", 23)
], format_func=lambda x: x[0])

ora = ora[1]


rad_gen = st.number_input("☀️ Radiație generală (W/m²)", value=0.1)
rad_dif = st.number_input("🌥️ Radiație difuză (W/m²)", value=0.1)

# Buton de predicție
if st.button("Calculează producția de energie"):
    input_data = np.array([[temp, umid, vant, ora, rad_gen, rad_dif]])
    rezultat = model.predict(input_data)[0]
    st.success(f"Estimare producție: **{rezultat:.2f} kWh**")

    
