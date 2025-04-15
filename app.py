import streamlit as st
import joblib
import numpy as np

# Încarcă modelul
model = joblib.load("model_consum.joblib")

# Titlu și descriere
st.title("Predicția consumului de energie cu Machine Learning")
st.markdown("""
Aplicație AI care estimează consumul energetic în funcție de condițiile meteorologice și ora din zi.
Modelul a fost antrenat pe date reale și include și efectele radiației solare.
""")

# Formulare
temp = st.number_input("🌡️ Temperatura [°C]", value=20.0)
umid = st.number_input("💧 Umiditate [%]", value=50.0)
vant = st.number_input("🌬️ Viteza vântului [m/s]", value=2.0)
ora = st.number_input("🕒 Ora din zi (0–23)", min_value=0, max_value=23, value=12)
rad_gen = st.number_input("☀️ Radiație generală (W/m²)", value=0.1)
rad_dif = st.number_input("🌥️ Radiație difuză (W/m²)", value=0.1)

# Buton de predicție
if st.button("Calculează consumul"):
    input_data = np.array([[temp, umid, vant, ora, rad_gen, rad_dif]])
    rezultat = model.predict(input_data)[0]

    st.success(f"Estimare consum: **{rezultat:.2f} kWh**")

    # Interpretare consum
    if rezultat < 10000:
        st.info("✅ Consum scăzut – foarte eficient energetic.")
    elif rezultat < 25000:
        st.warning("⚠️ Consum moderat – se încadrează în valorile normale.")
    else:
        st.error("🔥 Consum ridicat – potențial excesiv de energie.")


