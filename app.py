"""
TerraSpectra: Interactive Streamlit Demonstration App.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="TerraSpectra | Crop Disease Forecasting",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 TerraSpectra: Hyperspectral Crop Disease Forecasting")
st.markdown("""
*An intelligent hyperspectral remote sensing & ML diagnostic tool for early crop disease detection.*
**Intern:** Vishal Yadav | **Domain:** Data Science & Machine Learning | **Company:** Infotact Solutions
""")

st.divider()

# Sidebar controls
st.sidebar.header("🕹️ Simulation & Controls")
crop_type = st.sidebar.selectbox("Select Crop Type", ["Wheat (Puccinia / Rust)", "Tomato (Late Blight)", "Paddy Rice (Blast)", "Maize (Leaf Blight)"])
noise_level = st.sidebar.slider("Sensor Noise Level (SNR)", min_value=0.0, max_value=0.1, value=0.02, step=0.01)
infection_stage = st.sidebar.select_slider(
    "Infection Progression Stage",
    options=["Healthy / Asymptomatic", "Early Pre-symptomatic (Day 2-4)", "Moderate Infection (Day 5-7)", "Severe Necrosis (Day 8+)"]
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Simulated Spectral Signature")
    # Synthetic hyperspectral reflectance generation across 400nm - 1000nm (VNIR)
    wavelengths = np.linspace(400, 1000, 200)

    # Base healthy vegetation profile: low blue/red absorption, green peak ~550, sharp red edge ~700-750, high NIR plateau
    healthy = (
        0.05 * np.exp(-((wavelengths - 450) ** 2) / 1000) +
        0.12 * np.exp(-((wavelengths - 550) ** 2) / 1500) +
        0.04 * np.exp(-((wavelengths - 670) ** 2) / 1200) +
        0.48 / (1 + np.exp(-(wavelengths - 720) / 20))
    )

    stage_penalties = {
        "Healthy / Asymptomatic": 0.0,
        "Early Pre-symptomatic (Day 2-4)": 0.08,
        "Moderate Infection (Day 5-7)": 0.22,
        "Severe Necrosis (Day 8+)": 0.38
    }

    penalty = stage_penalties[infection_stage]
    # In disease: chlorophyll breakdown causes red reflectance to rise; cell structure collapse drops NIR reflectance
    simulated = healthy.copy()
    simulated[wavelengths >= 700] -= penalty
    simulated[(wavelengths >= 600) & (wavelengths <= 700)] += penalty * 0.4
    noise = np.random.normal(0, noise_level, size=wavelengths.shape)
    simulated = np.clip(simulated + noise, 0.01, 0.95)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wavelengths, healthy, "g--", label="Baseline (Healthy)", alpha=0.7)
    ax.plot(wavelengths, simulated, "r-", label="Current Sample", lw=2)
    ax.axvspan(690, 740, color="orange", alpha=0.15, label="Red-Edge Region")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"Spectral Profile: {crop_type}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

with col2:
    st.subheader("🧪 Extracted Spectral Vegetation Indices")

    # Pick index bands
    nir = simulated[np.argmin(np.abs(wavelengths - 800))]
    red = simulated[np.argmin(np.abs(wavelengths - 670))]
    red_edge = simulated[np.argmin(np.abs(wavelengths - 715))]
    green = simulated[np.argmin(np.abs(wavelengths - 550))]

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndre = (nir - red_edge) / (nir + red_edge + 1e-6)
    gndvi = (nir - green) / (nir + green + 1e-6)

    m1, m2, m3 = st.columns(3)
    m1.metric("NDVI (Vigor)", f"{ndvi:.3f}", delta=f"{ndvi - 0.75:.2f}")
    m2.metric("NDRE (Early Stress)", f"{ndre:.3f}", delta=f"{ndre - 0.50:.2f}")
    m3.metric("GNDVI (Chlorophyll)", f"{gndvi:.3f}", delta=f"{gndvi - 0.65:.2f}")

    st.subheader("🎯 Model Diagnostic Prediction")
    if penalty == 0.0:
        st.success("✅ **Status: Healthy Crop** — No pathological spectral anomaly detected.")
    elif penalty < 0.15:
        st.warning("⚠️ **Status: Pre-symptomatic Infection Warning** — Subtle Red-Edge shift detected (Early intervention recommended).")
    elif penalty < 0.30:
        st.error("🚨 **Status: Moderate Disease Outbreak** — High fungal/pathogen signature identified.")
    else:
        st.error("🛑 **Status: Severe Necrosis / Foliar Blight** — Immediate quarantine and treatment required.")

st.info("💡 **TerraSpectra Workflow:** Ingest Hyperspectral Cube ➡️ Radiometric Calibration & Noise Filter ➡️ Feature Selection (PCA/Indices) ➡️ ML/DL Classifier ➡️ Risk Mapping.")

