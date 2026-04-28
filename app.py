"""
╔══════════════════════════════════════════════════════════════╗
║         🌾 Crop Yield Prediction — Streamlit App             ║
╚══════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f5f7f0 0%, #e8f5e9 100%); }

[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #2e7d32;
}
[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #555; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #1b5e20; }

div.stButton > button {
    background: linear-gradient(90deg, #2e7d32, #43a047);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-size: 1.05rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.25s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #1b5e20, #2e7d32);
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(46,125,50,0.35);
}

.result-box {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border: 2px solid #2e7d32;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin-top: 20px;
}
.result-box h2 { color: #1b5e20; margin: 0; }
.result-box .yield-value {
    font-size: 3rem;
    font-weight: 800;
    color: #2e7d32;
    margin: 8px 0;
}

[data-testid="stSidebar"] { background: #1b5e20 !important; }
[data-testid="stSidebar"] * { color: white !important; }

/* Fix selectbox text visibility in sidebar */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
    color: #1b5e20 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stSelectbox span {
    color: #1b5e20 !important;
}

.section-header {
    background: white;
    border-radius: 10px;
    padding: 12px 20px;
    margin: 18px 0 12px 0;
    border-left: 5px solid #43a047;
    font-weight: 700;
    font-size: 1.05rem;
    color: #1b5e20;
    box-shadow: 0 1px 5px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    artefacts = {}

    required = ["scaler", "le_crop", "le_season", "le_state"]
    for name in required:
        path = f"{name}.pkl"
        if not os.path.exists(path):
            st.error(f"⚠️  Missing: {path}. Please run code.ipynb first.")
            st.stop()
        artefacts[name] = joblib.load(path)

    model_files = {
        "Linear Regression": "linear_regression.pkl",
        "Decision Tree":     "decision_tree.pkl",
        "Random Forest":     "random_forest.pkl",
        "XGBoost":           "xgboost.pkl",
    }
    for label, fname in model_files.items():
        if os.path.exists(fname):
            artefacts[label] = joblib.load(fname)

    return artefacts


artefacts = load_artefacts()
scaler    = artefacts["scaler"]
le_crop   = artefacts["le_crop"]
le_season = artefacts["le_season"]
le_state  = artefacts["le_state"]

AVAILABLE_MODELS = {
    k: v for k, v in artefacts.items()
    if k not in ["scaler", "le_crop", "le_season", "le_state"]
}

# ─────────────────────────────────────────────────────────────
# Sidebar  ← everything inside ONE with block
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Crop Yield Predictor")
    st.markdown("---")
    st.markdown("### ⚙️ Select Model")

    model_options = list(AVAILABLE_MODELS.keys())

    if not model_options:
        st.error("No models found! Run code.ipynb first.")
        st.stop()

    default_idx = (
        model_options.index("Random Forest")
        if "Random Forest" in model_options
        else 0
    )

    chosen_model_name = st.selectbox(
        "Algorithm",
        options=model_options,
        index=default_idx
    )
    chosen_model = AVAILABLE_MODELS[chosen_model_name]

    st.markdown("---")
    st.markdown("### 📊 About the Models")
    model_info = {
        "Linear Regression": "Fast baseline. Assumes linear relationship between features and yield.",
        "Decision Tree":     "Tree-based splits. Interpretable but may overfit.",
        "Random Forest":     "Ensemble of 150 trees. Robust & handles non-linearity well.",
        "XGBoost":           "Gradient boosting with 300 estimators. State-of-the-art performance.",
    }
    if chosen_model_name in model_info:
        st.info(model_info[chosen_model_name])

    st.markdown("---")
    st.markdown("### ℹ️ Dataset")
    st.write("19,689 records | 10 features")
    st.write("India Crop Data: 1997–2020")
    st.write("55 crops | 30 states | 6 seasons")


# ─────────────────────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px 0 10px 0;">
    <h1 style="color:#1b5e20; font-size:2.4rem; font-weight:800;">
        🌾 Crop Yield Prediction System
    </h1>
    <p style="color:#555; font-size:1.05rem;">
        Fill in the crop and field parameters below to predict the expected yield.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────────────────────
with st.form(key="prediction_form"):

    st.markdown('<div class="section-header">🌱 Crop & Location Details</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        crop = st.selectbox(
            "🌿 Crop",
            options=sorted(le_crop.classes_.tolist()),
            help="Select the crop type"
        )
    with col2:
        season = st.selectbox(
            "🗓️ Season",
            options=sorted(le_season.classes_.tolist()),
            help="Growing season"
        )
    with col3:
        state = st.selectbox(
            "📍 State",
            options=sorted(le_state.classes_.tolist()),
            help="Indian state"
        )

    st.markdown('<div class="section-header">📋 Field Information</div>',
                unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        crop_year = st.number_input(
            "📅 Crop Year",
            min_value=1997, max_value=2030,
            value=2022, step=1,
            help="Year of cultivation"
        )
    with col5:
        area = st.number_input(
            "🗺️ Area (hectares)",
            min_value=0.1, max_value=60_000_000.0,
            value=1000.0, step=100.0, format="%.2f",
            help="Cultivated area in hectares"
        )
    with col6:
        production = st.number_input(
            "📦 Production (tonnes)",
            min_value=0, max_value=500_000_000,
            value=1500, step=100,
            help="Expected / historical production in tonnes"
        )

    st.markdown('<div class="section-header">🌦️ Environmental & Input Factors</div>',
                unsafe_allow_html=True)

    col7, col8, col9 = st.columns(3)
    with col7:
        annual_rainfall = st.number_input(
            "🌧️ Annual Rainfall (mm)",
            min_value=0.0, max_value=5000.0,
            value=1100.0, step=10.0, format="%.1f",
            help="Total annual rainfall in mm"
        )
    with col8:
        fertilizer = st.number_input(
            "🧪 Fertilizer Used (kg)",
            min_value=0.0, max_value=200_000_000.0,
            value=50000.0, step=1000.0, format="%.2f",
            help="Total fertilizer applied in kg"
        )
    with col9:
        pesticide = st.number_input(
            "🐛 Pesticide Used (kg)",
            min_value=0.0, max_value=20_000_000.0,
            value=2000.0, step=100.0, format="%.2f",
            help="Total pesticide applied in kg"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 Predict Crop Yield", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────────────────────
if submitted:
    try:
        crop_enc   = le_crop.transform([crop])[0]
        season_enc = le_season.transform([season])[0]
        state_enc  = le_state.transform([state])[0]

        input_data = np.array([[
            crop_enc, crop_year, season_enc, state_enc,
            area, production, annual_rainfall, fertilizer, pesticide
        ]])

        input_scaled = scaler.transform(input_data)

        pred_log   = chosen_model.predict(input_scaled)[0]
        pred_yield = np.expm1(pred_log)

        st.markdown("---")
        st.markdown("## 📈 Prediction Results")

        st.markdown(f"""
        <div class="result-box">
            <h2>Predicted Crop Yield</h2>
            <div class="yield-value">{pred_yield:,.4f}</div>
            <p style="color:#388e3c; font-size:1.05rem; margin:4px 0;">
                tonnes per hectare &nbsp;|&nbsp; Model: <strong>{chosen_model_name}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌿 Crop",      crop)
        m2.metric("📍 State",     state)
        m3.metric("🗓️ Season",    season)
        m4.metric("📅 Crop Year", str(int(crop_year)))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("🗺️ Area (ha)",       f"{area:,.1f}")
        m6.metric("🌧️ Rainfall (mm)",   f"{annual_rainfall:,.1f}")
        m7.metric("🧪 Fertilizer (kg)", f"{fertilizer:,.0f}")
        m8.metric("🐛 Pesticide (kg)",  f"{pesticide:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        if pred_yield < 1:
            level, color, advice = "Very Low",      "#c62828", "Consider reviewing soil health, irrigation, and input quantities."
        elif pred_yield < 3:
            level, color, advice = "Low – Average", "#f57f17", "Yield is below national averages. Check fertilizer & water supply."
        elif pred_yield < 10:
            level, color, advice = "Good",          "#2e7d32", "Yield is in a healthy range. Maintain current practices."
        else:
            level, color, advice = "Excellent",     "#1565c0", "Very high yield predicted — typical of crops like Sugarcane or Coconut."

        st.markdown(f"""
        <div style="background:white; border-radius:12px; padding:18px 24px;
                    border-left:5px solid {color}; margin-top:8px;">
            <b style="color:{color}; font-size:1.1rem;">Yield Level: {level}</b><br>
            <span style="color:#444;">{advice}</span>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Make sure you have run all cells in code.ipynb to generate the model files.")

else:
    st.markdown("""
    <div style="background:white; border-radius:14px; padding:30px; text-align:center;
                box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-top:10px; color:#666;">
        <h3 style="color:#2e7d32;">👆 Fill in the form and click <em>Predict Crop Yield</em></h3>
        <p>The model will instantly estimate the expected yield based on your inputs.</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:10px 0;">
    🌾 Crop Yield Prediction System &nbsp;|&nbsp;
    Built with Streamlit · scikit-learn · XGBoost &nbsp;|&nbsp;
    Dataset: India Crop Data (1997–2020)
</div>
""", unsafe_allow_html=True)