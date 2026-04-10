import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Fetal Health Monitoring System",
    page_icon="🫀",
    layout="wide"
)

model = joblib.load("model.pkl")


st.markdown("""
<style>
.main-title {font-size:42px;font-weight:700;color:#2c3e50;text-align:center;}
.section-title {font-size:26px;font-weight:600;margin-top:30px;}
.card {padding:25px;border-radius:15px;background:#f9f9f9;box-shadow:0px 4px 10px rgba(0,0,0,0.08);margin-top:20px;}
.result-normal {font-size:32px;font-weight:700;color:#1e8449;text-align:center;}
.result-suspect {font-size:32px;font-weight:700;color:#f39c12;text-align:center;}
.result-pathological {font-size:32px;font-weight:700;color:#c0392b;text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Fetal Health Monitoring System</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-title">📂 Upload CTG Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CTG CSV File", type=["csv"])

with col2:
    st.markdown('<div class="section-title">ℹ️ System Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Uses CTG data + XGBoost ML model to predict fetal condition.
    </div>
    """, unsafe_allow_html=True)


normal_ranges = {
    "baseline_value": (110, 160),
    "accelerations": (0.001, 10),
    "fetal_movement": (0.001, 10),
    "uterine_contractions": (0, 5),
    "prolongued_decelerations": (0, 0),
    "abnormal_short_term_variability": (0, 5),
    "mean_value_of_short_term_variability": (3, 10)
}


if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if "fetal_health" in data.columns:
        data = data.drop(columns=["fetal_health"])

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    expected_features = model.get_booster().feature_names
    data = data[expected_features]

    pred = model.predict(data)[0]

    st.markdown('<div class="section-title">🧠 Prediction Result</div>', unsafe_allow_html=True)

    if pred == 0:
        st.markdown('<div class="result-normal">NORMAL</div>', unsafe_allow_html=True)
    elif pred == 1:
        st.markdown('<div class="result-suspect">SUSPECT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-pathological">PATHOLOGICAL</div>', unsafe_allow_html=True)

    
   
    st.markdown('<div class="section-title">📘 AI Explanation</div>', unsafe_allow_html=True)

    sample = data.iloc[0]
    abnormal_data = []

    for feature, (low, high) in normal_ranges.items():
        if feature in sample:
            value = sample[feature]
            if value < low or value > high:
                abnormal_data.append([
                    feature.replace("_", " ").title(),
                    f"{low} - {high}",
                    value
                ])

    if pred == 0:
        explanation = """
        All clinical indicators are within normal medical ranges.<br><br>
        The fetal condition appears stable based on CTG patterns.
        """
        st.markdown(f'<div class="card">{explanation}</div>', unsafe_allow_html=True)

    else:
        if abnormal_data:
            df = pd.DataFrame(abnormal_data, columns=["Factor", "Normal Range", "Sample Value"])

            st.markdown('<div class="card">The following clinical indicators are outside normal ranges:</div>', unsafe_allow_html=True)
            st.table(df)
        else:
            st.markdown('<div class="card">No major abnormal indicators detected.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Normal vs Sample Value Comparison</div>', unsafe_allow_html=True)


    features = []
    normal_means = []
    sample_values = []

    for feature, (low, high) in normal_ranges.items():
        if feature in sample:
            features.append(feature.replace("_", " ").title())
            normal_means.append((low + high) / 2)   # midpoint of normal range
            sample_values.append(sample[feature])

# Create Line Graph
    fig, ax = plt.subplots(figsize=(10, 5))   # width, height

    ax.plot(features, normal_means, marker='o', label="Normal Range (Mid Value)")
    ax.plot(features, sample_values, marker='o', label="Sample Value")

    ax.set_ylabel("Value")
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)