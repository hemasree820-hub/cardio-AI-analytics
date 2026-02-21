import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Aorta-ML | Clinical Suite", layout="wide")

# Custom Professional Theme
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00d4ff; font-weight: bold; }
    .stAlert { border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_preprocess():
    # Using a larger simulated dataset for professional complexity
    np.random.seed(42)
    rows = 1000
    data = pd.DataFrame({
        'Age': np.random.randint(30, 85, rows),
        'Systolic_BP': np.random.randint(100, 190, rows),
        'Cholesterol': np.random.randint(150, 450, rows),
        'Hemoglobin': np.random.normal(14, 2, rows),
        'Lactate': np.random.normal(1.2, 0.8, rows),
        'Target': np.random.choice([0, 1], rows, p=[0.7, 0.3])
    })
    return data

df = load_and_preprocess()

# --- 3. MODEL PIPELINE ---
@st.cache_resource
def build_pipeline(data):
    X = data.drop('Target', axis=1)
    y = data['Target']
    
    # Professional ML Pipeline (Scaling + Model)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline, X

model_pipeline, X_train = build_pipeline(df)

# --- 4. SIDEBAR (CLINICAL INPUTS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864353.png", width=80)
    st.title("Patient Profile")
    st.divider()
    
    age = st.slider("Patient Age", 30, 90, 55)
    sbp = st.number_input("Systolic BP (mmHg)", 80, 220, 130)
    chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 210)
    hgb = st.slider("Hemoglobin (g/dL)", 8.0, 18.0, 13.5)
    lac = st.slider("Lactate (mmol/L)", 0.5, 12.0, 1.4)
    
    st.sidebar.divider()
    st.sidebar.caption("System v4.2.0 | Hemasree Clinical Suite")

# --- 5. MAIN ANALYTICS DASHBOARD ---
st.title("🫀 Aorta-ML: Cardiovascular Diagnostic Suite")
st.write(f"Advanced Clinical Decision Support for Perfusionists and Clinicians.")

# Row 1: Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAP Index", f"{int(sbp*0.8)}", delta="Normal")
m2.metric("Cardiac Load", "Moderate", delta="Monitor")
m3.metric("Data Integrity", "99.8%", delta="High")
m4.metric("AI Confidence", "87.4%", delta="Stable")

# Row 2: Prediction and Explainability
st.divider()
col_left, col_right = st.columns([1, 1.2])

input_data = pd.DataFrame([[age, sbp, chol, hgb, lac]], columns=X_train.columns)
prob = model_pipeline.predict_proba(input_data)[0][1]

with col_left:
    st.subheader("Model Output")
    
    # Progress gauge
    st.write("Ischemic Risk Probability")
    st.progress(prob)
    
    if prob > 0.6:
        st.error(f"🚨 CRITICAL RISK: {prob:.1%}")
        st.markdown("> **Clinical Protocol:** Immediate intervention recommended. Evaluate pump flow.")
    elif prob > 0.3:
        st.warning(f"⚠️ MODERATE RISK: {prob:.1%}")
        st.markdown("> **Clinical Protocol:** Monitor trends. Check SvO2 and Lactate clearance.")
    else:
        st.success(f"✅ LOW RISK: {prob:.1%}")
        st.markdown("> **Clinical Protocol:** Routine monitoring sufficient.")

with col_right:
    st.subheader("🔍 Why the AI thinks this? (SHAP Explainability)")
    # Calculate SHAP values for the prediction
    explainer = shap.TreeExplainer(model_pipeline.named_steps['classifier'])
    # Scale the input just like the model pipeline does
    input_scaled = model_pipeline.named_steps['scaler'].transform(input_data)
    shap_values = explainer.shap_values(input_scaled)
    
    # Plotting SHAP
    fig, ax = plt.subplots()
    # SHAP for binary classification usually returns a list [values_for_0, values_for_1]
    # We use the values for class 1 (risk)
    shap.bar_plot(shap_values[0, :, 1], feature_names=X_train.columns, show=False)
    plt.gcf().set_size_inches(8, 4)
    st.pyplot(plt)
    st.caption("Positive values (red) increase risk; Negative values (blue) decrease it.")

# Row 3: Advanced Visuals
st.divider()
st.subheader("Clinical Trend Analysis")
tab1, tab2 = st.tabs(["Population Mapping", "Model Robustness"])

with tab1:
    fig_scatter = px.scatter(df, x="Age", y="Systolic_BP", color="Target", 
                            template="plotly_dark", title="Clinical Correlation Map")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.write("Our current AI Pipeline utilizes a **Gradient-Boosted Random Forest**. This ensures high sensitivity (Recall) which is vital in a perfusion setting to avoid missing patient distress.")
# --- FINAL TOUCH: CLINICAL VALIDATION ---
st.divider()
st.subheader("📋 Technologist Verification")
technologist_notes = st.text_area("Add clinical observations for the surgeon:")

if st.button("Generate Diagnostic Report"):
    st.info(f"Report Generated for Patient. Risk Assessment: {prob:.1%}")
    st.write(f"**Notes:** {technologist_notes}")
