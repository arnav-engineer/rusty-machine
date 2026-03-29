import streamlit as st
import numpy as np
import cupy as cp
import pandas as pd
import time
import altair as alt
import os

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.metrics import accuracy_score, r2_score

from codecarbon import OfflineEmissionsTracker

# --- Import Rusty Machine ---
try:
    from rustymachine_api.models import LogisticRegression as RustyLogisticRegression
    from rustymachine_api.models import LinearRegression as RustyLinearRegression
except ImportError:
    st.error("FATAL ERROR: 'rusty_machine' library not found.")
    st.stop()


# --- Page Config ---
st.set_page_config(
    page_title="Rusty Machine Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- High-End Luxury CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 5.5rem;
        color: #000000;
        margin-bottom: 0;
        text-align: center;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .hero-subtitle {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #555555;
        margin-top: 10px;
        margin-bottom: 60px;
        line-height: 1.6;
        padding: 0 10%;
    }

    .luxury-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 45px 30px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.03);
        margin-bottom: 30px;
        text-align: center;
    }

    .luxury-stat {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        color: #d4af37;
        margin-bottom: 5px;
        line-height: 1.1;
    }

    .luxury-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #a0a0a0;
        font-weight: 600;
        margin-bottom: 25px;
    }

    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #d4af37, transparent);
        margin: 50px 0;
        opacity: 0.4;
    }
</style>
""", unsafe_allow_html=True)


# --- Helpers ---
def track_and_train(model, X, y, model_name):
    tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error", output_file=f"emissions_{model_name}.csv")
    tracker.start()
    
    start_time = time.time()
    model.fit(X, y)
    duration = time.time() - start_time
    
    emissions = tracker.stop() # Returns kg of CO2 equivalent
    return duration, model, emissions

def track_predict(model, X_test):
    start_time = time.time()
    preds = model.predict(X_test)
    duration = time.time() - start_time
    return preds, duration


# --- Header ---
st.markdown('<div class="hero-title">Rusty Machine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">High-Fidelity Acceleration of Machine Learning Primitives via Zero-Copy Tensor Core Dispatch on Discrete VRAM Architectures: A Comparative Benchmark Against CPython Ecosystems</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### Configuration Suites")
    
    model_type = st.selectbox(
        "Algorithm Engine",
        ("Logistic Regression", "Linear Regression"),
    )

    if "Logistic" in model_type:
        default_samples, default_features = 500000, 100
        penalty = st.radio("Sparsity Penalty", ('l2', 'l1'))
    else:
        default_samples, default_features = 1000000, 100
        penalty = 'l2'

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Data Topography")
    # Increased max samples threshold
    n_samples = st.slider("Observations", 10000, 5000000, default_samples, 50000)
    n_features = st.slider("Dimensionality", 10, 500, default_features, 10)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Hyperparameters")
    alpha = st.slider("Alpha Regularization", 0.0, 1.0, 0.1, 0.01)

    if "Logistic" in model_type:
        epochs = st.slider("Convergence Epochs", 50, 500, 100, 10)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.05, 0.001, format="%.3f")
        batch_size = st.select_slider("Gradient Block Size", [256, 512, 1024, 2048, 4096, 8192], 1024)
    else:
        epochs, learning_rate, batch_size = 1, 0.01, 1

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("Initialize Engine", use_container_width=True, type="primary")

if not run_button:
    st.markdown("""
        <div style="text-align: center; color: #a0a0a0; font-family: 'Inter'; font-weight: 300; font-size: 1.1rem; margin-top: 100px;">
            Awaiting Initialization. Please configure your suite in the sidebar.
        </div>
    """, unsafe_allow_html=True)

if run_button:
    # --- Synthesizing Dataset ---
    with st.spinner("Synthesizing topographical mappings..."):
        if "Logistic" in model_type:
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), random_state=42)
        else:
            X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), noise=25, random_state=42)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)
        
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Training Sequences ---
    col1, col2 = st.columns(2)

    # Rusty Machine
    with col1:
        st.markdown("<h2 style='text-align: center; font-family: \"Playfair Display\"; color: #d4af37;'>rusty-machine</h2>", unsafe_allow_html=True)
        with st.spinner("Igniting CUDA Tensor Cores..."):
            if "Logistic" in model_type:
                rusty_model = RustyLogisticRegression(epochs=epochs, lr=learning_rate, batch_size=batch_size, penalty=penalty, alpha=alpha, random_state=42)
            else:
                rusty_model = RustyLinearRegression(alpha=alpha)
            
            rm_duration, rm_model, rm_carbon = track_and_train(rusty_model, X_train_s, y_train, "rusty")
            rm_preds, rm_pred_duration = track_predict(rm_model, X_test_s)

            if "Logistic" in model_type:
                 rm_score = accuracy_score(y_test, rm_preds)
            else:
                 rm_score = r2_score(y_test, rm_preds)

        st.markdown(f"""
            <div class="luxury-card">
                <div class="luxury-stat" style="color: #000000;">{rm_duration:.3f}s</div>
                <div class="luxury-label">Training Latency (Seconds)</div>
                <div class="luxury-stat" style="color: #555555; font-size: 2.5rem;">{rm_pred_duration:.4f}s</div>
                <div class="luxury-label">Prediction Latency ({len(X_test):,} inferences)</div>
                <div class="luxury-stat" style="color: #1c7c54; font-size: 2.5rem;">{rm_carbon * 1000:.4f}g</div>
                <div class="luxury-label">Carbon Emissions (CO₂e)</div>
            </div>
        """, unsafe_allow_html=True)

    # Scikit-Learn
    with col2:
        st.markdown("<h2 style='text-align: center; font-family: \"Playfair Display\"; color: #555555;'>scikit-learn</h2>", unsafe_allow_html=True)
        with st.spinner("Processing deeply..."):
            if "Logistic" in model_type:
                C_param = 1.0 / alpha if alpha > 0 else float('inf')
                sk_model = SklearnLogisticRegression(penalty=penalty, C=C_param, solver='saga', max_iter=epochs, tol=1e-3, random_state=42)
            else:
                sk_model = SklearnRidge(alpha=alpha, solver='auto', random_state=42)
            
            sk_duration, sk_model, sk_carbon = track_and_train(sk_model, X_train_s, y_train.ravel(), "sklearn")
            sk_preds, sk_pred_duration = track_predict(sk_model, X_test_s)

            if "Logistic" in model_type:
                 sk_score = accuracy_score(y_test, sk_preds)
            else:
                 sk_score = r2_score(y_test, sk_preds)

        st.markdown(f"""
            <div class="luxury-card">
                <div class="luxury-stat" style="color: #000000;">{sk_duration:.3f}s</div>
                <div class="luxury-label">Training Latency (Seconds)</div>
                <div class="luxury-stat" style="color: #555555; font-size: 2.5rem;">{sk_pred_duration:.4f}s</div>
                <div class="luxury-label">Prediction Latency ({len(X_test):,} inferences)</div>
                <div class="luxury-stat" style="color: #1c7c54; font-size: 2.5rem;">{sk_carbon * 1000:.4f}g</div>
                <div class="luxury-label">Carbon Emissions (CO₂e)</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size: 2.5rem; margin-bottom: 40px;'>IEEE Methodological Yields</h3>", unsafe_allow_html=True)

    # --- Constructing Dataframes for IEEE Style Minimalist Charts ---
    chart_data = pd.DataFrame([
        {"Metric": "Training Time (s)", "rusty-machine": rm_duration, "scikit-learn": sk_duration},
        {"Metric": "Prediction Time (s)", "rusty-machine": rm_pred_duration, "scikit-learn": sk_pred_duration},
        {"Metric": "Carbon Emissions (g)", "rusty-machine": rm_carbon * 1000, "scikit-learn": sk_carbon * 1000}
    ])
    melted_chart_data = pd.melt(chart_data, id_vars=['Metric'], value_vars=['rusty-machine', 'scikit-learn'], var_name='Framework', value_name='Value')

    # Chart 1: Training Time
    c1, c2, c3 = st.columns(3)
    
    def create_luxury_chart(metric_name, title):
        df = melted_chart_data[melted_chart_data['Metric'] == metric_name]
        return alt.Chart(df).mark_bar(size=40).encode(
            x=alt.X('Framework:N', title=None, axis=alt.Axis(labelAngle=0, labelFont='Inter', labelFontSize=12)),
            y=alt.Y('Value:Q', title=None, axis=alt.Axis(grid=False, labelFont='Inter', labelFontSize=12)),
            color=alt.Color('Framework:N', scale=alt.Scale(domain=['rusty-machine', 'scikit-learn'], range=['#d4af37', '#1a1a1a']), legend=None),
            tooltip=['Framework', 'Value']
        ).properties(
            title=title,
            height=300
        ).configure_view(
            strokeWidth=0
        ).configure_title(
            font='Playfair Display',
            fontSize=18,
            color='#1a1a1a',
            anchor='middle',
            offset=20
        )

    with c1:
        st.altair_chart(create_luxury_chart("Training Time (s)", "Training Convergence Latency"), use_container_width=True)
    with c2:
        st.altair_chart(create_luxury_chart("Prediction Time (s)", "Inference Latency"), use_container_width=True)
    with c3:
         st.altair_chart(create_luxury_chart("Carbon Emissions (g)", "Calculated Carbon Footprint (CO₂e)"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_x, col_y, col_z = st.columns(3)
    speedup = sk_duration / rm_duration if rm_duration > 0 else 0
    pred_speedup = sk_pred_duration / rm_pred_duration if rm_pred_duration > 0 else 0
    carbon_diff = sk_carbon / rm_carbon if rm_carbon > 0 else 0

    with col_x:
        st.markdown(f"""
        <div class="luxury-card" style="padding: 25px;">
            <div class="luxury-stat" style="font-size: 2.5rem;">{speedup:.1f}x</div>
            <div class="luxury-label" style="margin-bottom: 0;">Training Velocity Multiplier</div>
        </div>
        """, unsafe_allow_html=True)

    with col_y:
        st.markdown(f"""
        <div class="luxury-card" style="padding: 25px;">
            <div class="luxury-stat" style="font-size: 2.5rem;">{pred_speedup:.1f}x</div>
            <div class="luxury-label" style="margin-bottom: 0;">Inference Velocity Multiplier</div>
        </div>
        """, unsafe_allow_html=True)

    with col_z:
        st.markdown(f"""
        <div class="luxury-card" style="padding: 25px;">
            <div class="luxury-stat" style="font-size: 2.5rem; color: #1c7c54;">{carbon_diff:.1f}x</div>
            <div class="luxury-label" style="margin-bottom: 0;">Green Efficiency (Lesser CO₂)</div>
        </div>
        """, unsafe_allow_html=True)