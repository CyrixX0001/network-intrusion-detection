import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix
)

# Page configuration
st.set_page_config(
    page_title="CyberShield AI | Network IDS Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Dark Modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Gradient header banner */
    .ids-header {
        background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
    }

    .badge-cyber {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .status-pulse {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }

    /* Metric Cards */
    .metric-container {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Threat Status Badges */
    .threat-banner-normal {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(16, 185, 129, 0.2) 100%);
        border: 1px solid #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .threat-banner-attack {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.4) 0%, rgba(239, 68, 68, 0.2) 100%);
        border: 1px solid #ef4444;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.3);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Feature metadata
FEATURE_NAMES = [
    'packet_size',
    'connection_duration',
    'failed_logins',
    'bytes_transferred',
    'src_to_dst_ratio',
    'urgent_packets',
    'wrong_fragments',
    'num_compromised',
    'root_shell_attempts',
    'hot_indicators'
]

FEATURE_LABELS = {
    'packet_size': 'Packet Size (Bytes)',
    'connection_duration': 'Connection Duration (s)',
    'failed_logins': 'Failed Login Attempts',
    'bytes_transferred': 'Bytes Transferred (KB)',
    'src_to_dst_ratio': 'Src/Dst Packet Ratio',
    'urgent_packets': 'Urgent Flag Packets',
    'wrong_fragments': 'Wrong Fragment Packets',
    'num_compromised': 'Compromised Conditions',
    'root_shell_attempts': 'Root Shell Executions',
    'hot_indicators': 'Suspicious Indicator Hits'
}

# -------------------------------------------------------------
# Data Generation & Model Training with Caching
# -------------------------------------------------------------
@st.cache_data
def generate_ids_dataset(n_samples=1000, random_seed=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.60, 0.40], # 60% normal, 40% malicious
        flip_y=0.03,
        random_state=random_seed
    )
    
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    
    # Scale realistic network bounds
    df['packet_size'] = np.abs(df['packet_size'] * 350 + 512).round(1)
    df['connection_duration'] = np.abs(df['connection_duration'] * 25 + 5).round(2)
    df['failed_logins'] = np.maximum(0, (df['failed_logins'] * 3 + 1).round(0)).astype(int)
    df['bytes_transferred'] = np.abs(df['bytes_transferred'] * 5000 + 10240).round(1)
    df['src_to_dst_ratio'] = np.maximum(0.01, (df['src_to_dst_ratio'] + 2.0).round(2))
    df['urgent_packets'] = np.maximum(0, (df['urgent_packets'] * 2).round(0)).astype(int)
    df['wrong_fragments'] = np.maximum(0, (df['wrong_fragments'] * 2).round(0)).astype(int)
    df['num_compromised'] = np.maximum(0, (df['num_compromised'] * 4).round(0)).astype(int)
    df['root_shell_attempts'] = np.maximum(0, (df['root_shell_attempts'] * 1.5).round(0)).astype(int)
    df['hot_indicators'] = np.maximum(0, (df['hot_indicators'] * 3).round(0)).astype(int)
    
    df['label'] = y
    return df

@st.cache_resource
def train_model(df, n_estimators, max_depth, random_seed):
    X = df[FEATURE_NAMES]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': auc(*roc_curve(y_test, y_prob)[:2][::-1])
    }
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importances
    fi_df = pd.DataFrame({
        'Feature': [FEATURE_LABELS[f] for f in FEATURE_NAMES],
        'Feature_Key': FEATURE_NAMES,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics, (fpr, tpr), cm, fi_df

# -------------------------------------------------------------
# SIDEBAR: Controls & Hyperparameters
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <div style="font-size: 2rem;">🛡️</div>
            <div>
                <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #38bdf8;">IDS Control Hub</h2>
                <div style="font-size: 0.75rem; color: #94a3b8;"><span class="status-pulse"></span> SYSTEM ACTIVE</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Model Hyperparameters")
    
    n_estimators = st.slider(
        "Number of Estimators (Trees)",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
        help="The number of decision trees in the random forest ensemble."
    )
    
    max_depth = st.slider(
        "Maximum Tree Depth",
        min_value=2,
        max_value=25,
        value=8,
        step=1,
        help="The maximum depth allowed for each decision tree."
    )
    
    random_seed = st.number_input(
        "Random Seed Generator",
        min_value=1,
        max_value=9999,
        value=42,
        step=1,
        help="Ensures reproducibility across synthetic generation and forest splitting."
    )
    
    st.markdown("---")
    
    st.markdown("### 🗃️ Synthetic Dataset Stats")
    st.markdown("""
    - **Total Packets:** `1,000`
    - **Features Recorded:** `10`
    - **Train / Test Ratio:** `80% / 20%`
    - **Class Balance:** `60% Normal / 40% Attack`
    """)
    
    retrain_clicked = st.button("🔄 Retrain Model", use_container_width=True, type="primary")
    if retrain_clicked:
        train_model.clear()
        st.toast("Model retrained successfully!", icon="✅")

# -------------------------------------------------------------
# DATASET & MODEL INVOCATION
# -------------------------------------------------------------
df = generate_ids_dataset(1000, random_seed)
model, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics, roc_data, cm, fi_df = train_model(
    df, n_estimators, max_depth, random_seed
)

# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
st.markdown("""
    <div class="ids-header">
        <div class="badge-cyber">Network Security & Machine Learning</div>
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
            AI-Powered Network Intrusion Detection System
        </h1>
        <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 0.95rem; line-height: 1.5;">
            Real-time packet inspection and anomaly classifier trained with Scikit-learn Random Forests.
            Simulates deep telemetry analysis to separate benign enterprise traffic from malicious intrusions.
        </p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# METRIC CARDS
# -------------------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{metrics['accuracy'] * 100:.1f}%</div>
            <div style="font-size: 0.75rem; color: #10b981; margin-top: 4px;">▲ Test set performance</div>
        </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{metrics['precision'] * 100:.1f}%</div>
            <div style="font-size: 0.75rem; color: #38bdf8; margin-top: 4px;">Targeting intrusion rate</div>
        </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Recall (Sensitivity)</div>
            <div class="metric-value">{metrics['recall'] * 100:.1f}%</div>
            <div style="font-size: 0.75rem; color: #a855f7; margin-top: 4px;">Attacks successfully caught</div>
        </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">{metrics['f1'] * 100:.1f}%</div>
            <div style="font-size: 0.75rem; color: #f59e0b; margin-top: 4px;">Harmonic mean quality</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# VISUALIZATIONS SECTION
# -------------------------------------------------------------
v_col1, v_col2 = st.columns(2)

with v_col1:
    st.markdown("#### 🔍 Interactive Confusion Matrix")
    
    # Custom styled confusion matrix
    cm_text = [
        [f"True Normal: {cm[0, 0]}", f"False Attack: {cm[0, 1]}"],
        [f"False Normal: {cm[1, 0]}", f"True Attack: {cm[1, 1]}"]
    ]
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Malicious'],
        y=['Actual Normal', 'Actual Malicious'],
        text=cm_text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white", "family": "JetBrains Mono"},
        colorscale=[
            [0.0, "#0f172a"],
            [0.5, "#1e3a8a"],
            [1.0, "#0284c7"]
        ],
        colorbar=dict(title="Packets", tickfont=dict(color="#94a3b8")),
        hoverongaps=False
    ))
    
    fig_cm.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        font=dict(color="#cbd5e1")
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with v_col2:
    st.markdown(f"#### 📈 ROC-AUC Curve (AUC: {metrics['roc_auc']:.3f})")
    fpr, tpr = roc_data
    
    fig_roc = go.Figure()
    
    # Baseline
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#64748b', dash='dash', width=2),
        name='Random Baseline'
    ))
    
    # Classifier ROC
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color='#38bdf8', width=3),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.1)',
        name=f'Random Forest (AUC = {metrics["roc_auc"]:.3f})'
    ))
    
    fig_roc.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        font=dict(color="#cbd5e1"),
        legend=dict(x=0.45, y=0.1, bgcolor="rgba(15, 23, 42, 0.7)")
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# -------------------------------------------------------------
# FEATURE IMPORTANCE CHART
# -------------------------------------------------------------
st.markdown("#### 📊 Model Feature Importance (Gini Impurity)")
fig_fi = px.bar(
    fi_df,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale=[[0, '#0284c7'], [0.5, '#38bdf8'], [1, '#f43f5e']],
)

fig_fi.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(15, 23, 42, 0.4)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    yaxis=dict(autorange="reversed"),
    margin=dict(l=40, r=40, t=20, b=20),
    height=280,
    font=dict(color="#cbd5e1"),
    coloraxis_showscale=False
)
st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# LIVE INFERENCE PANEL
# -------------------------------------------------------------
st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: #38bdf8; display: flex; align-items: center; gap: 8px;">
            ⚡ Live Packet Telemetry & Threat Inference Simulator
        </h3>
        <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
            Test a custom network packet in real time. The controls below dynamically reflect the top 4 most decisive telemetry features identified by the model.
        </p>
    </div>
""", unsafe_allow_html=True)

# Determine Top 4 features dynamically
top_4_keys = fi_df['Feature_Key'].head(4).tolist()
top_4_labels = fi_df['Feature'].head(4).tolist()

inf_col_left, inf_col_right = st.columns([1.1, 0.9])

with inf_col_left:
    st.markdown("##### 🎛️ Adjust Top Feature Parameters")
    
    # Preset buttons for testing
    preset_cols = st.columns(3)
    preset_normal = preset_cols[0].button("🟢 Benign Normal", use_container_width=True)
    preset_portscan = preset_cols[1].button("🔴 DoS / Flood", use_container_width=True)
    preset_bruteforce = preset_cols[2].button("⚠️ Brute-Force / Exploit", use_container_width=True)

    input_values = {}
    
    # Compute median baseline for all 10 features
    for col in FEATURE_NAMES:
        input_values[col] = float(df[col].median())
        
    # Check presets
    if preset_normal:
        st.session_state['preset'] = 'normal'
    elif preset_portscan:
        st.session_state['preset'] = 'portscan'
    elif preset_bruteforce:
        st.session_state['preset'] = 'bruteforce'

    current_preset = st.session_state.get('preset', None)

    # Render inputs for top 4 features
    for i, key in enumerate(top_4_keys):
        col_min = float(df[key].min())
        col_max = float(df[key].max())
        col_median = float(df[key].median())
        
        # Decide default based on preset
        if current_preset == 'normal':
            default_val = float(df[df['label'] == 0][key].quantile(0.35))
        elif current_preset in ['portscan', 'bruteforce']:
            default_val = float(df[df['label'] == 1][key].quantile(0.85))
        else:
            default_val = col_median
            
        default_val = min(max(default_val, col_min), col_max)
        
        # Format step
        is_int_feat = key in ['failed_logins', 'urgent_packets', 'wrong_fragments', 'num_compromised', 'root_shell_attempts', 'hot_indicators']
        
        if is_int_feat:
            val = st.slider(
                f"{top_4_labels[i]} (`{key}`)",
                min_value=int(col_min),
                max_value=int(col_max),
                value=int(round(default_val)),
                step=1,
                key=f"slider_{key}"
            )
        else:
            val = st.slider(
                f"{top_4_labels[i]} (`{key}`)",
                min_value=round(col_min, 2),
                max_value=round(col_max, 2),
                value=round(default_val, 2),
                step=round((col_max - col_min) / 50, 2),
                key=f"slider_{key}"
            )
        input_values[key] = val

with inf_col_right:
    st.markdown("##### 🎯 Real-Time Threat Classification")
    
    # Construct input dataframe
    packet_df = pd.DataFrame([input_values])[FEATURE_NAMES]
    
    # Run prediction
    pred_class = model.predict(packet_df)[0]
    pred_probabilities = model.predict_proba(packet_df)[0]
    normal_prob = pred_probabilities[0]
    malicious_prob = pred_probabilities[1]
    
    if pred_class == 1:
        st.markdown(f"""
            <div class="threat-banner-attack">
                <div style="font-size: 2.5rem; margin-bottom: 5px;">🚨</div>
                <h2 style="color: #ef4444; margin: 0; font-size: 1.6rem; font-weight: 800;">THREAT DETECTED</h2>
                <div style="color: #fca5a5; font-size: 0.95rem; margin-top: 4px; font-weight: 600;">
                    MALICIOUS NETWORK INTRUSION CLASSIFIED
                </div>
                <div style="margin-top: 12px; font-size: 1.1rem; color: #fecaca; font-family: 'JetBrains Mono', monospace;">
                    Confidence Score: <b>{malicious_prob * 100:.1f}%</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="threat-banner-normal">
                <div style="font-size: 2.5rem; margin-bottom: 5px;">🛡️</div>
                <h2 style="color: #10b981; margin: 0; font-size: 1.6rem; font-weight: 800;">BENIGN TRAFFIC</h2>
                <div style="color: #6ee7b7; font-size: 0.95rem; margin-top: 4px; font-weight: 600;">
                    NORMAL NETWORK PACKET CONFIRMED
                </div>
                <div style="margin-top: 12px; font-size: 1.1rem; color: #d1fae5; font-family: 'JetBrains Mono', monospace;">
                    Confidence Score: <b>{normal_prob * 100:.1f}%</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Probability Breakdown Gauge Bar
    prob_fig = go.Figure()
    
    prob_fig.add_trace(go.Bar(
        y=['Probability'],
        x=[normal_prob * 100],
        name='Normal',
        orientation='h',
        marker=dict(color='#10b981'),
        text=f"{normal_prob * 100:.1f}%",
        textposition="inside",
        insidetextfont=dict(color="white", size=13, family="JetBrains Mono")
    ))
    
    prob_fig.add_trace(go.Bar(
        y=['Probability'],
        x=[malicious_prob * 100],
        name='Malicious',
        orientation='h',
        marker=dict(color='#ef4444'),
        text=f"{malicious_prob * 100:.1f}%",
        textposition="inside",
        insidetextfont=dict(color="white", size=13, family="JetBrains Mono")
    ))
    
    prob_fig.update_layout(
        barmode='stack',
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], title="Probability (%)", showgrid=False),
        yaxis=dict(showticklabels=False),
        height=100,
        margin=dict(l=10, r=10, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(prob_fig, use_container_width=True)

# Expandable telemetry inspector
with st.expander("🔬 Inspect Raw Simulated Packet Telemetry Record"):
    inspect_col1, inspect_col2 = st.columns(2)
    with inspect_col1:
        st.write("Feature Vector Values:")
        st.dataframe(packet_df.T.rename(columns={0: 'Value'}), use_container_width=True)
    with inspect_col2:
        st.write("Model Decision Breakdown:")
        st.json({
            "predicted_class": int(pred_class),
            "threat_label": "Malicious Attack" if pred_class == 1 else "Normal Traffic",
            "normal_probability": round(float(normal_prob), 4),
            "malicious_probability": round(float(malicious_prob), 4),
            "top_features_evaluated": top_4_keys
        })
