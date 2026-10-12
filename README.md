# Network Intrusion Detection System (IDS): An Explainable Machine Learning Case Study

An interactive, self-contained Network Intrusion Detection System (NIDS) simulator built with **Python**, **Scikit-learn**, **Streamlit**, **Pandas**, and **Plotly**. This application explores how ensemble machine learning algorithms can be applied to classify high-throughput network telemetry into benign enterprise communication versus malicious cyber intrusions.

---

## 1. Problem Statement

Modern enterprise networks generate massive volumes of packet-level telemetry every second. Traditional signature-based intrusion detection systems (such as legacy Snort or Suricata rulesets) rely heavily on known exploit signatures and fixed rule patterns. While effective against known threats, signature-based approaches exhibit critical vulnerabilities:

- **Zero-Day and Polymorphic Attacks:** Novel exploits and mutated malware evade static signature hashes.
- **High Operational Overhead:** Maintaining signature databases requires continuous human intervention and domain expertise.
- **Volume and Velocity:** Scaled volumetric attacks (such as SYN floods and distributed denial-of-service campaigns) require adaptive, statistical anomaly detection rather than exhaustive pattern matching.

A **Machine Learning-based Intrusion Detection System (ML-NIDS)** addresses these challenges by learning multivariate behavioral distributions from network connection metadata (e.g., packet payload size, connection lifetime, session flag anomalies, and login attempt velocities). The objective is to construct a binary classification model that accurately distinguishes between **Normal Traffic** ($y = 0$) and **Malicious Intrusions** ($y = 1$) while minimizing false alarms and providing explainable diagnostics to Security Operations Center (SOC) analysts.

---

## 2. Algorithm Justification: Why Random Forest?

For this case study, an ensemble **Random Forest Classifier** was selected over alternative paradigms (such as single decision trees, support vector machines, or deep artificial neural networks) for several key technical and practical reasons:

| Criterion | Random Forest Advantage | Alternative Model Limitations |
| :--- | :--- | :--- |
| **Overfitting Resistance** | Averages multiple de-correlated decision trees trained via bootstrap aggregation (bagging), drastically reducing variance. | Single decision trees overfit complex training noise. |
| **Non-Linear Interactions** | Captures complex, non-linear relationships between independent telemetry attributes (e.g., packet volume combined with login retry bursts). | Linear models (e.g., Logistic Regression) fail to capture interaction terms without manual feature engineering. |
| **Scale Invariance & Preprocessing** | Tree-based partitioning is invariant to monotonic feature scaling, eliminating the strict necessity for feature normalization or standardization. | SVMs and Neural Networks are highly sensitive to feature scaling and gradient explosions. |
| **Model Explainability & Auditing** | Provides native **Mean Decrease in Impurity (Gini Importance)** rankings, allowing security engineers to isolate the exact telemetry signals driving threat determinations. | Deep neural networks operate as opaque "black boxes," hindering regulatory compliance and root-cause analysis. |
| **Inference Latency** | Optimized traversal through ensemble binary decision paths yields sub-millisecond per-packet inference latency suitable for near-line detection. | Complex deep learning architectures introduce computational overhead requiring dedicated GPU hardware. |

---

## 3. Telemetry Features Overview

The synthetic telemetry pipeline simulates 10 real-world connection metrics:

1. `packet_size`: Length of the observed network packet in bytes.
2. `connection_duration`: Total active duration of the TCP/UDP session in seconds.
3. `failed_logins`: Frequency of unauthenticated or failed authorization attempts.
4. `bytes_transferred`: Cumulative payload transferred across the channel in kilobytes.
5. `src_to_dst_ratio`: Balance ratio between outbound and inbound transmitted packets.
6. `urgent_packets`: Count of packets containing high-priority/urgent flags.
7. `wrong_fragments`: Count of malformed or out-of-order fragment segments (e.g., teardrop attacks).
8. `num_compromised`: System states indicating anomalous security conditions.
9. `root_shell_attempts`: Observed attempts to spawn administrative/root privileged terminals.
10. `hot_indicators`: Access counts to sensitive system files, directories, or endpoints.

---

## 4. Setup & Installation

### Prerequisites
- **Python**: Version `3.10` or higher
- **Git**: Installed and available in terminal

### 1. Clone or Open the Repository
```bash
git clone <repository-url>
cd FAI
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute Unit Tests
Run the test suite to verify synthetic telemetry generation, train-test splitting, metric bounds, and inference routines:
```bash
pytest test_model.py -v
```

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
Upon execution, access the web console in your browser at:
`http://localhost:8501`

---

## 5. How to Use the Demo

### Step 1: Tune Hyperparameters & Retrain (Sidebar)
- Use the **Number of Estimators** slider ($10 - 300$) to control the ensemble capacity.
- Adjust **Maximum Tree Depth** ($2 - 25$) to balance model complexity against regularization.
- Change the **Random Seed Generator** to test performance across diverse synthetic data distributions.
- Click **🔄 Retrain Model** to immediately fit a new model and refresh all system telemetry.

### Step 2: Evaluate Model Diagnostics (Main Panel)
- **KPI Metrics**: Inspect test set **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- **Interactive Confusion Matrix**: Hover over the confusion matrix heatmap to analyze True Positives, False Positives, True Negatives, and False Negatives.
- **ROC-AUC Curve**: Examine the Receiver Operating Characteristic curve to verify discrimination ability above the baseline.
- **Feature Importance Chart**: View the ranked Gini impurity contributions to identify the leading threat indicators.

### Step 3: Run Live Packet Inference (Inference Panel)
- Utilize the sliders for the **Top 4 features** (dynamically determined by model importance) to craft custom packet scenarios.
- Click scenario presets for quick testing:
  - 🟢 **Benign Normal**: Typical business operations telemetry.
  - 🔴 **DoS / Flood**: High-throughput volumetric traffic signature.
  - ⚠️ **Brute-Force / Exploit**: Elevated login retries and privilege escalation signatures.
- Observe real-time classification badges (`🚨 THREAT DETECTED` or `🛡️ BENIGN TRAFFIC`) along with the probabilistic confidence gauge.
- Expand the **Inspect Raw Simulated Packet Telemetry Record** viewer for the complete feature payload and decision JSON payload.

---

## 6. Repository Structure

```text
├── app.py              # Streamlit interactive application and visualization dashboard
├── test_model.py       # Pytest unit tests for model training and inference validation
├── requirements.txt    # Project Python dependencies
└── README.md           # Academic documentation and project overview
```
