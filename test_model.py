import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def generate_network_data(n_samples=1000, random_state=42):
    """
    Generates realistic synthetic network traffic data for IDS demonstration.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.6, 0.4], # 60% normal traffic, 40% malicious
        flip_y=0.03,         # 3% noise to simulate real-world anomaly edge cases
        random_state=random_state
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    
    # Scale/transform features slightly to simulate realistic network metric magnitudes
    # (keeps distributions informative while making values intuitive)
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

    df['target'] = y
    return df

def train_ids_model(df, n_estimators=100, max_depth=10, random_state=42):
    """
    Trains a Random Forest Classifier on an 80/20 train-test split.
    """
    X = df[FEATURE_NAMES]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    return clf, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics

# --- UNIT TESTS ---

def test_generate_network_data_shape_and_features():
    df = generate_network_data(n_samples=1000)
    assert df.shape == (1000, 11)
    for col in FEATURE_NAMES:
        assert col in df.columns
    assert 'target' in df.columns
    assert set(df['target'].unique()).issubset({0, 1})

def test_model_training_and_metrics():
    df = generate_network_data(n_samples=1000)
    clf, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics = train_ids_model(df)
    
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert len(y_pred) == 200
    assert len(y_prob) == 200
    
    # Model should have strong predictive performance on synthetic data
    assert metrics['accuracy'] >= 0.70
    assert metrics['precision'] >= 0.70
    assert metrics['recall'] >= 0.70
    assert metrics['f1'] >= 0.70
    assert metrics['roc_auc'] >= 0.70

def test_feature_importances():
    df = generate_network_data(n_samples=1000)
    clf, _, _, _, _, _, _, _ = train_ids_model(df)
    importances = clf.feature_importances_
    assert len(importances) == 10
    assert np.isclose(np.sum(importances), 1.0)
    assert all(i >= 0 for i in importances)

def test_custom_packet_inference():
    df = generate_network_data(n_samples=1000)
    clf, _, _, _, _, _, _, _ = train_ids_model(df)
    
    # Synthesize sample packet record
    sample_packet = pd.DataFrame([{col: df[col].median() for col in FEATURE_NAMES}])
    prediction = clf.predict(sample_packet)[0]
    prob = clf.predict_proba(sample_packet)[0]
    
    assert prediction in [0, 1]
    assert len(prob) == 2
    assert np.isclose(np.sum(prob), 1.0)
