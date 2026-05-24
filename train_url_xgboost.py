import re
import json
import numpy as np
import pandas as pd
import joblib
import os
from collections import Counter
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

# ==========================================
# CONFIG
# ==========================================
DATASET_PATH  = "dataset/malicious_phish_url.csv"
MODEL_DIR     = "url_model_dir"          # new folder to keep things tidy
SAMPLES_PER_CLASS = 50000               # cap each class to keep training fast

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# DOMAIN SIGNALS
# ==========================================
KNOWN_BRANDS = [
    "paypal", "netflix", "apple", "microsoft", "amazon",
    "bank", "chase", "wellsfargo", "google", "facebook",
    "instagram", "twitter", "ebay", "dropbox", "linkedin"
]
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "billing",
    "confirm", "invoice", "signin", "password", "credential", "alert",
    "suspend", "locked", "unusual", "recover", "validate", "access",
    "authenticate", "authorize", "reset", "support"
]
SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".buzz", ".info",
    ".click", ".link", ".work", ".date", ".racing", ".win", ".bid",
    ".stream", ".download", ".party", ".loan", ".men", ".faith"
]

# ==========================================
# URL CLEANING (for TF-IDF)
# ==========================================
def clean_url(url: str) -> str:
    url = str(url).lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www\.", "", url)
    url = re.sub(r"[^a-z0-9./\-]", " ", url)
    return re.sub(r"\s+", " ", url).strip()


# ==========================================
# STRUCTURAL FEATURE EXTRACTION (22 dims)
# ==========================================
def extract_url_features(raw_url: str) -> list:
    url_lower = str(raw_url).lower()

    # Basic counts
    url_len      = len(raw_url)
    num_dots     = raw_url.count(".")
    num_hyphens  = raw_url.count("-")
    num_slashes  = raw_url.count("/")
    num_digits   = sum(c.isdigit() for c in raw_url)
    num_at       = raw_url.count("@")
    num_ques     = raw_url.count("?")
    num_amp      = raw_url.count("&")
    num_eq       = raw_url.count("=")
    num_percent  = raw_url.count("%")   # URL encoding is suspicious

    # Protocol flags
    has_https    = int("https://" in url_lower)
    has_http     = int("http://" in url_lower and "https://" not in url_lower)
    no_protocol  = int(not url_lower.startswith(("http://", "https://")))

    # Domain parsing
    try:
        parsed_url = url_lower if url_lower.startswith(("http://", "https://")) else "http://" + url_lower
        parsed  = urlparse(parsed_url)
        domain  = parsed.netloc
        path    = parsed.path
    except Exception:
        domain, path = "", ""

    domain_len     = len(domain)
    path_len       = len(path)
    num_subdomains = max(0, domain.count(".") - 1)

    # IP in domain
    has_ip = int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain)))

    # Brand spoof (brand name in domain but not at root)
    brand_spoof = sum(int(b in domain) for b in KNOWN_BRANDS)

    # Suspicious keyword count
    suspicious_count = sum(int(w in url_lower) for w in SUSPICIOUS_WORDS)

    # Suspicious TLD
    has_suspicious_tld = int(any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS))

    # Domain entropy (random-looking = high entropy)
    domain_entropy = 0.0
    if domain:
        freq   = Counter(domain)
        total  = len(domain)
        domain_entropy = -sum((c / total) * np.log2(c / total) for c in freq.values())

    # Digit ratio in domain
    digit_ratio = sum(c.isdigit() for c in domain) / max(len(domain), 1)

    # Port detection
    has_port = int(":" in domain.split(".")[-1]) if domain else 0

    return [
        url_len, num_dots, num_hyphens, num_slashes, num_digits,
        num_at, num_ques, num_amp, num_eq, num_percent,
        has_https, has_http, no_protocol,
        domain_len, path_len, num_subdomains,
        has_ip, brand_spoof, suspicious_count,
        has_suspicious_tld, domain_entropy, digit_ratio, has_port
    ]  # 23 features


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("URL Phishing Detection — XGBoost Training")
    print("=" * 60)

    # ------- 1. Load & Label -------
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    # benign=0, everything else (phishing, malware, defacement)=1
    df["label"] = (df["type"].str.lower() != "benign").astype(int)
    print(f"  Total rows: {len(df)}")
    print(f"  Label dist: {df['label'].value_counts().to_dict()}")

    # Balance
    df_safe  = df[df["label"] == 0].sample(n=min(len(df[df["label"]==0]), SAMPLES_PER_CLASS), random_state=42)
    df_mal   = df[df["label"] == 1].sample(n=min(len(df[df["label"]==1]), SAMPLES_PER_CLASS), random_state=42)
    df = pd.concat([df_safe, df_mal]).sample(frac=1, random_state=42).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"  After balancing: {len(df)} ({df['label'].value_counts().to_dict()})")

    # ------- 2. Structural Features -------
    print("\n[2/5] Extracting structural features...")
    X_struct = np.array([extract_url_features(u) for u in df["url"]], dtype=np.float32)
    print(f"  Structural shape: {X_struct.shape}")

    # ------- 3. TF-IDF Char N-Grams -------
    print("\n[3/5] TF-IDF char n-gram vectorization...")
    df["url_clean"] = df["url"].apply(clean_url)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=30000,
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(df["url_clean"])
    print(f"  TF-IDF shape: {X_tfidf.shape}")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "url_vectorizer.pkl"))
    print("  Saved url_vectorizer.pkl")

    # ------- 4. Combine & Split -------
    print("\n[4/5] Stacking features & splitting...")
    X_all = hstack([X_tfidf, csr_matrix(X_struct)])
    y     = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ------- 5. Train XGBoost + Calibrate -------
    print("\n[5/5] Training XGBoost + isotonic calibration (fixes extreme probs)...")

    base_model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )

    # CalibratedClassifierCV wraps XGBoost — prevents 0.99/0.01 only outputs
    calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated_model.fit(X_train, y_train)

    # ------- Evaluation -------
    y_pred  = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe/Benign", "Malicious"]))

    print("\nProbability distribution (should span 0–1):")
    print(f"  Min : {y_proba.min():.4f}")
    print(f"  Max : {y_proba.max():.4f}")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Std : {y_proba.std():.4f}")
    bins = np.histogram(y_proba, bins=10, range=(0, 1))[0]
    print(f"  Histogram (0→1): {bins.tolist()}")

    # ------- Save -------
    joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "url_model.pkl"))
    print(f"\nSaved → {MODEL_DIR}/url_model.pkl")
    print(f"Saved → {MODEL_DIR}/url_vectorizer.pkl")

    # Also overwrite root-level files so app.py finds them immediately
    joblib.dump(calibrated_model, "url_model.pkl")
    joblib.dump(vectorizer, "url_vectorizer.pkl")
    print("Also overwrote root url_model.pkl & url_vectorizer.pkl (used by app.py)")

    meta = {
        "model_type": "xgb_calibrated",
        "n_tfidf_features": 30000,
        "n_structural_features": 23,
        "feature_order": ["tfidf_char_ngrams", "structural"],
        "threshold": 0.70
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta.json")

    print("\n" + "=" * 60)
    print("URL TRAINING COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
