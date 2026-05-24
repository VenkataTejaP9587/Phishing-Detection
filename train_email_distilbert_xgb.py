import os
import re
import time
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm
# ==========================================
# CONFIG
# ==========================================
DATASET_PATH  = "dataset/CEAS_08.csv"
MODEL_DIR     = "email_xgb_model"
MAX_LENGTH    = 128      # shorter = faster; 128 is plenty for CLS token
BATCH_SIZE    = 64       # For DistilBERT embedding extraction
SAMPLES_PER_CLASS = 18000  # balance classes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# PHISHING KEYWORDS
# ==========================================
PHISHING_KEYWORDS = [
    "verify", "suspend", "account", "click here", "confirm", "urgent",
    "password", "credit card", "ssn", "bank", "login", "update your",
    "expire", "immediately", "winner", "won", "prize", "congratulations",
    "lottery", "free", "offer", "limited time", "act now", "risk",
    "unauthorized", "security alert", "unusual activity", "locked",
    "delivery failed", "invoice", "payment", "refund", "claim",
    "bitcoin", "wire transfer", "gift card", "social security",
    "dear customer", "dear user", "dear member", "click below",
    "unsubscribe", "viagra", "pharmacy", "discount", "buy now",
    "replica", "cheap", "order now", "special offer", "deal",
    "million dollars", "inheritance", "beneficiary", "nigeria"
]

# ==========================================
# TEXT UTILS
# ==========================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def keyword_features(text: str) -> list:
    text_lower = str(text).lower()
    hits = [int(kw in text_lower) for kw in PHISHING_KEYWORDS]
    hits.append(sum(hits))        # total keyword hit count
    hits.append(len(text_lower))  # character length
    return hits

def structural_features(row) -> list:
    subject = str(row.get("subject", ""))
    body    = str(row.get("body", ""))
    sender  = str(row.get("sender", ""))
    combined = subject + " " + body

    word_count    = len(combined.split())
    body_len      = len(body)
    subject_len   = len(subject)
    sender_len    = len(sender)
    has_html      = int("<html" in body.lower() or "<a href" in body.lower())
    exclamation   = body.lower().count("!")
    dollar_signs  = body.lower().count("$")
    all_caps_words = sum(1 for w in combined.split() if w.isupper() and len(w) > 2)
    url_count     = len(re.findall(r"http\S+|www\S+", body.lower()))
    digit_ratio   = sum(c.isdigit() for c in combined) / max(len(combined), 1)

    return [
        word_count, body_len, subject_len, sender_len,
        has_html, exclamation, dollar_signs,
        all_caps_words, url_count, digit_ratio
    ]

# ==========================================
# DISTILBERT EMBEDDING EXTRACTOR (This function converts emails into semantic embeddings)
# ==========================================
def extract_distilbert_embeddings(texts, tokenizer, model, batch_size=64, max_len=128):

    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting DistilBERT embeddings"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden  = outputs.last_hidden_state  # (B, seq_len, 768)

        # Mean-pool over non-padding tokens for richer representation
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = (summed / counts).cpu().numpy()

        all_embeddings.append(mean_pooled)

    return np.vstack(all_embeddings).astype(np.float32)

# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("DistilBERT + XGBoost Email Phishing Training")
    print("=" * 60)

    # ------- 1. Load & Prepare Data -------
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df["sender"]  = df["sender"].fillna("")
    df["subject"] = df["subject"].fillna("")
    df["body"]    = df["body"].fillna("")
    df = df.drop_duplicates().dropna(subset=["label"])
    df["label"]   = df["label"].astype(int)
    print(f"  Total rows: {len(df)}")
    print(f"  Label dist: {df['label'].value_counts().to_dict()}")

    # Balance
    df = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), SAMPLES_PER_CLASS), random_state=42))
          .reset_index(drop=True)
    )
    print(f"  After balancing: {len(df)} ({SAMPLES_PER_CLASS}/class)")

    # Combined text (for TF-IDF + DistilBERT)
    df["combined"] = (
        "sender: " + df["sender"]
        + " subject: " + df["subject"]
        + " body: " + df["body"]
    )
    df["combined_clean"] = df["combined"].apply(clean_text)

    # ------- 2. Keyword & Structural Features -------
    print("\n[2/6] Building keyword & structural features...")
    kw_feats   = np.array([keyword_features(t) for t in df["combined"]], dtype=np.float32)
    st_feats   = np.array([structural_features(row) for _, row in df.iterrows()], dtype=np.float32)
    print(f"  Keyword features shape: {kw_feats.shape}")
    print(f"  Structural features shape: {st_feats.shape}")

    # ------- 3. TF-IDF Features -------
    print("\n[3/6] TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(df["combined_clean"])
    print(f"  TF-IDF shape: {X_tfidf.shape}")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print("  Saved tfidf_vectorizer.pkl")

    # ------- 4. DistilBERT Embeddings -------
    print("\n[4/6] Extracting DistilBERT (mean-pool) embeddings...")
    tokenizer   = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bert_model  = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    
    # Freeze all – we only use it as a feature extractor here
    for p in bert_model.parameters():
        p.requires_grad = False
    
    texts = df["combined"].tolist()
    t0 = time.time()
    X_bert = extract_distilbert_embeddings(texts, tokenizer, bert_model, BATCH_SIZE, MAX_LENGTH)
    print(f"  DistilBERT shape: {X_bert.shape}  (took {(time.time()-t0)/60:.1f} min)")

    # Save embeddings cache so you don't re-run DistilBERT on every retrain
    np.save(os.path.join(MODEL_DIR, "bert_embeddings_cache.npy"), X_bert)
    print("  Saved bert_embeddings_cache.npy")

    # ------- 5. Stack All Features & Split -------
    print("\n[5/6] Stacking features & splitting...")
    X_dense = np.hstack([X_bert, kw_feats, st_feats]) 
    X_all   = hstack([X_tfidf, csr_matrix(X_dense)])
    y       = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ------- 6. Train XGBoost + Calibrate -------
    print("\n[6/6] Training XGBoost + Platt-scaling calibration...")

    base_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",   # faster
        n_jobs=-1
    )

    # CalibratedClassifierCV wraps XGBoost with Platt scaling (sigmoid)
    # This is the KEY fix for extreme probabilities.
    # method='isotonic' gives a piecewise-monotone calibration (better for large data)
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="isotonic",   # or "sigmoid" for smaller datasets
        cv=3
    )

    print("  Fitting calibrated model (this may take a while on large data)...")
    t0 = time.time()
    calibrated_model.fit(X_train, y_train)
    print(f"  Training done in {(time.time()-t0)/60:.1f} min")

    # ------- Evaluation -------
    y_pred  = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    # Show probability distribution to verify it's no longer polarised
    print(f"\nProbability distribution (should span 0-1):")
    print(f"  Min prob : {y_proba.min():.4f}")
    print(f"  Max prob : {y_proba.max():.4f}")
    print(f"  Mean prob: {y_proba.mean():.4f}")
    print(f"  Std prob : {y_proba.std():.4f}")
    bins = np.histogram(y_proba, bins=10, range=(0, 1))[0]
    print(f"  Histogram (0→1): {bins.tolist()}")

    # ------- Save -------
    joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "xgb_calibrated_model.pkl"))
    print(f"\nModel saved to {MODEL_DIR}/xgb_calibrated_model.pkl")
    print("Vectorizer saved to {MODEL_DIR}/tfidf_vectorizer.pkl")

    # Save metadata so app.py knows what to load
    import json
    meta = {
        "model_type": "distilbert_xgb",
        "max_length": MAX_LENGTH,
        "bert_model_name": "distilbert-base-uncased",
        "feature_order": ["tfidf", "bert_meanpool", "keyword", "structural"],
        "keyword_count": len(PHISHING_KEYWORDS) + 2,
        "structural_count": 10,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta.json")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE ✓")
    print("=" * 60)

if __name__ == "__main__":
    main()