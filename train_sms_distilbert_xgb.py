import os
import re
import time
import json
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
DATASET_PATH  = "dataset/spam_msg.csv"
MODEL_DIR     = "sms_xgb_model"
MAX_LENGTH    = 128       # SMS is short; 128 is more than enough
BATCH_SIZE    = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# PHISHING INDICATORS (same as original code)
# ==========================================
PHISHING_INDICATORS = {
    "urgency": ["urgent", "immediately", "right away", "act now", "expire",
                "suspended", "locked", "terminated", "limited time", "hurry",
                "last chance", "final warning"],
    "credentials": ["password", "ssn", "credit card", "bank account",
                    "pin number", "verify your", "confirm your", "update your"],
    "prizes": ["winner", "won", "prize", "congratulations", "lottery",
               "jackpot", "million", "free", "selected", "reward", "claim"],
    "action": ["click here", "click below", "click this", "call now",
               "text to", "reply to", "send to", "log in", "sign in"],
    "financial": ["cash", "money", "pounds", "dollars", "credit",
                  "offer", "discount", "deal", "cost", "price"],
    "threat": ["unauthorized", "suspicious", "compromised", "hacked",
               "blocked", "disabled", "fraud"],
    "impersonation": ["dear customer", "dear user", "dear member", "dear valued",
                      "dear account holder", "dear sir/madam", "dear beneficiary"],
    "spam_words": ["viagra", "pharmacy", "replica", "cheap",
                   "buy now", "order now", "special offer", "unsubscribe",
                   "opt out", "bulk email", "mass email"]
}
INDICATOR_CATEGORIES = list(PHISHING_INDICATORS.keys())


# ==========================================
# FEATURE EXTRACTION
# ==========================================
def count_indicators(text: str):
    text_lower = text.lower()
    triggered, total = 0, 0
    for kws in PHISHING_INDICATORS.values():
        hit = False
        for kw in kws:
            if kw in text_lower:
                total += 1
                hit = True
        if hit:
            triggered += 1
    return triggered, total


def extract_sms_features(text: str) -> list:
    """Structural + phishing-indicator feature vector for one SMS."""
    text_lower = text.lower()

    # Structural
    has_url        = int(bool(re.search(r'http\S+|www\.\S+', text_lower)))
    has_phone      = int(bool(re.search(r'\b\d{5,}\b', text_lower)))
    has_shortener  = int(bool(re.search(r'bit\.ly|tinyurl|goo\.gl|t\.co', text_lower)))
    url_count      = len(re.findall(r'http\S+|www\.\S+', text_lower))
    exclamation    = text.count('!')
    caps_ratio     = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    text_length    = len(text)
    word_count     = len(text.split())
    digit_count    = sum(c.isdigit() for c in text)
    special_count  = sum(not c.isalnum() and not c.isspace() for c in text)

    # Phishing frequency
    cats_hit, total_matches = count_indicators(text)

    # Per-category binary flags
    cat_flags = [
        int(any(kw in text_lower for kw in kws))
        for kws in PHISHING_INDICATORS.values()
    ]

    return [
        has_url, has_phone, has_shortener, url_count, exclamation,
        caps_ratio, text_length, word_count, digit_count, special_count,
        cats_hit, total_matches
    ] + cat_flags     # total: 12 + 8 = 20 dims


def clean_for_tfidf(text: str) -> str:
    """Keep URL token for TF-IDF (matches original training code)."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)   # replace with token
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


# ==========================================
# DISTILBERT EMBEDDING EXTRACTOR
# ==========================================
def extract_distilbert_embeddings(texts, tokenizer, model, batch_size=64, max_len=128):
    """Returns (N, 768) float32 array of mean-pooled DistilBERT embeddings."""
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="DistilBERT embeddings"):
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
            out    = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state                       # (B, L, 768)
            mask   = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pool = (summed / counts).cpu().numpy()

        all_embeddings.append(mean_pool.astype(np.float32))

    return np.vstack(all_embeddings)


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("DistilBERT + XGBoost SMS Phishing Training")
    print("=" * 60)

    # ------- 1. Load & Prepare Data -------
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(DATASET_PATH, encoding="latin1")

    # Support both column name styles
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]]
    else:
        raise ValueError("Dataset must have columns (v1,v2) or (label,text)")

    if df["label"].dtype == object:
        df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["text", "label"])

    print(f"  Total rows: {len(df)}")
    print(f"  Label dist: {df['label'].value_counts().to_dict()}")

    # ---- NO upsampling: use original imbalanced data ----
    # Upsampling with replace=True creates duplicate rows that leak into the test
    # set, causing inflated (100%) accuracy. Instead pass class weights to XGBoost
    # via scale_pos_weight so it handles imbalance natively.
    n_ham  = (df["label"] == 0).sum()
    n_spam = (df["label"] == 1).sum()
    spw    = n_ham / n_spam   # e.g. 4825 / 747 ≈ 6.46
    print(f"  scale_pos_weight = {spw:.2f}  (no upsampling — avoids data leakage)")
    df_bal = df.copy()

    df_bal["text_clean"] = df_bal["text"].apply(clean_for_tfidf)

    # ------- 2. Phishing Indicator + Structural Features -------
    print("\n[2/6] Building phishing indicator & structural features...")
    X_feats = np.array([extract_sms_features(t) for t in df_bal["text"]], dtype=np.float32)
    print(f"  Feature shape: {X_feats.shape}")

    # ------- 3. TF-IDF -------
    print("\n[3/6] TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(df_bal["text_clean"])
    print(f"  TF-IDF shape: {X_tfidf.shape}")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print("  Saved tfidf_vectorizer.pkl")

    # ------- 4. DistilBERT Embeddings -------
    print("\n[4/6] Extracting DistilBERT mean-pool embeddings...")
    tokenizer  = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    for p in bert_model.parameters():
        p.requires_grad = False

    texts = df_bal["text"].tolist()
    t0 = time.time()
    X_bert = extract_distilbert_embeddings(texts, tokenizer, bert_model, BATCH_SIZE, MAX_LENGTH)
    print(f"  BERT shape: {X_bert.shape}  (took {(time.time()-t0)/60:.1f} min)")
    np.save(os.path.join(MODEL_DIR, "bert_embeddings_cache.npy"), X_bert)
    print("  Saved bert_embeddings_cache.npy")

    # ------- 5. Stack & Split -------
    print("\n[5/6] Stacking features and splitting...")
    X_dense = np.hstack([X_bert, X_feats])           # (N, 768+20)
    X_all   = hstack([X_tfidf, csr_matrix(X_dense)])
    y       = df_bal["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ------- 6. Train XGBoost + Calibrate -------
    print("\n[6/6] Training XGBoost + isotonic calibration...")

    base_model = XGBClassifier(
        n_estimators=300,          # fewer trees → less overfitting on small data
        max_depth=4,               # shallower trees → better generalization
        learning_rate=0.05,        # slower learning → less memorization
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,        # higher → less splitting on rare spam patterns
        gamma=0.2,
        reg_alpha=0.3,             # L1 regularization
        reg_lambda=2.0,            # L2 regularization
        scale_pos_weight=spw,      # handles class imbalance WITHOUT upsampling
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )

    # Use sigmoid (Platt scaling) — better calibration for small datasets
    # isotonic needs many samples; sigmoid works well with < 10k
    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)

    t0 = time.time()
    calibrated_model.fit(X_train, y_train)
    print(f"  Training done in {(time.time()-t0)/60:.1f} min")

    # ------- Evaluation -------
    y_pred  = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe/Ham", "Phishing/Spam"]))

    print(f"\nProbability distribution (check it spans 0–1):")
    print(f"  Min : {y_proba.min():.4f}")
    print(f"  Max : {y_proba.max():.4f}")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Std : {y_proba.std():.4f}")
    bins = np.histogram(y_proba, bins=10, range=(0, 1))[0]
    print(f"  Histogram (0→1): {bins.tolist()}")

    # ------- Save -------
    joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "xgb_calibrated_model.pkl"))
    print(f"\nModel saved → {MODEL_DIR}/xgb_calibrated_model.pkl")

    meta = {
        "model_type": "distilbert_xgb",
        "max_length": MAX_LENGTH,
        "bert_model_name": "distilbert-base-uncased",
        "feature_order": ["tfidf", "bert_meanpool", "phishing_indicators", "structural"],
        "indicator_categories": INDICATOR_CATEGORIES,
        "n_indicator_features": 20,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta.json")

    print("\n" + "=" * 60)
    print("SMS TRAINING COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
