"""
Training script: DistilBERT + Bidirectional LSTM for SMS Phishing Detection
Specifically adapted to handle short messages and prevent false-positives on benign notifications.
"""

import os
import sys
import re
import time
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# CONFIG & HYPERPARAMETERS
# ==========================================
DATASET_PATH = "dataset/spam_msg.csv"
MODEL_SAVE_DIR = "sms_model_bilstm"
MAX_LENGTH = 160            # Increase max length slightly
BATCH_SIZE = 16
EPOCHS = 12                 # Slightly more epochs
LEARNING_RATE = 2e-5        # Lower LR for deeper fine-tuning
LSTM_HIDDEN = 256
LSTM_LAYERS = 2             # 2 LSTM layers for better feature capture

# Regularization
DROPOUT = 0.3               # Lower dropout to capture more patterns
WEIGHT_DECAY = 1e-4     
LABEL_SMOOTHING = 0.05
PATIENCE = 4                # More patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}", flush=True)

# ==========================================
# TEXT CLEANING
# ==========================================
def clean_sms(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# ==========================================
# DATA AUGMENTATION (to expand tiny phishing dataset)
# ==========================================
def synonym_replace(text, n=2):
    """Simple word shuffle augmentation for short texts."""
    words = text.split()
    if len(words) < 4:
        return text
    augmented = []
    for _ in range(n):
        w = words.copy()
        # Swap two random words
        if len(w) >= 2:
            i, j = random.sample(range(len(w)), 2)
            w[i], w[j] = w[j], w[i]
        augmented.append(" ".join(w))
    return augmented

def random_deletion(text, p=0.1):
    """Randomly delete words with probability p."""
    words = text.split()
    if len(words) <= 3:
        return text
    remaining = [w for w in words if random.random() > p]
    if len(remaining) == 0:
        return text
    return " ".join(remaining)

def augment_phishing_data(texts, labels, augment_factor=8):
    """Augment phishing (label=1) samples aggressively to balance dataset."""
    aug_texts = list(texts)
    aug_labels = list(labels)
    
    phishing_texts = [t for t, l in zip(texts, labels) if l == 1]
    
    for _ in range(augment_factor):
        for text in phishing_texts:
            # Word swap augmentation
            swapped = synonym_replace(text, n=2)
            if isinstance(swapped, list):
                for s in swapped:
                    aug_texts.append(s)
                    aug_labels.append(1)
            
            # Random deletion augmentation  
            deleted = random_deletion(text, p=0.20)
            if deleted != text:
                aug_texts.append(deleted)
                aug_labels.append(1)
    
    return aug_texts, aug_labels

# ==========================================
# DATASET
# ==========================================
class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ==========================================
# MODEL: DistilBERT + BiLSTM
# ==========================================
class DistilBertBiLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=2,
                 dropout=0.4, pooling="last_timestep"):
        super().__init__()
        self.pooling = pooling

        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze all BERT layers first
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Unfreeze last 3 transformer layers for deeper fine-tuning
        for layer in self.distilbert.transformer.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
    
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # Fine-tune unfrozen layers (no torch.no_grad here)
        bert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = bert_output.last_hidden_state
        lstm_out, (hidden, _) = self.bilstm(sequence_output)

        if self.pooling == "last_timestep":
            pooled = lstm_out[:, -1, :]
        else:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            pooled = torch.cat((forward_hidden, backward_hidden), dim=1)

        out = self.dropout(pooled)
        logits = self.classifier(out)
        return logits


# ==========================================
# TRAINING FUNCTION
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    start_time = time.time()
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
            print(f"  Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f} | ETA: {eta/60:.1f}min", flush=True)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) 

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"  --> Regularization check: Highest prob: {max(all_probs):.4f}, Lowest prob: {min(all_probs):.4f}")
    
    return avg_loss, accuracy, all_preds, all_labels


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("DistilBERT + BiLSTM SMS Phishing Training (Improved)")
    print("=" * 60)

    # --- Load Data ---
    print(f"\n[1/5] Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, encoding='latin-1')
    df = df[['v1', 'v2']].dropna()
    print(f"  Total samples: {len(df)}", flush=True)

    # Map target
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
    
    print(f"  Original distribution:\n{df['label'].value_counts().to_string()}", flush=True)
    
    # Use ALL data (don't cap to 700 — we need every sample)
    df["clean_msg"] = df["v2"].apply(clean_sms)
    texts = df["clean_msg"].tolist()
    labels = df["label"].tolist()
    
    # Augment phishing samples aggressively to balance classes
    print("  Augmenting phishing samples aggressively...", flush=True)
    texts, labels = augment_phishing_data(texts, labels, augment_factor=8)
    
    # Count after augmentation
    n_safe = sum(1 for l in labels if l == 0)
    n_phish = sum(1 for l in labels if l == 1)
    print(f"  After augmentation: {n_safe} safe, {n_phish} phishing ({len(texts)} total)", flush=True)

    # --- Split Data ---
    print("\n[2/5] Splitting data (80/20)...", flush=True)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(train_texts)} | Val: {len(val_texts)}", flush=True)

    # --- Tokenizer & Datasets ---
    print("\n[3/5] Preparing tokenizer and dataloaders...", flush=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    print("  Tokenizer loaded.", flush=True)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    train_dataset = SMSDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SMSDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model ---
    print("\n[4/5] Building DistilBERT + BiLSTM model (last 2 BERT layers unfrozen)...", flush=True)
    model = DistilBertBiLSTM(
        num_classes=2,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        pooling="last_timestep",
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}", flush=True)
    print(f"  Trainable parameters: {trainable_params:,} (BERT last 2 layers + BiLSTM + classifier)")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    # Separate param groups: lower LR for BERT, higher for BiLSTM
    bert_params = [p for n, p in model.named_parameters() if 'distilbert' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'distilbert' not in n and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': LEARNING_RATE},
        {'params': other_params, 'lr': LEARNING_RATE * 3}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # --- Training Loop ---
    print(f"\n[5/5] Training ({EPOCHS} epochs, early stopping patience={PATIENCE})...")
    print("-" * 60)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_preds, val_labels_list = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "lstm_hidden": LSTM_HIDDEN,
                "lstm_layers": LSTM_LAYERS,
                "dropout": DROPOUT,
                "num_classes": 2,
                "max_length": MAX_LENGTH,
                "pooling": "last_timestep",
            }, os.path.join(MODEL_SAVE_DIR, "model.pt"))

            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            print(f"  [SUCCESS] Best model saved! (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch + 1}!")
                break

    # --- Final Report ---
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, "model.pt"), map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, DEVICE)

    print("\nClassification Report:")
    print(classification_report(
        final_labels, final_preds,
        target_names=["Safe (0)", "Phishing (1)"]
    ))

    print(f"\nModel saved to: {MODEL_SAVE_DIR}/")

if __name__ == "__main__":
    main()
