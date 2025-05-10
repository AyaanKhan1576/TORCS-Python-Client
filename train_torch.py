# train_torch.py
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    expand_lists, build_scaler, scale_row,
    encode_categories, ALL_NUM, TARGETS, save_preproc
)
from models import TORCSModel

# ───── Hyperparameters ─────
EPOCHS = 50
BATCH  = 128
LR     = 1e-3

# ───── 1. Load & concatenate all CSVs ─────
dfs = []
for fn in glob.glob("*.csv"):
    base = os.path.splitext(fn)[0]
    try:
        track, car, _ = base.split("_", 2)
    except ValueError:
        continue

    df = pd.read_csv(fn, low_memory=False)
    df = expand_lists(df)
    df["track_name"] = track
    df["car_name"]   = car

    # convert to numeric, drop bad rows
    df[ALL_NUM + TARGETS] = df[ALL_NUM + TARGETS].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.dropna(subset=TARGETS)
    if df.empty:
        continue

    df[ALL_NUM] = df[ALL_NUM].fillna(0.0)
    dfs.append(df)

if not dfs:
    raise RuntimeError("No valid CSV data found!")

df = pd.concat(dfs, ignore_index=True)

# ───── 2. Preprocess ─────
cat_idx, cat_maps = encode_categories(df)
scaler_stats      = build_scaler(df)

X_num = np.stack([scale_row(row, scaler_stats) for _, row in df.iterrows()])
y      = df[TARGETS].to_numpy(dtype=np.float32)

# ───── 3. Train/Val split ─────
Xn_tr, Xn_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
    X_num, cat_idx, y, test_size=0.2, random_state=42
)

train_ds = TensorDataset(
    torch.tensor(Xn_tr, dtype=torch.float32),
    torch.tensor(Xc_tr, dtype=torch.long),
    torch.tensor(y_tr,  dtype=torch.float32),
)
val_ds = TensorDataset(
    torch.tensor(Xn_val, dtype=torch.float32),
    torch.tensor(Xc_val, dtype=torch.long),
    torch.tensor(y_val,  dtype=torch.float32),
)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH)

# ───── 4. Build model ─────
emb_sizes = {k: max(m.values()) + 1 for k, m in cat_maps.items()}
model     = TORCSModel(len(ALL_NUM), emb_sizes, seq_len=1)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ───── 5. Training loop ─────
for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    total_train = 0.0
    for xb_num, xb_cat, yb in train_dl:
        pred = model(xb_num, xb_cat)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train += loss.item()
    avg_train = total_train / len(train_dl)

    # validate
    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for xb_num, xb_cat, yb in val_dl:
            total_val += criterion(model(xb_num, xb_cat), yb).item()
    avg_val = total_val / len(val_dl)

    print(f"Epoch {epoch:02}/{EPOCHS} | train_loss {avg_train:.4f} | val_loss {avg_val:.4f}")

# ───── 6. Save artifacts ─────
torch.save(model.state_dict(), "torcs_model.pt")
save_preproc(scaler_stats, cat_maps, "preproc.pkl")
print("✅ Training complete. Model and preproc saved.")
