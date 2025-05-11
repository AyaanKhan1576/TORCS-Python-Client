# train_torch_seq.py
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
EPOCHS   = 20          # more epochs for sequence training
BATCH    = 64          # moderate batch size
LR       = 1e-3        # learning rate
SEQ_LEN  = 5           # use past 5 frames to predict next control
PENALTY  = 0.1         # weight for positional/angle penalty

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
    df[ALL_NUM + TARGETS] = df[ALL_NUM + TARGETS].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.dropna(subset=TARGETS)
    if df.empty:
        continue
    df[ALL_NUM] = df[ALL_NUM].fillna(0.0)
    df['track_name'], df['car_name'] = track, car
    dfs.append(df)

if not dfs:
    raise RuntimeError("No valid CSV data found!")

full_df = pd.concat(dfs, ignore_index=True)

# ───── 2. Preprocess & encode ─────
cat_idx, cat_maps = encode_categories(full_df)
scaler_stats      = build_scaler(full_df)

# build raw arrays
X_num_all = np.stack([scale_row(row, scaler_stats) for _, row in full_df.iterrows()])
y_all     = full_df[TARGETS].to_numpy(dtype=np.float32)

# find indices for penalty features
i_track = ALL_NUM.index('trackPos')
i_angle = ALL_NUM.index('angle')

# ───── 3. Sequence construction ─────
def make_sequences(X_num, cat_idx, y, seq_len):
    X_seq, C_seq, Y_seq = [], [], []
    for i in range(len(X_num) - seq_len - 1):
        X_seq.append(X_num[i:i+seq_len])
        C_seq.append(cat_idx[i])
        # predict the action at the next frame after sequence
        Y_seq.append(y[i+seq_len])
    return (
        np.stack(X_seq),
        np.array(C_seq, dtype=np.int64),
        np.stack(Y_seq)
    )

Xn, Xc, y_seq = make_sequences(X_num_all, cat_idx, y_all, SEQ_LEN)

# ───── 4. Train/Val split ─────
Xn_tr, Xn_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
    Xn, Xc, y_seq, test_size=0.2, random_state=42
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

# ───── 5. Build model ─────
emb_sizes = {k: max(m.values()) + 1 for k, m in cat_maps.items()}
model     = TORCSModel(len(ALL_NUM), emb_sizes, seq_len=SEQ_LEN)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ───── 6. Training loop with penalty ─────
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb_num, xb_cat, yb in train_dl:
        pred = model(xb_num, xb_cat)
        # basic control loss
        loss = criterion(pred, yb)
        # penalty for off-center track position & angle extremes
        last = xb_num[:, -1, :]
        track_pen = torch.mean(torch.abs(last[:, i_track]))
        angle_pen = torch.mean(torch.abs(last[:, i_angle]))
        loss = loss + PENALTY * (track_pen + angle_pen)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train = train_loss / len(train_dl)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb_num, xb_cat, yb in val_dl:
            pred = model(xb_num, xb_cat)
            loss_val = criterion(pred, yb)
            last = xb_num[:, -1, :]
            track_pen = torch.mean(torch.abs(last[:, i_track]))
            angle_pen = torch.mean(torch.abs(last[:, i_angle]))
            loss_val = loss_val + PENALTY * (track_pen + angle_pen)
            val_loss += loss_val.item()
    avg_val = val_loss / len(val_dl)

    print(f"Epoch {epoch:02}/{EPOCHS} | train_loss {avg_train:.4f} | val_loss {avg_val:.4f}")

# ───── 7. Save artifacts ─────
torch.save(model.state_dict(), "torcs_model.pt")
save_preproc(scaler_stats, cat_maps, "preproc.pkl")
print("✅ Sequence training complete with penalties. Model and preproc saved.")
