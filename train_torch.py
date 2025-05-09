# # train_torch.py -------------------------------------------------------
# # Train a track- & car-aware MLP on logged driving_data.csv
# import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from utils import expand_lists, build_scaler, scale_row, encode_categories, \
#                   ALL_NUM, TARGETS, save_preproc

# CSV_FILE   = "driving_data_test.csv"
# MODEL_FILE = "torcs_model.pt"
# EPOCHS     = 12
# BATCH      = 1024

# # ── load & preprocess -------------------------------------------------
# df = pd.read_csv(CSV_FILE)
# df = expand_lists(df).dropna(subset=ALL_NUM + TARGETS)

# cat_idx, cat_maps = encode_categories(df)
# scaler_stats      = build_scaler(df)

# X_num = np.stack([scale_row(r, scaler_stats) for _, r in df.iterrows()])
# y     = df[TARGETS].to_numpy(dtype=np.float32)

# Xn_tr, Xn_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
#     X_num, cat_idx, y, test_size=0.15, random_state=42)

# tr_ds = TensorDataset(torch.tensor(Xn_tr), torch.tensor(Xc_tr), torch.tensor(y_tr))
# va_ds = TensorDataset(torch.tensor(Xn_val), torch.tensor(Xc_val), torch.tensor(y_val))

# tr_dl = DataLoader(tr_ds, BATCH, shuffle=True)
# va_dl = DataLoader(va_ds, BATCH)

# # ── model -------------------------------------------------------------
# EMB_SZ = {k: max(m.values()) + 1 for k, m in cat_maps.items()}

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.emb_track = nn.Embedding(EMB_SZ["track_name"], 8)
#         self.emb_car   = nn.Embedding(EMB_SZ["car_name"],   4)
#         self.net = nn.Sequential(
#             nn.Linear(len(ALL_NUM) + 12, 256), nn.ReLU(),
#             nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(256, 128), nn.ReLU(),
#             nn.Linear(128, 3), nn.Tanh()     # steer ∈ [-1,1]
#         )
#     def forward(self, xn, xc):
#         z = torch.cat([xn,
#                        self.emb_track(xc[:, 0]),
#                        self.emb_car(xc[:, 1])], 1)
#         out = self.net(z)
#         steer = out[:, 0]
#         accel = torch.clamp(out[:, 1], 0, 1)
#         brake = torch.clamp(out[:, 2], 0, 1)
#         return torch.stack([steer, accel, brake], 1)

# model = MLP()
# opt   = optim.Adam(model.parameters(), 3e-4)
# mse   = nn.MSELoss()

# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     for xn, xc, yb in tr_dl:
#         loss = mse(model(xn, xc), yb)
#         opt.zero_grad(); loss.backward(); opt.step()
#     model.eval()
#     with torch.no_grad():
#         val_loss = sum(mse(model(xn, xc), yb).item() for xn, xc, yb in va_dl) / len(va_dl)
#     print(f"Epoch {epoch:02}/{EPOCHS}  val MSE {val_loss:.5f}")

# torch.save(model.state_dict(), MODEL_FILE)
# save_preproc(scaler_stats, cat_maps)
# print("✓ saved", MODEL_FILE, "and preproc.pkl")


# train_model.py ---------------------------------------------------------------
# Train a neural network that learns from every CSV in the current directory.
# Each CSV must be named  <TrackName>_<CarName>_<n>.csv
# Results: torcs_model.pt   and   preproc.pkl
# -----------------------------------------------------------------------------


import os, glob, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import expand_lists, build_scaler, scale_row, \
                  encode_categories, ALL_NUM, TARGETS, save_preproc


# ─────────── hyper-parameters ──────────────────────────────────────────
MODEL_FILE   = "torcs_model.pt"
PREPROC_FILE = "preproc.pkl"
EPOCHS       = 15
BATCH        = 1024
LR           = 3e-4
# ----------------------------------------------------------------------


# ══════════ 1. Load & merge every valid CSV ════════════════════════════
dataframes = []
for fn in glob.glob("*.csv"):
    base = os.path.basename(fn).replace(".csv", "")
    try:
        track_name, car_name, _ = base.split("_", maxsplit=2)
    except ValueError:
        print(f"[WARN] Skipping file without Track_Car prefix: {fn}")
        continue

    try:
        df = pd.read_csv(fn)
        df = expand_lists(df).dropna(subset=ALL_NUM + TARGETS)
        df["track_name"] = track_name
        df["car_name"] = car_name
        dataframes.append(df)
        print(f"✓ Loaded {len(df):6} rows from {fn}")
    except Exception as e:
        print(f"[ERROR] Could not read {fn}: {e}")

if not dataframes:
    raise RuntimeError("No training CSVs found!")

df = pd.concat(dataframes, ignore_index=True)
print(f"\nTOTAL samples: {len(df)}\n")

# ══════════ 2. Pre-process ════════════════════════════════════════════
cat_idx, cat_maps = encode_categories(df)
scaler_stats      = build_scaler(df)

X_num = np.stack([scale_row(r, scaler_stats) for _, r in df.iterrows()])
y     = df[TARGETS].astype(np.float32).to_numpy()

Xn_tr, Xn_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
    X_num, cat_idx, y, test_size=0.15, random_state=42
)

train_ds = TensorDataset(torch.tensor(Xn_tr),
                         torch.tensor(Xc_tr),
                         torch.tensor(y_tr))
val_ds   = TensorDataset(torch.tensor(Xn_val),
                         torch.tensor(Xc_val),
                         torch.tensor(y_val))

train_dl = DataLoader(train_ds, BATCH, shuffle=True)
val_dl   = DataLoader(val_ds,   BATCH)

# ══════════ 3. Define the network ═════════════════════════════════════
EMB_SZ = {k: max(m.values()) + 1 for k, m in cat_maps.items()}

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_track = nn.Embedding(EMB_SZ["track_name"], 8)
        self.emb_car   = nn.Embedding(EMB_SZ["car_name"],   4)
        self.net = nn.Sequential(
            nn.Linear(len(ALL_NUM) + 12, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),   nn.Tanh()    # steer ∈ [-1,1]; accel/brake clamped later
        )

    def forward(self, xn, xc):
        z = torch.cat([xn,
                       self.emb_track(xc[:, 0]),
                       self.emb_car(  xc[:, 1])], dim=1)
        out = self.net(z)
        steer = out[:, 0]
        accel = torch.clamp(out[:, 1], 0, 1)
        brake = torch.clamp(out[:, 2], 0, 1)
        return torch.stack([steer, accel, brake], 1)


model = MLP()
optimiser = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ══════════ 4. Training loop ══════════════════════════════════════════
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb_num, xb_cat, yb in train_dl:
        pred = model(xb_num, xb_cat)
        loss = criterion(pred, yb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(xn, xc), yb).item()
                       for xn, xc, yb in val_dl) / len(val_dl)
    print(f"Epoch {epoch:02}/{EPOCHS} | val MSE {val_loss:.6f}")

# ══════════ 5. Save artefacts ═════════════════════════════════════════
torch.save(model.state_dict(), MODEL_FILE)
save_preproc(scaler_stats, cat_maps, PREPROC_FILE)
print(f"\nSaved  ➜  {MODEL_FILE}, {PREPROC_FILE}")
