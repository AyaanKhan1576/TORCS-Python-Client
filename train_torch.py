# # train_torch.py ------------------------------------------------------
# """
# Train an MLP on every CSV in the folder whose name is <Track>_<Car>_<n>.csv
# Produces  torcs_model.pt  and  preproc.pkl
# """
# import os, glob, numpy as np, pandas as pd, torch
# import torch.nn as nn, torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
# from utils import expand_lists, build_scaler, scale_row, \
#                   encode_categories, ALL_NUM, TARGETS, save_preproc
# from models import build_mlp

# MODEL_FILE   = "torcs_model.pt"
# PREPROC_FILE = "preproc.pkl"
# EPOCHS = 15
# BATCH  = 1024
# LR     = 3e-4

# def main():
#     dfs = []
#     for fn in glob.glob("*.csv"):
#         base = os.path.splitext(fn)[0]
#         try:
#             track, car, _ = base.split("_", 2)
#         except ValueError:
#             print(f"[skip] {fn}: bad filename")
#             continue

#         df = pd.read_csv(fn)
#         df = expand_lists(df)
#         df["track_name"] = df.get("track_name","unknown").replace("unknown",track)
#         df["car_name"]   = df.get("car_name","unknown").replace("unknown",car)

#         df[ALL_NUM+TARGETS] = df[ALL_NUM+TARGETS].apply(
#             pd.to_numeric, errors="coerce"
#         )
#         df = df.dropna(subset=TARGETS)
#         if df.empty:
#             print(f"[WARN] {fn}: no usable rows")
#             continue
#         df[ALL_NUM] = df[ALL_NUM].fillna(0.0)
#         dfs.append(df)
#         print(f"✓ {fn:30} rows kept {len(df)}")

#     if not dfs:
#         raise RuntimeError("No valid training rows found!")

#     df = pd.concat(dfs, ignore_index=True)
#     print(f"\nTOTAL rows used: {len(df)}")

#     cat_idx, cat_maps = encode_categories(df)
#     scaler_stats      = build_scaler(df)
#     X_num = np.stack([scale_row(r, scaler_stats) for _, r in df.iterrows()])
#     y     = df[TARGETS].to_numpy(dtype=np.float32)

#     Xn_tr,Xn_val,Xc_tr,Xc_val,y_tr,y_val = train_test_split(
#         X_num, cat_idx, y, test_size=0.15, random_state=42
#     )
#     tr_ds = TensorDataset(torch.tensor(Xn_tr), torch.tensor(Xc_tr), torch.tensor(y_tr))
#     va_ds = TensorDataset(torch.tensor(Xn_val), torch.tensor(Xc_val), torch.tensor(y_val))
#     tr_dl = DataLoader(tr_ds, BATCH, shuffle=True)
#     va_dl = DataLoader(va_ds, BATCH)

#     emb_sz = {k: max(m.values())+1 for k,m in cat_maps.items()}
#     model  = build_mlp(len(ALL_NUM), emb_sz)
#     opt    = optim.Adam(model.parameters(), LR)
#     mse    = nn.MSELoss()

#     for epoch in range(1, EPOCHS+1):
#         model.train()
#         for xn, xc, yb in tr_dl:
#             loss = mse(model(xn, xc), yb)
#             opt.zero_grad(); loss.backward(); opt.step()
#         model.eval()
#         with torch.no_grad():
#             val = sum(mse(model(xn, xc), yb).item()
#                       for xn, xc, yb in va_dl)/len(va_dl)
#         print(f"Epoch {epoch:02}/{EPOCHS}  val MSE {val:.6f}")

#     torch.save(model.state_dict(), MODEL_FILE)
#     save_preproc(scaler_stats, cat_maps, PREPROC_FILE)
#     print(f"\nSaved ➜ {MODEL_FILE}, {PREPROC_FILE}")

# if __name__ == "__main__":
#     main()

"""
train_torch.py
──────────────
Train a track- and car-aware MLP on every driving CSV in the folder.

CSV filenames must follow   <TrackName>_<CarName>_<N>.csv
Outputs:
    • torcs_model.pt   – state_dict
    • preproc.pkl      – scaler statistics + category maps
"""
from __future__ import annotations
import os, glob, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from utils   import expand_lists, build_scaler, scale_row, \
                     encode_categories, ALL_NUM, TARGETS, save_preproc
from models  import build_mlp

# ───────── hyper-parameters ───────────────────────────────────────────
EPOCHS        = 15
BATCH         = 1024
LR            = 3e-4
MODEL_FILE    = "torcs_model.pt"
PREPROC_FILE  = "preproc.pkl"

# weighted loss knobs
BRAKE_WEIGHT  = 2.5
TARGET_SPEED  = 25.0      # m/s   (~90 km/h)   reward for being too slow
λ_SPEED       = 0.2

# ───────── 1. Load & clean every CSV ──────────────────────────────────
dfs: list[pd.DataFrame] = []

for fn in glob.glob("*.csv"):
    base = os.path.splitext(fn)[0]
    try:
        track, car, _ = base.split("_", 2)
    except ValueError:
        print(f"[skip] {fn}: filename not Track_Car_N")
        continue

    try:
        df = pd.read_csv(fn)
        df = expand_lists(df)

        # replace 'unknown' with filename info
        df["track_name"] = df.get("track_name", "unknown").replace("unknown", track)
        df["car_name"]   = df.get("car_name",   "unknown").replace("unknown", car)

        # convert all numeric cols → float; bad tokens → NaN
        df[ALL_NUM + TARGETS] = df[ALL_NUM + TARGETS].apply(
            pd.to_numeric, errors="coerce"
        )

        # keep rows that contain all three pedal targets
        before = len(df)
        df = df.dropna(subset=TARGETS)
        if df.empty:
            print(f"[WARN] {fn}: 0 rows after cleaning")
            continue

        # fill remaining NaNs in sensor cols with 0
        df[ALL_NUM] = df[ALL_NUM].fillna(0.0)
        dfs.append(df)
        print(f"✓ {fn:30} rows kept {len(df):6} / {before}")
    except Exception as e:
        print(f"[ERROR] {fn}: {e}")

if not dfs:
    raise RuntimeError("No valid training rows found!")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTOTAL rows used: {len(df)}")

# ───────── 2. Pre-process ─────────────────────────────────────────────
cat_idx, cat_maps     = encode_categories(df)          # categorical IDs
scaler_stats          = build_scaler(df)               # z-score stats
X_num = np.stack([scale_row(r, scaler_stats) for _, r in df.iterrows()])
y      = df[TARGETS].astype(np.float32).to_numpy()
speed_arr = df["speedX"].to_numpy(dtype=np.float32)    # ground truth speed

Xn_tr,Xn_val,Xc_tr,Xc_val,y_tr,y_val,spd_tr,spd_val = train_test_split(
    X_num, cat_idx, y, speed_arr, test_size=0.15, random_state=42
)

train_ds = TensorDataset(torch.tensor(Xn_tr),
                         torch.tensor(Xc_tr),
                         torch.tensor(y_tr),
                         torch.tensor(spd_tr))
val_ds   = TensorDataset(torch.tensor(Xn_val),
                         torch.tensor(Xc_val),
                         torch.tensor(y_val),
                         torch.tensor(spd_val))

train_dl = DataLoader(train_ds, BATCH, shuffle=True)
val_dl   = DataLoader(val_ds,   BATCH)

# ───────── 3. Build network ───────────────────────────────────────────
emb_sizes = {k: max(m.values())+1 for k,m in cat_maps.items()}
model      = build_mlp(len(ALL_NUM), emb_sizes)
optimiser  = optim.Adam(model.parameters(), lr=LR)

# weighted loss function
def weighted_loss(pred, target, true_speed):
    """
    pred, target: (B,3)  [steer, accel, brake]
    true_speed:   (B,)
    """
    err = F.mse_loss(pred, target, reduction='none')     # (B,3)

    w = torch.stack([
        1.0 + target[:,0].abs(),            # corners>straights
        torch.ones_like(target[:,1]),
        BRAKE_WEIGHT*torch.ones_like(target[:,2])
    ], dim=1)
    mse = (err * w).mean()

    # speed bonus: penalise being BELOW target speed
    pred_speed = X_batch_speed  # filled in loop
    speed_pen  = λ_SPEED * torch.clamp(TARGET_SPEED - pred_speed, min=0).mean()
    return mse + speed_pen

# ───────── 4. Training loop ───────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    model.train()
    for xn, xc, yb, xb_speed in train_dl:
        global X_batch_speed
        X_batch_speed = xb_speed             # for speed penalty inside loss
        loss = weighted_loss(model(xn, xc), yb, xb_speed)
        optimiser.zero_grad(); loss.backward(); optimiser.step()

    # validation
    model.eval()
    with torch.no_grad():
        val = 0.0
        for xn, xc, yb, xb_speed in val_dl:
            X_batch_speed = xb_speed
            val += weighted_loss(model(xn, xc), yb, xb_speed).item()
        val /= len(val_dl)
    print(f"Epoch {epoch:02}/{EPOCHS}  val loss {val:.6f}")

# ───────── 5. Save artefacts ──────────────────────────────────────────
torch.save(model.state_dict(), MODEL_FILE)
save_preproc(scaler_stats, cat_maps, PREPROC_FILE)
print(f"\nSaved ➜ {MODEL_FILE}, {PREPROC_FILE}")
