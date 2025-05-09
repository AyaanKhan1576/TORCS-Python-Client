# # utils.py  ───────────────────────────────────────────────────────────
# # Shared helpers: feature scaling, list-sensor flattening, category maps
# import pickle, numpy as np, pandas as pd
# from collections import defaultdict

# # ── scalar sensors ───────────────────────────────────────────────────
# BASE_NUM = [
#     "speedX","speedY","speedZ",
#     "angle","trackPos","rpm","gear",
#     "distRaced","damage"
# ]

# # ── list-type sensors and their lengths ──────────────────────────────
# LIST_SENSORS = {
#     "track":        19,
#     "focus":         5,
#     "wheelSpinVel":  4,
#     "opponents":    36
# }

# ALL_NUM = BASE_NUM + [f"{s}{i}" for s,n in LIST_SENSORS.items() for i in range(n)]
# TARGETS = ["steering","accel","brake"]
# EMBEDS  = ["track_name","car_name"]

# # ── robust string→list(float) parser ─────────────────────────────────
# def _to_list(val):
#     """
#     '[1 2 3]' → [1.0,2.0,3.0] ; non-numeric tokens → np.nan
#     """
#     if isinstance(val, list):
#         return val
#     if isinstance(val, str):
#         stripped = val.replace("[","").replace("]","").replace(",", " ")
#         out=[]
#         for tok in stripped.split():
#             try:
#                 out.append(float(tok))
#             except ValueError:
#                 out.append(np.nan)
#         return out
#     return []

# # ── flatten list-sensor columns, create NaNs for truly missing cols ──
# def expand_lists(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Ensures EVERY column named in ALL_NUM exists in df (filled with NaN
#     where data is unavailable).  Handles CSVs that either
#       • contain a single list column  (e.g. 'focus'),
#       • already contain flat columns (focus0..focus4),
#       • or are missing the sensor entirely.
#     """
#     for col, length in LIST_SENSORS.items():

#         # 1️⃣  list column present → expand & drop original
#         if col in df.columns:
#             df[col] = df[col].apply(_to_list)
#             for i in range(length):
#                 df[f"{col}{i}"] = df[col].apply(
#                     lambda lst: lst[i] if i < len(lst) else np.nan
#                 )
#             df = df.drop(columns=[col])
#             continue

#         # 2️⃣  already flattened → verify; else fall through
#         if all(f"{col}{i}" in df.columns for i in range(length)):
#             continue

#         # 3️⃣  sensor completely missing → fabricate NaN columns
#         for i in range(length):
#             df[f"{col}{i}"] = np.nan

#     return df

# # ── scaler / encoder helpers (unchanged) ─────────────────────────────
# def build_scaler(df):
#     return {c: (df[c].mean(), df[c].std() + 1e-8) for c in ALL_NUM}

# def scale_row(row, stats):
#     return np.array([(row[c] - stats[c][0]) / stats[c][1] for c in ALL_NUM],
#                     dtype=np.float32)

# def encode_categories(df, maps=None):
#     if maps is None:
#         maps = {k: defaultdict(lambda: len(m)) for k, m in [(x, {}) for x in EMBEDS]}
#     idx = {k: df[k].apply(lambda v: maps[k][v]).astype(np.int64) for k in EMBEDS}
#     return np.stack([idx[k] for k in EMBEDS], 1), maps

# # def save_preproc(stats, maps, fn="preproc.pkl"):
# #     with open(fn, "wb") as f:
# #         pickle.dump({"stats": stats, "maps": maps}, f)

# def save_preproc(stats, maps, fn="preproc.pkl"):
#     """
#     Convert each defaultdict -> normal dict so pickle works.
#     """
#     serial_maps = {k: dict(v) for k, v in maps.items()}
#     with open(fn, "wb") as f:
#         pickle.dump({"stats": stats, "maps": serial_maps}, f)


# # def load_preproc(fn="preproc.pkl"):
# #     with open(fn, "rb") as f:
# #         d = pickle.load(f)
# #     return d["stats"], d["maps"]

# def load_preproc(fn="preproc.pkl"):
#     """
#     Return stats and plain-dict maps; driver will use .get(key, fallback)
#     so defaultdict is no longer required.
#     """
#     with open(fn, "rb") as f:
#         d = pickle.load(f)
#     return d["stats"], d["maps"]

# utils.py ------------------------------------------------------------
import pickle, numpy as np, pandas as pd
from collections import defaultdict

# scalar sensors
BASE_NUM = [
    "speedX","speedY","speedZ",
    "angle","trackPos","rpm","gear",
    "distRaced","damage"
]

# list sensors
LIST_SENSORS = {
    "track":        19,
    "focus":         5,
    "wheelSpinVel":  4,
    "opponents":    36
}

ALL_NUM = BASE_NUM + [f"{s}{i}" for s,n in LIST_SENSORS.items() for i in range(n)]
TARGETS = ["steering","accel","brake"]
EMBEDS  = ["track_name","car_name"]

# ---------- helpers ---------------------------------------------------
def _to_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.replace("[","").replace("]","").replace(","," ")
        out=[]
        for tok in s.split():
            try: out.append(float(tok))
            except ValueError: out.append(np.nan)
        return out
    return []

def expand_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee every <sensor><idx> column exists.
    Accepts list-style or pre-flattened CSVs.
    """
    for col, length in LIST_SENSORS.items():
        # list column present → expand
        if col in df.columns:
            df[col] = df[col].apply(_to_list)
            for i in range(length):
                df[f"{col}{i}"] = df[col].apply(lambda l: l[i] if i < len(l) else np.nan)
            df = df.drop(columns=[col])
            continue
        # already flattened?
        if all(f"{col}{i}" in df.columns for i in range(length)):
            continue
        # otherwise fabricate NaNs
        for i in range(length):
            df[f"{col}{i}"] = np.nan
    return df

def build_scaler(df):
    return {c: (df[c].mean(), df[c].std() + 1e-8) for c in ALL_NUM}

def scale_row(row, stats):
    return np.array([(row[c] - stats[c][0]) / stats[c][1] for c in ALL_NUM],
                    dtype=np.float32)

def encode_categories(df, maps=None):
    if maps is None:
        maps = {k: defaultdict(lambda: len(m)) for k, m in [(x,{}) for x in EMBEDS]}
    idx = {k: df[k].apply(lambda v: maps[k][v]).astype(np.int64) for k in EMBEDS}
    return np.stack([idx[k] for k in EMBEDS], 1), maps

def save_preproc(stats, maps, fn="preproc.pkl"):
    serial_maps = {k: dict(v) for k,v in maps.items()}
    with open(fn, "wb") as f:
        pickle.dump({"stats": stats, "maps": serial_maps}, f)

def load_preproc(fn="preproc.pkl"):
    with open(fn, "rb") as f:
        d = pickle.load(f)
    return d["stats"], d["maps"]
