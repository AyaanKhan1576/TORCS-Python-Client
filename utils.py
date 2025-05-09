# utils.py -------------------------------------------------------------
# Shared helpers: feature scaling, list-sensor flattening, category maps
import pickle, ast, numpy as np
from collections import defaultdict

# ── scalar sensors ────────────────────────────────────────────────────
BASE_NUM = [
    "speedX","speedY","speedZ",
    "angle","trackPos","rpm","gear",
    "distRaced","damage"
]

# ── list-type sensors to flatten (name: length) ───────────────────────
LIST_SENSORS = {
    "track":        19,   # forward range-finders
    "focus":        5,    # look-ahead rays
    "wheelSpinVel": 4,    # wheel angular speed
    "opponents":    36    # 360° occupancy grid
}

ALL_NUM = BASE_NUM + [
    f"{s}{i}" for s,n in LIST_SENSORS.items() for i in range(n)
]

TARGETS = ["steering","accel","brake"]
EMBEDS  = ["track_name","car_name"]

# ── list string → list[float] ─────────────────────────────────────────
def _to_list(val):
    if isinstance(val, list): return val
    if isinstance(val, str):
        s = val.replace("[","").replace("]","").replace(","," ")
        return [float(x) for x in s.split() if x]
    return []

# ── expand LIST_SENSORS into scalar cols ──────────────────────────────
def expand_lists(df):
    for col, n in LIST_SENSORS.items():
        df[col] = df[col].apply(_to_list)
        for i in range(n):
            df[f"{col}{i}"] = df[col].apply(lambda l: l[i] if i < len(l) else 0.0)
    return df.drop(columns=list(LIST_SENSORS.keys()))

# ── z-score stats -----------------------------------------------------
def build_scaler(df):
    return {c: (df[c].mean(), df[c].std() + 1e-8) for c in ALL_NUM}

def scale_row(row, stats):
    return np.array([(row[c] - stats[c][0]) / stats[c][1] for c in ALL_NUM],
                    dtype=np.float32)

# ── categorical encoders ---------------------------------------------
def encode_categories(df, maps=None):
    if maps is None:
        maps = {k: defaultdict(lambda: len(m)) for k, m in [(x, {}) for x in EMBEDS]}
    idx = {k: df[k].apply(lambda v: maps[k][v]).astype(np.int64) for k in EMBEDS}
    return np.stack([idx[k] for k in EMBEDS], 1), maps

# ── save / load preprocessing artefacts -------------------------------
def save_preproc(stats, maps, fn="preproc.pkl"):
    with open(fn, "wb") as f:
        pickle.dump({"stats": stats, "maps": maps}, f)

def load_preproc(fn="preproc.pkl"):
    with open(fn, "rb") as f:
        d = pickle.load(f)
    return d["stats"], d["maps"]
