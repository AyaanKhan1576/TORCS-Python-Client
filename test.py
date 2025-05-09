import pandas as pd
bad = pd.read_csv("G-Speedway_Peugeot-406_01.csv")
bad = bad.replace("unknown", pd.NA)
missing = bad[ALL_NUM + TARGETS].isna().all()
print("Entire columns that are NaN:", missing[missing].index.tolist())
