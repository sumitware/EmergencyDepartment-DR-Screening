import pandas as pd
import numpy as np
import json

df = pd.read_csv("data.csv")
print("shape:", df.shape)
print("columns:", list(df.columns))

y = df["class"].values
X_raw = df.drop(columns=["quality", "class"])  # exclude near constant image quality column

ma_cols = ["ma1", "ma2", "ma3", "ma4", "ma5", "ma6"]
exudate_cols = [c for c in X_raw.columns if c.startswith("exudate")]
print("exudate columns present:", exudate_cols, "(", len(exudate_cols), "of an expected 8 )")

base_cols = ["pre_screening"] + ma_cols + exudate_cols + \
            ["macula_opticdisc_distance", "opticdisc_diameter", "am_fm_classification"]
print("base feature count:", len(base_cols))

X = X_raw[base_cols].copy()

# five engineered features, adapted to the columns actually present in this file
def ma_slope(row):
    xs = np.arange(len(ma_cols))
    ys = row[ma_cols].values.astype(float)
    slope = np.polyfit(xs, ys, 1)[0]
    return slope

X["eng_ma_slope"] = X_raw.apply(ma_slope, axis=1)
X["eng_exudate_sum"] = X_raw[exudate_cols].sum(axis=1)
X["eng_exudate_ratio_loose_strict"] = X_raw[exudate_cols[0]] / (X_raw[exudate_cols[-1]] + 1e-6)
X["eng_distance_diameter_ratio"] = X_raw["macula_opticdisc_distance"] / (X_raw["opticdisc_diameter"] + 1e-6)
X["eng_ma_sum"] = X_raw[ma_cols].sum(axis=1)

engineered_cols = ["eng_ma_slope", "eng_exudate_sum", "eng_exudate_ratio_loose_strict",
                    "eng_distance_diameter_ratio", "eng_ma_sum"]

print("total feature count (base + engineered):", X.shape[1])
print("positive cases:", int(y.sum()), "negative cases:", int((1 - y).sum()))

X.to_csv("X_full.csv", index=False)
np.save("y.npy", y)

meta = {
    "n_records": int(len(df)),
    "n_positive": int(y.sum()),
    "n_negative": int((1 - y).sum()),
    "base_cols": base_cols,
    "engineered_cols": engineered_cols,
    "exudate_cols_present": exudate_cols,
    "n_exudate_cols_present": len(exudate_cols),
    "n_base_features": len(base_cols),
    "n_total_features": X.shape[1],
}
with open("meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
