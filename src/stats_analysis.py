import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

conn = sqlite3.connect("data/processed/turbofan.db")
df = pd.read_sql_query("SELECT * FROM train_fd001_rul", conn)
conn.close()

sensors = [c for c in df.columns if c.startswith("s_")]
sensors = [c for c in sensors if df[c].nunique() > 2]

early = df[df["rul"] > 100]
late = df[df["rul"] <= 30]
print(f"Early-life rows: {len(early)}, late-life rows: {len(late)}")

rows = []
for col in sensors:
    rho, _ = spearmanr(df[col], df["rul"])
    pooled_sd = np.sqrt((early[col].var() + late[col].var()) / 2)
    d = (late[col].mean() - early[col].mean()) / pooled_sd
    _, p = mannwhitneyu(early[col], late[col])
    rows.append({
        "sensor": col,
        "spearman_vs_rul": rho,
        "effect_size_d": d,
        "p_value": p,
    })

results = pd.DataFrame(rows)
results["abs_d"] = results["effect_size_d"].abs()
results = results.sort_values("abs_d", ascending=False).drop(columns="abs_d").round(4)
print(results.to_string(index=False))

results.to_csv("data/processed/sensor_stats.csv", index=False)