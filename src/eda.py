import sqlite3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("data/processed/turbofan.db")
df = pd.read_sql_query("SELECT * FROM train_fd001_rul", conn)
conn.close()

sensors = [c for c in df.columns if c.startswith("s_")]

print(df[sensors].nunique().to_string())

flat = [c for c in sensors if df[c].nunique() <= 2]
useful = [c for c in sensors if c not in flat]
print("\nConstant or near-constant sensors:", flat)
print("Sensors kept:", useful)

fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True)
for ax, col in zip(axes.flat, useful):
    for unit in range(1, 6):
        engine = df[df["unit"] == unit]
        ax.plot(engine["rul"], engine[col], color="#2a78d6", alpha=0.5, linewidth=1)
    ax.set_title(col, fontsize=10, color="#52514e")
for ax in list(axes.flat)[len(useful):]:
    ax.set_visible(False)

axes.flat[0].invert_xaxis()
fig.supxlabel("Cycles remaining until failure (time runs left to right)")
fig.tight_layout()

Path("charts").mkdir(exist_ok=True)
fig.savefig("charts/sensor_degradation.png", dpi=200, bbox_inches="tight")
plt.show()