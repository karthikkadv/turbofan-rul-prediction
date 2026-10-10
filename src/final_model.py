import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DB_PATH = "data/processed/turbofan.db"
RUL_CAP = 125
D_THRESHOLD = 2


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


stats = pd.read_csv("data/processed/sensor_stats.csv", index_col=0)
features = stats[stats["effect_size_d"].abs() >= D_THRESHOLD].index.tolist()

con = sqlite3.connect(DB_PATH)
train = pd.read_sql_query("SELECT * FROM train_fd001_rul", con)
test = pd.read_sql_query("SELECT * FROM test_fd001", con)
true_rul = pd.read_sql_query("SELECT * FROM rul_fd001", con)
con.close()

X_train = train[features]
y_train = train["rul"].clip(upper=RUL_CAP)

last = test.sort_values("cycle").groupby("unit").tail(1)
last = last.merge(true_rul, on="unit").sort_values("unit")
X_test, y_test = last[features], last["rul"]

# models
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(
    n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
).fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
dummy_rmse = rmse(y_test, np.full(len(y_test), y_train.mean()))

results = {
    "Guess the mean": dummy_rmse,
    "Linear regression": rmse(y_test, lr_pred),
    "Random forest": rmse(y_test, rf_pred),
}
for name, score in results.items():
    print(f"{name}: RMSE {score:.1f}")

# chart 1: model comparison
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(results.keys(), results.values(), color=["#898781", "#eb6834", "#2a78d6"])
ax.set_ylabel("RMSE (cycles, lower is better)")
ax.set_title("Test-set error by model")
for i, v in enumerate(results.values()):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center")
fig.tight_layout()
fig.savefig("charts/model_comparison.png", dpi=150)

# chart 2: which sensors the forest relied on
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(imp.index, imp.values, color="#2a78d6")
ax.set_xlabel("Importance")
ax.set_title("Random forest feature importance")
fig.tight_layout()
fig.savefig("charts/feature_importance.png", dpi=150)
print(imp.sort_values(ascending=False).head(5))

# exports for Tableau
out = pd.DataFrame({
    "unit": last["unit"].values,
    "true_rul": y_test.values,
    "linear_pred": lr_pred.round(1),
    "forest_pred": rf_pred.round(1),
})
out["forest_error"] = out["forest_pred"] - out["true_rul"]
out["linear_error"] = out["linear_pred"] - out["true_rul"]
out.to_csv("data/processed/predictions.csv", index=False)

trends = train[["unit", "cycle", "rul"] + features]
trends.to_csv("data/processed/sensor_trends.csv", index=False)
print("Saved predictions.csv and sensor_trends.csv")