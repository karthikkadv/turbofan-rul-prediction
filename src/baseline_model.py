import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DB_PATH = "data/processed/turbofan.db"
RUL_CAP = 125
D_THRESHOLD = 2


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# 1. choose features from the stats table
stats = pd.read_csv("data/processed/sensor_stats.csv", index_col=0)
features = stats[stats["effect_size_d"].abs() >= D_THRESHOLD].index.tolist()
print(f"Using {len(features)} sensors: {features}")

# 2. training data, with the capped target
con = sqlite3.connect(DB_PATH)
train = pd.read_sql_query("SELECT * FROM train_fd001_rul", con)
X_train = train[features]
y_train = train["rul"].clip(upper=RUL_CAP)

# 3. test data: last recorded cycle of each engine plus its true RUL
test = pd.read_sql_query("SELECT * FROM test_fd001", con)
true_rul = pd.read_sql_query("SELECT * FROM rul_fd001", con)
con.close()

last = test.sort_values("cycle").groupby("unit").tail(1)
last = last.merge(true_rul, on="unit").sort_values("unit")
X_test = last[features]
y_test = last["rul"]

# 4. dummy baseline: always predict the training average
dummy_pred = np.full(len(y_test), y_train.mean())
print(f"Dummy RMSE (guess the mean): {rmse(y_test, dummy_pred):.1f}")

# 5. linear regression
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Linear regression RMSE: {rmse(y_test, pred):.1f}")