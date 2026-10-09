import sqlite3
from pathlib import Path
import pandas as pd
from load_data import load_split, load_rul

DB_PATH = Path("data/processed/turbofan.db")
SUBSET = "FD001"


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    train = load_split("train", SUBSET)
    test = load_split("test", SUBSET)
    rul = load_rul(SUBSET)
    rul.insert(0, "unit", range(1, len(rul) + 1))

    name = SUBSET.lower()
    train.to_sql(f"train_{name}", conn, if_exists="replace", index=False)
    test.to_sql(f"test_{name}", conn, if_exists="replace", index=False)
    rul.to_sql(f"rul_{name}", conn, if_exists="replace", index=False)

    conn.execute(f"DROP TABLE IF EXISTS train_{name}_rul")
    conn.execute(f"""
        CREATE TABLE train_{name}_rul AS
        SELECT *, MAX(cycle) OVER (PARTITION BY unit) - cycle AS rul
        FROM train_{name}
    """)
    conn.commit()

    check = pd.read_sql_query(f"""
        SELECT unit, cycle, rul
        FROM train_{name}_rul
        WHERE unit = 1
        ORDER BY cycle DESC
        LIMIT 5
    """, conn)
    print(check)

    lifespans = pd.read_sql_query(f"""
        SELECT MIN(life) AS shortest, ROUND(AVG(life)) AS average, MAX(life) AS longest
        FROM (SELECT unit, MAX(cycle) AS life FROM train_{name} GROUP BY unit)
    """, conn)
    print(lifespans)

    conn.close()


if __name__ == "__main__":
    main()