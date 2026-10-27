"""Shared data preparation for AirSense Pune."""
import numpy as np, pandas as pd

import os
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "all_stations_combined.csv")
CUTOFF = pd.Timestamp("2025-06-01")          # chronological split: train < cutoff <= test
POLL = ["pm25", "pm10", "no2", "so2", "co", "o3"]
# physically implausible / saturated sensor values -> treated as faulty readings
CAPS = {"pm25": 500, "pm10": 1000, "no2": 400, "so2": 200, "co": 20, "o3": 500}
SHORT = {"Karve Road Pune, Pune - MPCB": "Karve Road",
         "Katraj Dairy, Pune - MPCB": "Katraj",
         "Park Street Wakad, Pimpri Chinchwad - MPCB": "Wakad",
         "Revenue Colony-Shivajinagar, Pune - IITM": "Shivajinagar",
         "Savitribai Phule Pune University, Pune - MPCB": "SPPU"}

def season(m):
    return ("winter" if m in (12, 1, 2) else "summer" if m in (3, 4, 5)
            else "monsoon" if m in (6, 7, 8, 9) else "post_monsoon")

def aqi_class(a):
    if pd.isna(a): return np.nan
    return 0 if a <= 100 else 1 if a <= 200 else 2 if a <= 300 else 3

def load_clean():
    d = pd.read_csv(CSV, parse_dates=["date"])
    d["station"] = d.station_name.map(SHORT)
    n_raw = int(d[POLL].notna().sum().sum())
    removed = {}
    for c, cap in CAPS.items():
        bad = d[c] > cap
        removed[c] = int(bad.sum())
        d.loc[bad, c] = np.nan
    d = d.drop_duplicates(["station", "date"]).sort_values(["station", "date"])
    return d, n_raw, removed

def build_frame(d):
    """Daily grid per station, past-only features, next-day targets."""
    out = []
    for st, g in d.groupby("station"):
        g = g.set_index("date")[POLL + ["aqi"]]
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        raw = g.copy()
        f = g.ffill(limit=2)                          # past-only gap filling (<=2 days)
        x = pd.DataFrame(index=g.index)
        for c in POLL + ["aqi"]: x[c] = f[c]
        for k in (1, 2, 3, 7): x[f"pm25_l{k}"] = f["pm25"].shift(k)
        x["pm25_r3"] = f["pm25"].rolling(3, min_periods=1).mean()
        x["pm25_r7"] = f["pm25"].rolling(7, min_periods=1).mean()
        doy = x.index.dayofyear.values
        x["doy_sin"] = np.sin(2*np.pi*doy/365.25); x["doy_cos"] = np.cos(2*np.pi*doy/365.25)
        x["season"] = [season(m) for m in x.index.month]
        x["station"] = st
        x["aqi_today_cls"] = x["aqi"].map(aqi_class)
        x["y_pm25"] = raw["pm25"].shift(-1)           # next-day targets (raw values)
        x["y_aqi"] = raw["aqi"].shift(-1)
        x["y_cls"] = x["y_aqi"].map(aqi_class)
        x["date"] = x.index
        out.append(x.reset_index(drop=True))
    return pd.concat(out, ignore_index=True).sort_values(["date", "station"]).reset_index(drop=True)

NUM = POLL + ["aqi", "pm25_l1", "pm25_l2", "pm25_l3", "pm25_l7", "pm25_r3", "pm25_r7", "doy_sin", "doy_cos"]
CAT = ["station", "season"]
CORE = ["pm25", "pm10", "no2", "pm25_l1", "pm25_r3"]   # features used for polynomial expansion
