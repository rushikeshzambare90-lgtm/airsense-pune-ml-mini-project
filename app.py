"""AirSense Pune dashboard.  Run:  streamlit run app.py   (after running p1..p4 once)"""
import numpy as np, pandas as pd, joblib, streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from common import *

ACTIONS = ["Normal activity", "Limit outdoor exercise", "Wear a mask outdoors", "Stay indoors"]
CLASSES = ["Good / Satisfactory", "Moderate", "Poor", "Very Poor / Severe"]
REGIMES = ["Very clean", "Clean", "Moderate", "Elevated", "High (smog)", "Dust-heavy"]

@st.cache_resource
def load():
    d, _, _ = load_clean(); F = build_frame(d)
    def pipe(m):
        pre = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
                                 ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
        return Pipeline([("p", pre), ("m", m)])
    r = F[F.y_pm25.notna() & F.pm25.notna()]; c = F[F.y_cls.notna() & F.pm25.notna()]
    reg = pipe(RandomForestRegressor(300, min_samples_leaf=3, random_state=42, n_jobs=-1)).fit(r, r.y_pm25)
    clf = pipe(RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)).fit(c, c.y_cls.astype(int))
    try: Q = joblib.load("q.joblib")["Q"]
    except Exception: Q = None
    try: R = pd.read_pickle("regimes.pkl")
    except Exception: R = None
    return F, reg, clf, Q, R

st.title("AirSense Pune")
st.caption("Next-day PM2.5 forecast, AQI class, pollution regime and health advice (demo; not medical advice)")
F, reg, clf, Q, R = load()
stations = sorted(F.station.unique())
station = st.selectbox("Station", stations, index=stations.index("Katraj"))
dates = F[(F.station == station) & F.pm25.notna()].date.dt.date
day = st.date_input("Date (today's readings)", value=dates.max(), min_value=dates.min(), max_value=dates.max())
row = F[(F.station == station) & (F.date.dt.date == day)]
if row.empty or pd.isna(row.pm25.iloc[0]):
    st.warning("No PM2.5 reading for this station and date."); st.stop()
pm = float(reg.predict(row)[0]); cls = int(clf.predict(row)[0])
today_cls = int(row.aqi_today_cls.fillna(0).iloc[0])
c1, c2, c3 = st.columns(3)
c1.metric("Today's PM2.5", f"{row.pm25.iloc[0]:.0f} µg/m³"); c2.metric("Forecast PM2.5 (tomorrow)", f"{pm:.0f} µg/m³"); c3.metric("Forecast AQI class", CLASSES[cls])
if R is not None:
    m = R[(R.station == station) & (R.date.dt.date == day)]
    st.write("**HMM regime:** " + (REGIMES[int(m.regime.iloc[0])] if len(m) else "not available for this day (needs PM2.5 and PM10)"))
if Q is not None:
    state = cls * 3 + int(np.sign(cls - today_cls) + 1); a = int(Q[state].argmax())
    st.success(f"**Advice:** {ACTIONS[a]}")
st.caption("Models are Random Forests trained on all available days; advice comes from the Q-Learning table learned in p2_hmm_cluster_rl.py.")
