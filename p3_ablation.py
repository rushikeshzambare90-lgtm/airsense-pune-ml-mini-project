import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from common import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
SEED = 42
d, _, _ = load_clean(); F = build_frame(d)
R = pd.read_pickle("regimes.pkl")[["station", "date", "regime"]]
F = F.merge(R, on=["station", "date"], how="left"); F["regime"] = F["regime"].fillna(-1).astype(int).astype(str)
def prep(extra):
    return ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT + list(extra))])
def split(tgt):
    r = F[F[tgt].notna() & F.pm25.notna()].sort_values(["date", "station"]).reset_index(drop=True)
    return r[r.date < CUTOFF], r[r.date >= CUTOFF]
tscv = TimeSeriesSplit(5); out = {}
tr, te = split("y_pm25")
mods = {"Ridge": lambda: RidgeCV(alphas=np.logspace(-2, 3, 30), cv=tscv),
        "Random Forest": lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED)}
for nm, mk in mods.items():
    row = {}
    for tag, extra in [("without", []), ("with", ["regime"])]:
        m = Pipeline([("p", prep(extra)), ("m", mk())]).fit(tr, tr.y_pm25); p = m.predict(te)
        row[tag] = dict(rmse=float(np.sqrt(mean_squared_error(te.y_pm25, p))), mae=float(mean_absolute_error(te.y_pm25, p)), r2=float(r2_score(te.y_pm25, p)))
    out["reg_" + nm] = row; print(nm, row, flush=True)
trc, tec = split("y_cls")
cm = {"Random Forest": lambda: RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
      "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED, eval_metric="mlogloss")}
for nm, mk in cm.items():
    row = {}
    for tag, extra in [("without", []), ("with", ["regime"])]:
        m = Pipeline([("p", prep(extra)), ("m", mk())]).fit(trc, trc.y_cls.astype(int)); p = m.predict(tec)
        row[tag] = dict(acc=float(accuracy_score(tec.y_cls.astype(int), p)), f1=float(f1_score(tec.y_cls.astype(int), p, average="macro", zero_division=0)))
    out["cls_" + nm] = row; print(nm, row, flush=True)
json.dump(out, open("res_p3.json", "w"), indent=1)
