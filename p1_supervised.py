import json, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from common import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import (LinearRegression, RidgeCV, LassoCV, LogisticRegression, Perceptron)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,
    StackingRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier,
    StackingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
    precision_recall_fscore_support, roc_auc_score)
from xgboost import XGBRegressor, XGBClassifier

SEED = 42
d, n_raw, removed = load_clean()
F = build_frame(d)

def prep(extra_cat=()):
    num = NUM; cat = CAT + list(extra_cat)
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

def split(task, df=None):
    df = F if df is None else df
    tgt = "y_pm25" if task == "reg" else "y_cls"
    r = df[df[tgt].notna() & df["pm25"].notna()].sort_values(["date", "station"]).reset_index(drop=True)
    tr, te = r[r.date < CUTOFF], r[r.date >= CUTOFF]
    return tr, te, tgt

tscv = TimeSeriesSplit(n_splits=5)
res = {}

# ================= REGRESSION =================
tr, te, tgt = split("reg")
Xtr, ytr, Xte, yte = tr, tr[tgt].values, te, te[tgt].values
res["n"] = dict(reg_train=len(tr), reg_test=len(te), raw_values=n_raw, removed=removed)

def poly_pipe(final):
    ct = ColumnTransformer([
        ("poly", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                           ("pf", PolynomialFeatures(2, include_bias=False))]), CORE),
        ("rest", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         [c for c in NUM if c not in CORE]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("p", ct), ("m", final)])

def P(m, extra=()): return Pipeline([("p", prep(extra)), ("m", m)])

def reg_models(extra=()):
    base = [("rf", RandomForestRegressor(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(random_state=SEED)),
            ("xgb", XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED))]
    return {
     "Linear Regression": P(LinearRegression(), extra),
     "Polynomial Regression (deg 2)": poly_pipe(LinearRegression()),
     "Ridge Regression": P(RidgeCV(alphas=np.logspace(-2, 3, 30), cv=tscv), extra),
     "Lasso Regression": P(LassoCV(cv=tscv, random_state=SEED, max_iter=20000), extra),
     "Bagging (trees)": P(BaggingRegressor(n_estimators=100, random_state=SEED, n_jobs=-1), extra),
     "Random Forest": P(base[0][1], extra),
     "Gradient Boosting": P(base[1][1], extra),
     "AdaBoost": P(AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED), extra),
     "XGBoost": P(base[2][1], extra),
     "Stacking (RF+GB+XGB -> Ridge)": P(StackingRegressor(base, final_estimator=RidgeCV(), cv=5), extra),
     "MLP (ReLU)": P(TransformedTargetRegressor(MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=1500,
                       early_stopping=True, random_state=SEED), transformer=StandardScaler()), extra),
    }

def reg_metrics(y, p):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, p))), mae=float(mean_absolute_error(y, p)), r2=float(r2_score(y, p)))

reg = {}; reg_pred = {}
persist = te["pm25"].values
reg["Persistence baseline (tomorrow = today)"] = reg_metrics(yte, persist)
reg_pred["Persistence"] = persist
fitted_reg = {}
for name, m in reg_models().items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    reg[name] = reg_metrics(yte, p); reg_pred[name] = p; fitted_reg[name] = m
    print(f"{name:35s} {reg[name]}", flush=True)
res["regression"] = reg
# Lasso feature importance
lm = fitted_reg["Lasso Regression"]
names = lm.named_steps["p"].get_feature_names_out()
coef = pd.Series(lm.named_steps["m"].coef_, index=names)
res["lasso_top"] = {k: float(v) for k, v in coef[coef != 0].reindex(coef[coef != 0].abs().sort_values(ascending=False).index).head(8).items()}
res["lasso_alpha"] = float(lm.named_steps["m"].alpha_); res["lasso_nonzero"] = int((coef != 0).sum()); res["lasso_total"] = len(coef)
res["ridge_alpha"] = float(fitted_reg["Ridge Regression"].named_steps["m"].alpha_)
np.save("reg_test_true.npy", yte); joblib.dump(reg_pred, "reg_pred.joblib")

# ================= CLASSIFICATION =================
trc, tec, tgt = split("cls")
Xc, yc, Xt, yt = trc, trc[tgt].astype(int).values, tec, tec[tgt].astype(int).values
res["n"].update(cls_train=len(trc), cls_test=len(tec),
                cls_train_dist=np.bincount(yc, minlength=4).tolist(), cls_test_dist=np.bincount(yt, minlength=4).tolist())

def multi_auc(y, prob, classes=range(4)):
    aucs = []
    for k in classes:
        yy = (y == k).astype(int)
        if 0 < yy.sum() < len(yy): aucs.append(roc_auc_score(yy, prob[:, k]))
    return float(np.mean(aucs))

def cls_metrics(y, pred, prob):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(acc=float(accuracy_score(y, pred)), prec=float(p), rec=float(r), f1=float(f),
                auc=multi_auc(y, prob) if prob is not None else None)

def Pc(m, extra=()): return Pipeline([("p", prep(extra)), ("m", m)])

# tuned models
knn_grid = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 11, 15, 21, 31], "metric": ["euclidean", "manhattan"],
                        "weights": ["uniform", "distance"]}, cv=tscv, scoring="f1_macro", n_jobs=-1)
svm_lin = GridSearchCV(SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED), {"C": [0.01, 0.1, 1, 10]}, cv=tscv, scoring="f1_macro", n_jobs=-1)
svm_poly = GridSearchCV(SVC(kernel="poly", degree=3, probability=True, class_weight="balanced", random_state=SEED), {"C": [0.1, 1, 10]}, cv=tscv, scoring="f1_macro", n_jobs=-1)
svm_rbf = GridSearchCV(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}, cv=tscv, scoring="f1_macro", n_jobs=-1)

def cls_models(extra=()):
    base = [("rf", RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(random_state=SEED)),
            ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED, eval_metric="mlogloss"))]
    return {
     "Logistic Regression": Pc(LogisticRegression(max_iter=3000, class_weight="balanced"), extra),
     "LDA": Pc(LDA(), extra),
     "QDA (reg_param=0.5)": Pc(QDA(reg_param=0.5), extra),
     "k-NN": Pc(knn_grid, extra),
     "SVM (linear)": Pc(svm_lin, extra),
     "SVM (polynomial)": Pc(svm_poly, extra),
     "SVM (RBF)": Pc(svm_rbf, extra),
     "Perceptron": Pc(Perceptron(class_weight="balanced", random_state=SEED), extra),
     "Bagging (trees)": Pc(BaggingClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), extra),
     "Random Forest": Pc(base[0][1], extra),
     "Gradient Boosting": Pc(base[1][1], extra),
     "AdaBoost": Pc(AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED), extra),
     "XGBoost": Pc(base[2][1], extra),
     "Stacking (RF+GB+XGB -> LogReg)": Pc(StackingClassifier(base, final_estimator=LogisticRegression(max_iter=3000), cv=5), extra),
     "MLP (ReLU)": Pc(MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=1500, early_stopping=True, random_state=SEED), extra),
    }

cls = {}; cls_prob = {}; cls_pred = {}; fitted_cls = {}
for name, m in cls_models().items():
    m.fit(Xc, yc)
    pred = m.predict(Xt)
    prob = m.predict_proba(Xt) if hasattr(m, "predict_proba") else None
    cls[name] = cls_metrics(yt, pred, prob); cls_pred[name] = pred; cls_prob[name] = prob; fitted_cls[name] = m
    print(f"{name:35s} {cls[name]}", flush=True)
# best params
res["knn_best"] = fitted_cls["k-NN"].named_steps["m"].best_params_
res["svm_best"] = {k: fitted_cls[k].named_steps["m"].best_params_ for k in ["SVM (linear)", "SVM (polynomial)", "SVM (RBF)"]}
res["classification"] = cls

# MLP activation comparison
act = {}
for a in ["relu", "tanh", "logistic"]:
    m = Pc(MLPClassifier(hidden_layer_sizes=(64, 32), activation=a, max_iter=1500, early_stopping=True, random_state=SEED)); m.fit(Xc, yc)
    act[a] = cls_metrics(yt, m.predict(Xt), m.predict_proba(Xt))
res["mlp_activation"] = act
print("activation", act)

# Logistic regression binary: unhealthy tomorrow (class >=2)
yb_tr, yb_te = (yc >= 2).astype(int), (yt >= 2).astype(int)
lb = Pc(LogisticRegression(max_iter=3000, class_weight="balanced")).fit(Xc, yb_tr)
pb = lb.predict(Xt); prb = lb.predict_proba(Xt)[:, 1]
p, r, f, _ = precision_recall_fscore_support(yb_te, pb, average="binary", zero_division=0)
res["unhealthy_logreg"] = dict(acc=float(accuracy_score(yb_te, pb)), prec=float(p), rec=float(r), f1=float(f),
                                auc=float(roc_auc_score(yb_te, prb)), pos_test=int(yb_te.sum()), n_test=len(yb_te))
print("unhealthy", res["unhealthy_logreg"])
# ROC data for binary unhealthy flag (several models, via P(class>=2) = sum of prob cols 2,3)
roc = {}
from sklearn.metrics import roc_curve
roc["Logistic Regression"] = roc_curve(yb_te, prb)[:2]
for nm in ["k-NN", "SVM (RBF)", "Random Forest", "XGBoost", "Stacking (RF+GB+XGB -> LogReg)", "MLP (ReLU)"]:
    pr = cls_prob[nm][:, 2:].sum(1); roc[nm] = roc_curve(yb_te, pr)[:2]
    res.setdefault("unhealthy_auc", {})[nm] = float(roc_auc_score(yb_te, pr))
joblib.dump({k: (list(map(float, a)), list(map(float, b))) for k, (a, b) in roc.items()}, "roc.joblib")

# choose best classifier by time-series CV macro-F1 on TRAIN only (used by Q-learning)
from sklearn.model_selection import cross_val_score
cands = {"Random Forest": Pc(cls_models()["Random Forest"].named_steps["m"]), "XGBoost": Pc(cls_models()["XGBoost"].named_steps["m"]),
         "Gradient Boosting": Pc(cls_models()["Gradient Boosting"].named_steps["m"])}
cvs = {k: float(cross_val_score(m, Xc, yc, cv=tscv, scoring="f1_macro").mean()) for k, m in cands.items()}
res["cv_f1_candidates"] = cvs; best = max(cvs, key=cvs.get); res["best_cls_by_cv"] = best
print("cv", cvs, best)
joblib.dump(dict(Xc=Xc, yc=yc, Xt=Xt, yt=yt, best=best, cls_pred=cls_pred), "cls_data.joblib")
json.dump(res, open("res_p1.json", "w"), indent=1)
