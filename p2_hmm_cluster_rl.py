import json, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from common import *
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score, adjusted_rand_score,
                             homogeneity_score, completeness_score)
SEED = 42; np.random.seed(SEED)
d, n_raw, removed = load_clean(); F = build_frame(d)
res = json.load(open("res_p1.json"))

# ============ HMM (Unit IV) ============
OBS = ["pm25", "pm10"]
H = F[["station", "date", "season"] + OBS].copy()
H = H.mask(H[OBS].le(0).any(axis=1) , other=np.nan) if False else H
for _c in OBS: H.loc[H[_c] <= 0, _c] = np.nan   # zero readings = faulty/stuck sensor, not real air
for c in OBS: H[c] = np.log1p(H[c])
H = H[(H[OBS] > 0).all(axis=1)] if False else H
H = H.dropna(subset=OBS).sort_values(["station", "date"]).reset_index(drop=True)
# split into runs of consecutive days
H["run"] = ((H.station != H.station.shift()) | (H.date.diff().dt.days != 1)).cumsum()
runs = H.groupby("run").filter(lambda g: len(g) >= 5)
sc = StandardScaler().fit(runs[runs.date < CUTOFF][OBS])
runs = runs.copy(); runs[OBS] = sc.transform(runs[OBS])
tr_runs = runs[runs.date < CUTOFF]
Xtr = tr_runs[OBS].values; Ltr = tr_runs.groupby("run").size().values
res["hmm_n"] = dict(train_days=int(len(tr_runs)), runs=int(len(Ltr)), all_days=int(len(runs)))

def nparams(k, kind="ergodic"):
    dd = len(OBS)
    return (k - 1) + k * (k - 1) + k * dd + k * dd   # start + trans + means + diag vars
bic = {}; models = {}
for k in range(2, 9):
    best = None
    for s in range(6):
        m = GaussianHMM(k, covariance_type="diag", n_iter=300, random_state=SEED + s, tol=1e-4).fit(Xtr, Ltr)
        ll = m.score(Xtr, Ltr)
        if best is None or ll > best[0]: best = (ll, m)
    ll, m = best
    bic[k] = float(-2 * ll + nparams(k) * np.log(len(Xtr))); models[k] = m
res["hmm_bic"] = bic; K = min(bic, key=bic.get); res["hmm_K"] = int(K)
print("BIC", bic, "K", K, flush=True)
hmm = models[K]
# order states by mean PM2.5 (original scale of log1p)
order = np.argsort(hmm.means_[:, 0]); remap = {int(o): i for i, o in enumerate(order)}

# left-to-right HMM (same K) for comparison
def left_to_right(k):
    best = None
    for s in range(6):
        m = GaussianHMM(k, covariance_type="diag", n_iter=300, random_state=SEED + s, tol=1e-4, init_params="mc", params="stmc")
        st = np.zeros(k); st[0] = 1; tm = np.zeros((k, k))
        for i in range(k):
            tm[i, i] = 0.9
            if i < k - 1: tm[i, i + 1] = 0.1
            else: tm[i, i] = 1.0
        m.startprob_, m.transmat_ = st, tm
        m.fit(Xtr, Ltr); ll = m.score(Xtr, Ltr)
        if best is None or ll > best[0]: best = (ll, m)
    return best
llr, mlr = left_to_right(K)
res["hmm_lr"] = dict(loglik=float(llr), bic=float(-2 * llr + (K * 2 + K * 2 * len(OBS) + 1) * np.log(len(Xtr))))
res["hmm_ergodic_loglik"] = float(hmm.score(Xtr, Ltr))

# Viterbi decode (for timeline / interpretation) and past-only filtered state (used as feature)
def filtered_states(model, X):
    """Forward algorithm: P(state_t | obs_1..t) -- uses only past and present observations."""
    from scipy.special import logsumexp
    fl = model._compute_log_likelihood(X)
    with np.errstate(divide="ignore"):
        logA = np.log(model.transmat_); logpi = np.log(model.startprob_)
    alpha = np.zeros_like(fl); alpha[0] = logpi + fl[0]
    for t in range(1, len(X)):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + logA, axis=0) + fl[t]
    return alpha.argmax(1)
rows = []
for r, g in runs.groupby("run"):
    X = g[OBS].values
    vit = hmm.predict(X)                       # Viterbi
    fil = filtered_states(hmm, X)              # forward algorithm only (no future info)
    gg = g[["station", "date", "season"]].copy()
    gg["viterbi"] = [remap[int(s)] for s in vit]; gg["regime"] = [remap[int(s)] for s in fil]
    rows.append(gg)
R = pd.concat(rows)
R = R.merge(F[["station", "date", "pm25", "pm10", "no2"]], on=["station", "date"], how="left")
tab = []
for s in range(K):
    g = R[R.viterbi == s]
    tab.append(dict(state=s, days=int(len(g)), pm25=float(g.pm25.mean()), pm10=float(g.pm10.mean()), no2=float(g.no2.mean()),
                    season=g.season.value_counts().index[0], season_share=float(g.season.value_counts(normalize=True).iloc[0])))
res["hmm_states"] = tab
res["hmm_transmat"] = [[float(hmm.transmat_[order[i], order[j]]) for j in range(K)] for i in range(K)]
print(pd.DataFrame(tab)); print(np.round(res["hmm_transmat"], 2))
R.to_pickle("regimes.pkl")


# ============ CLUSTERING (Unit V) ============
C = F[["station", "date", "season", "pm25", "pm10", "no2", "aqi"]].dropna(subset=["pm25", "pm10", "no2", "aqi"]).reset_index(drop=True)
C["cls"] = C.aqi.map(aqi_class).astype(int)
Z = StandardScaler().fit_transform(np.log1p(C[["pm25", "pm10", "no2"]].values))
res["clu_n"] = int(len(C))
ks = range(2, 9); inertia = []; sil = []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z); inertia.append(float(km.inertia_)); sil.append(float(silhouette_score(Z, km.labels_)))
res["elbow"] = dict(k=list(ks), inertia=inertia, silhouette=sil)
KC = 4  # same k for all partitioning methods (matches the 4 AQI classes) -> fair comparison
res["clu_k"] = KC; res["clu_best_sil_k"] = int(list(ks)[int(np.argmax(sil))])

def kmedoids(Z, k, seed=SEED, iters=100):
    rng = np.random.RandomState(seed); n = len(Z)
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    best = None
    for _ in range(5):
        med = rng.choice(n, k, replace=False)
        for _ in range(iters):
            lab = D[:, med].argmin(1); new = med.copy()
            for j in range(k):
                idx = np.where(lab == j)[0]
                if len(idx): new[j] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
            if (new == med).all(): break
            med = new
        cost = D[np.arange(n), med[D[:, med].argmin(1)]].sum()
        if best is None or cost < best[0]: best = (cost, D[:, med].argmin(1))
    return best[1]

# DBSCAN: eps from k-distance plot heuristic, pick min_samples=10, eps maximising silhouette with noise <10%
nn = NearestNeighbors(n_neighbors=10).fit(Z); kd = np.sort(nn.kneighbors(Z)[0][:, -1])
best_db = None
for eps in np.linspace(0.2, 1.0, 17):
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Z)
    nc = len(set(lab) - {-1}); noise = (lab == -1).mean()
    if nc >= 2 and noise < 0.10:
        sc_ = silhouette_score(Z[lab != -1], lab[lab != -1])
        if best_db is None or sc_ > best_db[0]: best_db = (sc_, eps, lab)
if best_db is None:
    eps = float(np.quantile(kd, 0.95)); lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Z); best_db = (None, eps, lab)
db_eps, db_lab = best_db[1], best_db[2]
res["dbscan_eps"] = float(db_eps)

algos = {
 "K-Means": KMeans(KC, n_init=10, random_state=SEED).fit_predict(Z),
 "K-Medoids": kmedoids(Z, KC),
 "Hierarchical (Ward)": AgglomerativeClustering(KC, linkage="ward").fit_predict(Z),
 "Hierarchical (average)": AgglomerativeClustering(KC, linkage="average").fit_predict(Z),
 "DBSCAN": db_lab,
 "Spectral": SpectralClustering(KC, affinity="nearest_neighbors", n_neighbors=15, random_state=SEED).fit_predict(Z),
 "BIRCH": Birch(n_clusters=KC).fit_predict(Z),
}
clu = {}
for nm, lab in algos.items():
    m = lab != -1
    ncl = len(set(lab[m]))
    row = dict(clusters=int(ncl), noise=float((~m).mean()))
    if ncl >= 2:
        row.update(sil=float(silhouette_score(Z[m], lab[m])), db=float(davies_bouldin_score(Z[m], lab[m])))
    row.update(ari=float(adjusted_rand_score(C.cls.values, lab)), hom=float(homogeneity_score(C.cls.values, lab)),
               comp=float(completeness_score(C.cls.values, lab)))
    clu[nm] = row; print(nm, row, flush=True)
res["clustering"] = clu
# outliers
out = C[db_lab == -1]
res["dbscan_outliers"] = dict(n=int(len(out)), pct=float((db_lab == -1).mean()),
    mean_pm25=float(out.pm25.mean()), mean_pm10=float(out.pm10.mean()), by_station=out.station.value_counts().to_dict(),
    high_pm10_over_pm25=int((out.pm10 > 3 * out.pm25).sum()))
pca = PCA(2, random_state=SEED).fit(Z); P2 = pca.transform(Z)
joblib.dump(dict(P2=P2, km=algos["K-Means"], db=db_lab, ev=pca.explained_variance_ratio_.tolist()), "clu_plot.joblib")

# ============ Q-LEARNING (Unit VI) ============
import joblib as jl
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
dat = jl.load("cls_data.joblib"); Xc, yc, Xt, yt = dat["Xc"], dat["yc"], dat["Xt"], dat["yt"]
def mk():
    pre = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("p", pre), ("m", RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1))])
# out-of-fold forecasts on the training period (expanding window) -> state for the agent
oof = np.full(len(Xc), -1)
for a, b in TimeSeriesSplit(n_splits=5).split(Xc):
    oof[b] = mk().fit(Xc.iloc[a], yc[a]).predict(Xc.iloc[b])
final = mk().fit(Xc, yc); pte = final.predict(Xt)
def states(df, pred):
    today = df.aqi_today_cls.fillna(0).astype(int).values
    trend = np.sign(pred - today) + 1            # 0 falling, 1 steady, 2 rising
    return (pred * 3 + trend).astype(int)
s_tr_mask = oof >= 0
S_tr, y_tr = states(Xc[s_tr_mask], oof[s_tr_mask]), yc[s_tr_mask]
S_te, y_te = states(Xt, pte), yt
NS, NA = 12, 4
def reward(a, y):
    return 2.0 if a == y else (-3.0 * (y - a) if a < y else -1.0 * (a - y))   # under-warning hurts 3x more
# sequences (contiguous per station) so the next state follows the day order
def seq_next(df):
    nxt = np.arange(1, len(df) + 1); nxt[-1] = len(df) - 1
    st = df.station.values; same = np.r_[st[1:] == st[:-1], False]
    return nxt, same
def run_q(S, Y, df, episodes=300, alpha=0.1, gamma=0.9, eps0=1.0, eps_min=0.05, seed=SEED):
    rng = np.random.RandomState(seed); Q = np.zeros((NS, NA)); curve = []
    d_ = df.reset_index(drop=True).sort_values(["station", "date"]).index
    order_ = df.reset_index(drop=True).sort_values(["station", "date"]).index.values
    S2, Y2 = S[order_], Y[order_]; st2 = df.reset_index(drop=True).station.values[order_]
    for ep in range(episodes):
        eps = max(eps_min, eps0 * (0.985 ** ep)); tot = 0
        for t in range(len(S2)):
            s = S2[t]
            a = rng.randint(NA) if rng.rand() < eps else int(Q[s].argmax())
            r = reward(a, Y2[t]); tot += r
            last = (t == len(S2) - 1) or (st2[t + 1] != st2[t])
            target = r if last else r + gamma * Q[S2[t + 1]].max()
            Q[s, a] += alpha * (target - Q[s, a])
        curve.append(tot / len(S2))
    return Q, curve
Q, curve = run_q(S_tr, y_tr, Xc[s_tr_mask])
def evaluate(policy_actions, y):
    r = np.array([reward(a, yy) for a, yy in zip(policy_actions, y)])
    return dict(avg_reward=float(r.mean()), correct=float((policy_actions == y).mean()),
                under=float((policy_actions < y).mean()), over=float((policy_actions > y).mean()))
rng = np.random.RandomState(SEED)
ql = {"Random advice": evaluate(rng.randint(NA, size=len(y_te)), y_te),
      "Always normal activity": evaluate(np.zeros(len(y_te), int), y_te),
      "Fixed rule (advice = forecast class)": evaluate(pte, y_te),
      "Q-Learning agent": evaluate(Q[S_te].argmax(1), y_te)}
res["qlearning"] = ql; res["q_curve"] = [float(x) for x in curve]
res["q_policy"] = Q.argmax(1).tolist(); res["q_states_seen"] = int(len(np.unique(S_tr)))
res["q_n"] = dict(train=int(s_tr_mask.sum()), test=int(len(y_te)))
for k, v in ql.items(): print(k, v)
jl.dump(dict(Q=Q), "q.joblib")
json.dump(res, open("res_p2.json", "w"), indent=1)
