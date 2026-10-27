import json, numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8})
res = {**json.load(open("res_p1.json")), **json.load(open("res_p2.json"))}
R = pd.read_pickle("regimes.pkl")
from common import *
d, _, _ = load_clean(); F = build_frame(d)
fig, ax = plt.subplots(2, 2, figsize=(8.2, 6.0), dpi=200)
# (a) HMM timeline
a = ax[0, 0]; g = R[R.station == "Katraj"].merge(F[["station", "date"]], on=["station", "date"]).sort_values("date")
K = res["hmm_K"]; cols = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, K))
for s in range(K):
    m = g[g.viterbi == s]; a.scatter(m.date, m.pm25, s=4, color=cols[s], label=f"State {s}")
a.set_title("(a) HMM regimes (Viterbi) - Katraj, PM2.5 (µg/m³)"); a.legend(ncol=5, fontsize=6, loc="upper center", frameon=False, handletextpad=0.1, columnspacing=0.8)
a.set_ylim(0, 200); a.tick_params(axis="x", labelsize=7)
# (b) ROC
a = ax[0, 1]; roc = joblib.load("roc.joblib")
for k, (fpr, tpr) in roc.items(): a.plot(fpr, tpr, lw=1.1, label=k.replace(" (RF+GB+XGB -> LogReg)", ""))
a.plot([0, 1], [0, 1], "k--", lw=0.7); a.set_title("(b) ROC: unhealthy tomorrow (Poor or worse)"); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); a.legend(fontsize=6, loc="lower right")
# (c) clusters PCA
a = ax[1, 0]; cp = joblib.load("clu_plot.joblib"); P2 = cp["P2"]; db = cp["db"]; km = cp["km"]
sc = a.scatter(P2[:, 0], P2[:, 1], c=km, s=5, cmap="viridis", alpha=0.7)
a.scatter(P2[db == -1, 0], P2[db == -1, 1], s=22, facecolors="none", edgecolors="red", lw=0.8, label="DBSCAN outliers")
a.set_title("(c) K-Means clusters (PCA view) and DBSCAN outliers"); a.set_xlabel(f"PC1 ({cp['ev'][0]*100:.0f}%)"); a.set_ylabel(f"PC2 ({cp['ev'][1]*100:.0f}%)"); a.legend(fontsize=6)
# (d) reward curve
a = ax[1, 1]; q = np.array(res["q_curve"]); a.plot(q, lw=0.9, color="#1f77b4")
a.plot(pd.Series(q).rolling(15, min_periods=1).mean(), color="k", lw=1.0, label="15-episode mean")
a.set_title("(d) Q-Learning: average reward per episode"); a.set_xlabel("Episode"); a.set_ylabel("Avg reward / day"); a.legend(fontsize=6)
plt.tight_layout(); plt.savefig("figs/fig2_results.png", facecolor="white"); print("saved")
# extra: elbow / silhouette
fig, a = plt.subplots(1, 2, figsize=(7, 2.6), dpi=200); e = res["elbow"]
a[0].plot(e["k"], e["inertia"], "o-"); a[0].set_title("Elbow method (K-Means)"); a[0].set_xlabel("k"); a[0].set_ylabel("Inertia")
a[1].plot(e["k"], e["silhouette"], "o-", color="g"); a[1].set_title("Silhouette vs k"); a[1].set_xlabel("k")
plt.tight_layout(); plt.savefig("figs/elbow.png", facecolor="white")
