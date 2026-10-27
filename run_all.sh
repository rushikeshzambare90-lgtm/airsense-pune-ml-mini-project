#!/bin/bash
# Put all_stations_combined.csv path in common.py (CSV = ...) then:
python p1_supervised.py      # Units I, II, III, VI (models + metrics)  -> res_p1.json
python p2_hmm_cluster_rl.py  # Units IV, V, VI (HMM, clustering, Q-learning) -> res_p2.json
python p3_ablation.py        # does the HMM regime help?                -> res_p3.json
python p4_figures.py         # plots in figs/
streamlit run app.py         # dashboard
