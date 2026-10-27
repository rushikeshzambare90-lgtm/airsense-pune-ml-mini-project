AirSense Pune - mini project code
Data: all_stations_combined.csv (Mendeley Data, DOI 10.17632/y6p778djzj.1). Already included in data/ (licence CC BY 4.0, credit the authors).
Run order: see run_all.sh. Every random seed is 42, so numbers in the report are reproducible
(requires scikit-learn 1.8, hmmlearn 0.3.3, xgboost 3.x; other versions may change numbers slightly).

Notes
- Split is chronological: train < 2025-06-01, test >= 2025-06-01 (test = Katraj, SPPU, Wakad only).
- AQI class labels come from the source's AQI column, which can itself contain values built from faulty readings.
- HMM regime used as a feature comes from the forward algorithm (no future information).
- K-Medoids is implemented in p2 (scikit-learn-extra does not run with NumPy 2).
