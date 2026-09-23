"""
E2 - Robustness to label noise.

Klimes-Dougan et al. (2022): retrospective self-reports of self-injurious thoughts and
behaviours show ~39% inconsistency, and the forgetting is not random: it concentrates
among individuals with better adaptive functioning. That is, asymmetric noise (1->0)
correlated with severity.

Question: above what noise level does a LABEL-FREE score outperform a supervised model
trained on those same corrupted labels?

Design:
  - Noise is injected ONLY into the training labels.
  - Evaluation is ALWAYS against the clean hold-out labels.
  - Two threshold regimes for the label-free scores:
      (a) contamination = OBSERVED (noisy) prevalence -> inherits the bias
      (b) contamination = FIXED institutional capacity -> immune by design
  - Two noise mechanisms: severity-targeted (Klimes-Dougan) and uniform random (control).

Usage:  python e2_label_noise_robustness.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_REPS = 100
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.39]
Z_THRESHOLD = 3.0
PSYCHOMETRIC_PREFIXES = ['phq', 'gad', 'cape', 'rps', 'beck', 'cbt', 'erq', 'erc', 'er_']
DISTRESS_PREFIXES = ['phq', 'gad', 'cape', 'beck']
L1_PARAMS = dict(C=0.04935326607350617, penalty='l1', solver='liblinear',
                 class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)

report = []


def log(msg=""):
    print(msg)
    report.append(str(msg))


# ------------------------------------------------------------------ DATA
df = pd.read_csv("datos.csv", delimiter=";", encoding="latin1")
df.columns = df.columns.str.strip()
df.drop(["id", "fecha_0", "curso", "establecimiento", "grupo"], axis=1, inplace=True)
df.drop(["phq9_total_0", "gad7_total_0", "cape-p15_total_0", "rps_total_0", "beck_total_0",
         "cape-p15_pi_0", "cape-p15_be_0", "cape-p15_pa_0", "cbt_ac_0", "cbt_rc_0",
         "erc_rc_0", "er_es_0", "gad7_8_0"], axis=1, inplace=True)
df.rename(columns={"cbt_13_0": "cbt_14_0", "cbt_14_0": "cbt_16_0"}, inplace=True)
df.replace(",", ".", regex=True, inplace=True)
df.replace("#¡NULO!", float("nan"), inplace=True)
df.drop([f"siq_{i}_0" for i in range(1, 16)], axis=1, inplace=True)
df = df.astype(float).dropna()

X = df.drop("siq_total_0", axis=1)
siq = df["siq_total_0"] - 15
y_severe = siq >= 31

X_train, X_test, y_train, y_test, siq_train, _ = train_test_split(
    X, y_severe, siq, test_size=0.3, stratify=y_severe, random_state=RANDOM_STATE)

y_tr_clean = y_train.to_numpy().astype(int)
y_te = y_test.to_numpy().astype(int)          # SIEMPRE limpio
siq_tr = siq_train.to_numpy()
base_prev = float(y_tr_clean.mean())

psych = [c for c in X.columns if any(c.lower().startswith(p) for p in PSYCHOMETRIC_PREFIXES)]
distress = [c for c in psych if any(c.lower().startswith(p) for p in DISTRESS_PREFIXES)]
Xtr_p, Xte_p = X_train[psych], X_test[psych]

scaler_p = StandardScaler().fit(Xtr_p)
Ztr_p, Zte_p = scaler_p.transform(Xtr_p), scaler_p.transform(Xte_p)
d_idx = [psych.index(c) for c in distress]

scaler_full = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler_full.transform(X_train), scaler_full.transform(X_test)

log("=" * 80)
log("E2 - ROBUSTEZ AL RUIDO DE ETIQUETA (supervisado vs. label-free)")
log("=" * 80)
log(f"Train {len(y_tr_clean)} (clean prevalence {base_prev:.4f}) | Test {len(y_te)} (clean, {y_te.sum()} cases)")
log(f"Replicates per level: {N_REPS} | Noise levels (flip 1->0): {NOISE_LEVELS}")


# --------------------------------------------------------- LABEL-FREE SCORES
def robust_z(Xp, med, mad):
    return ((Xp - med) / (1.4826 * mad.replace(0, 1e-9))).abs().gt(Z_THRESHOLD) \
        .sum(axis=1).to_numpy() / float(Xp.shape[1])


med_tr, mad_tr = Xtr_p.median(), (Xtr_p - Xtr_p.median()).abs().median()
pca = PCA(n_components=1, random_state=RANDOM_STATE).fit(Ztr_p)
pc1_tr = pca.transform(Ztr_p).ravel()
sign = np.sign(np.corrcoef(pc1_tr, Ztr_p[:, d_idx].sum(axis=1))[0, 1]) or 1.0

LF = {  # name -> (score_train, score_test). None uses labels.
    "S1 Robust Z-Score (MAD)": (robust_z(Xtr_p, med_tr, mad_tr), robust_z(Xte_p, med_tr, mad_tr)),
    "S3 Sum of z (distress)": (Ztr_p[:, d_idx].sum(axis=1), Zte_p[:, d_idx].sum(axis=1)),
    "S5 PC1 (p-factor)": (sign * pc1_tr, sign * pca.transform(Zte_p).ravel()),
}
# PR AUC of label-free scores: noise-invariant, computed once.
LF_PRAUC = {k: average_precision_score(y_te, v[1]) for k, v in LF.items()}
# Regime (b): fixed institutional capacity, defined WITHOUT looking at labels.
FIXED_CAPACITY = 0.19


def inject_noise(y_clean, severity, rate, rng, targeted=True):
    """Flip 1->0. If targeted, forgetting probability decreases with severity
    (Klimes-Dougan: better-functioning individuals forget more)."""
    y = y_clean.copy()
    pos = np.flatnonzero(y_clean == 1)
    n_flip = int(round(rate * len(pos)))
    if n_flip == 0:
        return y
    if targeted:
        s = severity[pos].astype(float)
        w = (s.max() - s) + 1e-6          # lower severity -> higher forgetting weight
        w = w / w.sum()
        chosen = rng.choice(pos, size=n_flip, replace=False, p=w)
    else:
        chosen = rng.choice(pos, size=n_flip, replace=False)
    y[chosen] = 0
    return y


# ------------------------------------------------------------ EXPERIMENT
rows = []
for mechanism, targeted in [("severity-targeted", True), ("uniform random (control)", False)]:
    for rate in NOISE_LEVELS:
        rng = np.random.default_rng(RANDOM_STATE + int(rate * 1000))
        reps = 1 if rate == 0.0 else N_REPS
        sup_pr, sup_f2, sup_rc, obs_prev = [], [], [], []
        lf_noisy = {k: {"F2": [], "Recall": []} for k in LF}

        for _ in range(reps):
            y_noisy = inject_noise(y_tr_clean, siq_tr, rate, rng, targeted)
            prev_obs = float(y_noisy.mean())
            obs_prev.append(prev_obs)

            clf = LogisticRegression(**L1_PARAMS).fit(Xtr_s, y_noisy)
            p_te = clf.predict_proba(Xte_s)[:, 1]
            pred = clf.predict(Xte_s).astype(int)
            sup_pr.append(average_precision_score(y_te, p_te))
            sup_f2.append(fbeta_score(y_te, pred, beta=2.0, zero_division=0))
            sup_rc.append(recall_score(y_te, pred, zero_division=0))

            # Regime (a): the label-free threshold inherits the noisy prevalence.
            for k, (s_tr, s_te) in LF.items():
                thr = np.percentile(s_tr, 100 * (1 - prev_obs))
                fl = (s_te >= thr).astype(int)
                lf_noisy[k]["F2"].append(fbeta_score(y_te, fl, beta=2.0, zero_division=0))
                lf_noisy[k]["Recall"].append(recall_score(y_te, fl, zero_division=0))

        rows.append({"Mechanism": mechanism, "Noise": rate, "Model": "L1-LR supervised",
                     "Threshold regime": "labels", "PR AUC": np.mean(sup_pr),
                     "PR AUC DE": np.std(sup_pr), "F2": np.mean(sup_f2),
                     "Recall": np.mean(sup_rc), "Observed prev.": np.mean(obs_prev)})
        for k in LF:
            rows.append({"Mechanism": mechanism, "Noise": rate, "Model": k,
                         "Threshold regime": "(a) observed prevalence", "PR AUC": LF_PRAUC[k],
                         "PR AUC DE": 0.0, "F2": np.mean(lf_noisy[k]["F2"]),
                         "Recall": np.mean(lf_noisy[k]["Recall"]),
                         "Observed prev.": np.mean(obs_prev)})
            thr_fx = np.percentile(LF[k][0], 100 * (1 - FIXED_CAPACITY))
            fl_fx = (LF[k][1] >= thr_fx).astype(int)
            rows.append({"Mechanism": mechanism, "Noise": rate, "Model": k,
                         "Threshold regime": "(b) fixed capacity 19%", "PR AUC": LF_PRAUC[k],
                         "PR AUC DE": 0.0,
                         "F2": fbeta_score(y_te, fl_fx, beta=2.0, zero_division=0),
                         "Recall": recall_score(y_te, fl_fx, zero_division=0),
                         "Observed prev.": np.mean(obs_prev)})

res = pd.DataFrame(rows)

# ------------------------------------------------------------- REPORT
for mech in res["Mechanism"].unique():
    sub = res[res["Mechanism"] == mech]
    log()
    log("-" * 80)
    log(f"NOISE MECHANISM: {mech.upper()}")
    log("-" * 80)
    piv = sub[sub["Threshold regime"] != "(b) fixed capacity 19%"].pivot_table(
        index="Noise", columns="Model", values="PR AUC")
    log("PR AUC on the clean hold-out:")
    log(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    log()
    log("F2 at the operating point, regime (a) = threshold tied to observed prevalence:")
    log(sub[sub["Threshold regime"] != "(b) fixed capacity 19%"].pivot_table(
        index="Noise", columns="Model", values="F2").to_string(float_format=lambda v: f"{v:.4f}"))
    log()
    log("F2 at the operating point, regime (b) = fixed institutional capacity (label-free):")
    log(sub[sub["Threshold regime"] != "(a) observed prevalence"].pivot_table(
        index="Noise", columns="Model", values="F2").to_string(float_format=lambda v: f"{v:.4f}"))

# ------------------------------------------------------------ CROSSOVERS
log()
log("=" * 80)
log("CROSSOVER POINTS (PR AUC): noise level at which the supervised model falls below each score")
log("=" * 80)
cross = []
for mech in res["Mechanism"].unique():
    sup = res[(res["Mechanism"] == mech) & (res["Model"] == "L1-LR supervised")] \
        .set_index("Noise")["PR AUC"]
    for k, v in LF_PRAUC.items():
        below = sup[sup < v]
        xp = float(below.index.min()) if len(below) else None
        cross.append({"Mechanism": mech, "Label-free score": k, "Label-free PR AUC": v,
                      "Crossover (noise level)": xp if xp is not None else ">0.39",
                      "Supervised PR AUC @0.39": float(sup.loc[0.39])})
        log(f"  [{mech}] {k}: PR AUC {v:.4f} | supervised falls below at noise = "
            f"{f'{xp:.0%}' if xp is not None else 'never within the evaluated range'}")
df_cross = pd.DataFrame(cross)

sup0 = res[(res["Model"] == "L1-LR supervised") & (res["Noise"] == 0.0)]["PR AUC"].iloc[0]
sup39 = res[(res["Model"] == "L1-LR supervised") & (res["Noise"] == 0.39) &
            (res["Mechanism"] == "severity-targeted")]["PR AUC"].iloc[0]

log()
log("=" * 80)
log("READING")
log("=" * 80)
log(f"Supervised degradation under targeted noise 0% -> 39%: "
    f"PR AUC {sup0:.4f} -> {sup39:.4f}  ({100 * (sup39 - sup0) / sup0:+.1f}%)")
best_lf = max(LF_PRAUC, key=LF_PRAUC.get)
log(f"Best label-free score: {best_lf} (PR AUC {LF_PRAUC[best_lf]:.4f}, constant by construction)")
log(f"S1 Robust Z-Score: PR AUC {LF_PRAUC['S1 Robust Z-Score (MAD)']:.4f} "
    f"-> {'IS the best label-free score' if best_lf.startswith('S1') else 'is NOT the best label-free score'}")

res.to_csv("results/E2_label_noise_curves.csv", index=False)
df_cross.to_csv("results/E2_crossover_points.csv", index=False)
with open("results/E2_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(report) + "\n")
with open("results/E2_metadata.json", "w", encoding="utf-8") as fh:
    json.dump({"random_state": RANDOM_STATE, "n_reps": N_REPS, "noise_levels": NOISE_LEVELS,
               "fixed_capacity": FIXED_CAPACITY, "clean_train_prevalence": base_prev,
               "lf_prauc": LF_PRAUC, "sup_prauc_clean": float(sup0),
               "sup_prauc_noise39": float(sup39)}, fh, indent=2)
print("\n-> results/E2_label_noise_curves.csv, E2_crossover_points.csv, E2_report.txt, E2_metadata.json")
