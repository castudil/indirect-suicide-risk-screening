"""
Supporting experiments for the manuscript.

P1-E1  Leave-one-school-out CV across the 21 schools -> quasi-external validation
P1-E2  Criterion-contamination sensitivity (excluding BHS, BHS+CAPE)
P1-E3  Decision curve analysis + number needed to screen per tier
P1-E4  Feature-selection stability (1,000 L1 bootstraps)
P1-E5  Bootstrap internal validation with optimism correction (Harrell)
P1-E6  Realistic comparator: distal risk factors
P1-E7  Advanced calibration: intercept, slope, ICI, Brier (Van Calster hierarchy)
P1-E8  Label-noise robustness figure (reuses the E2 output)

Every experiment leaves a trace in results/RUN_MANIFEST.json and results/AUDIT-LOG.md.

Usage:  python p1_experiments.py [E1 E2 ...]   (no arguments runs all)
"""

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import _cohort as C
from _audit import Run

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
INPUTS = ["datos.csv", "_cohort.py"]

# --------------------------------------------------------------- shared data
X, y, siq, clusters = C.load_cohort()
Xtr, Xte, ytr, yte, siqtr, siqte, ctr, cte = C.split(X, y, siq, clusters)
y_tr, y_te = ytr.astype(int).to_numpy(), yte.astype(int).to_numpy()
kinds = C.columns_by_kind(X)
scaler = StandardScaler().fit(Xtr)
Atr, Ate = scaler.transform(Xtr), scaler.transform(Xte)
clf_main = LogisticRegression(**C.L1_PARAMS).fit(Atr, y_tr)
p_te = clf_main.predict_proba(Ate)[:, 1]
pred_te = clf_main.predict(Ate).astype(int)


def fit_eval(cols, params=None, name=""):
    """Fit the L1-LR on a subset of columns and evaluate on the hold-out."""
    sc = StandardScaler().fit(Xtr[cols])
    m = LogisticRegression(**(params or C.L1_PARAMS)).fit(sc.transform(Xtr[cols]), y_tr)
    pr = m.predict_proba(sc.transform(Xte[cols]))[:, 1]
    pd_ = m.predict(sc.transform(Xte[cols])).astype(int)
    return {"Model": name, "n_features": len(cols),
            "Precision": precision_score(y_te, pd_, zero_division=0),
            "Recall": recall_score(y_te, pd_, zero_division=0),
            "F2": fbeta_score(y_te, pd_, beta=2.0, zero_division=0),
            "AUC": roc_auc_score(y_te, pr), "PR AUC": average_precision_score(y_te, pr)}


# =============================================================== P1-E1  LOSO-CV
def e1_loso():
    r = Run("P1-E1", "Leave-One-School-Out CV (21 establecimientos)",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "The design is a cluster-RCT; the original 70/30 split is at participant level, "
            "so students from the same school appear in both train and test. LOSO corrects this "
            "cluster leakage and provides quasi-external validation by site.")
    rows = []
    for tr_i, va_i in LeaveOneGroupOut().split(Xtr, y_tr, ctr):
        yv = y_tr[va_i]
        school = ctr.iloc[va_i].iloc[0]
        if len(np.unique(yv)) < 2:
            rows.append({"School": school, "n": len(va_i), "Cases": int(yv.sum()),
                         "AUC": np.nan, "Recall": np.nan, "F2": np.nan}); continue
        sc = StandardScaler().fit(Xtr.iloc[tr_i])
        m = LogisticRegression(**C.L1_PARAMS).fit(sc.transform(Xtr.iloc[tr_i]), y_tr[tr_i])
        Av = sc.transform(Xtr.iloc[va_i])
        rows.append({"School": school, "n": len(va_i), "Cases": int(yv.sum()),
                     "AUC": roc_auc_score(yv, m.predict_proba(Av)[:, 1]),
                     "Recall": recall_score(yv, m.predict(Av).astype(int), zero_division=0),
                     "F2": fbeta_score(yv, m.predict(Av).astype(int), beta=2.0, zero_division=0)})
    df = pd.DataFrame(rows).sort_values("AUC")
    ok = df.dropna(subset=["AUC"])
    w = np.average(ok["AUC"], weights=ok["n"])
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nAUC    mean {ok['AUC'].mean():.3f} +/- {ok['AUC'].std():.3f} | size-weighted: {w:.3f}"
          f" | range [{ok['AUC'].min():.3f}, {ok['AUC'].max():.3f}]")
    print(f"Recall mean {ok['Recall'].mean():.3f} +/- {ok['Recall'].std():.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    yp = np.arange(len(ok))
    ax.scatter(ok["AUC"], yp, s=np.clip(ok["n"] * 1.4, 25, 260), color="#2b6cb0",
               zorder=3, edgecolor="white", linewidth=.8)
    ax.axvline(w, color="#c53030", ls="--", lw=1.6,
               label=f"Size-weighted AUC = {w:.3f}")
    ax.axvline(roc_auc_score(y_te, p_te), color="#2f855a", ls=":", lw=1.6,
               label=f"70/30 hold-out AUC = {roc_auc_score(y_te, p_te):.3f}")
    ax.set_yticks(yp)
    ax.set_yticklabels([f"School {s}  (n={n}, {k} cases)"
                        for s, n, k in zip(ok["School"], ok["n"], ok["Cases"])], fontsize=8)
    ax.set_xlabel("AUC on the held-out school")
    ax.set_title("Leave-one-school-out validation across 21 institutions\n"
                 "Point size is proportional to school enrolment", fontsize=11)
    ax.grid(axis="x", ls=":", alpha=.6); ax.legend(loc="upper left", fontsize=9, framealpha=.95)
    ax.set_ylim(-1.2, len(ok) + 0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"results/P1E1_loso_forest.{ext}")
    plt.close(fig)

    df.to_csv("results/P1E1_loso_per_school.csv", index=False)
    r.finding("auc_mean", round(float(ok["AUC"].mean()), 4))
    r.finding("auc_sd", round(float(ok["AUC"].std()), 4))
    r.finding("auc_weighted", round(float(w), 4), "weighted by school size")
    r.finding("auc_range", [round(float(ok["AUC"].min()), 4), round(float(ok["AUC"].max()), 4)])
    r.finding("recall_mean", round(float(ok["Recall"].mean()), 4))
    r.finding("schools_evaluable", int(len(ok)), f"of {len(df)} (single-class folds excluded)")
    r.output("results/P1E1_loso_per_school.csv", "AUC/Recall/F2 per school")
    r.output("results/P1E1_loso_forest.pdf", "Forest plot for the manuscript")
    r.output("results/P1E1_loso_forest.png")
    r.close()


# ================================================ P1-E2  contaminacion de criterio
def e2_criterion():
    r = Run("P1-E2", "Criterion-contamination sensitivity (BHS / BHS+CAPE excluded)",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "The BHS (hopelessness) and CAPE-P15 share semantic variance with the SIQ-JR. "
            "If performance holds when they are excluded, the indirect design does not depend on "
            "items close to the outcome construct.")
    allc = list(X.columns)
    sets = {
        "Completo (108 features)": allc,
        "Sin BHS": [c for c in allc if not c.startswith("beck")],
        "Sin BHS + CAPE-P15": [c for c in allc if not c.startswith(("beck", "cape"))],
        "PHQ-8 + GAD-7 only (affective core)": [c for c in allc if c.startswith(("phq", "gad"))],
    }
    df = pd.DataFrame([fit_eval(cols, name=n) for n, cols in sets.items()])
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    base = df.iloc[0]
    for _, row in df.iloc[1:].iterrows():
        print(f"  {row['Model']:38s} ΔRecall {row['Recall']-base['Recall']:+.4f} | "
              f"ΔAUC {row['AUC']-base['AUC']:+.4f}")
    df.to_csv("results/P1E2_criterion_sensitivity.csv", index=False)
    r.finding("recall_full", round(float(base["Recall"]), 4))
    r.finding("recall_without_bhs", round(float(df.iloc[1]["Recall"]), 4))
    r.finding("recall_without_bhs_cape", round(float(df.iloc[2]["Recall"]), 4))
    r.finding("auc_without_bhs_cape", round(float(df.iloc[2]["AUC"]), 4))
    r.output("results/P1E2_criterion_sensitivity.csv", "Performance by scale subset")
    r.close()


# ========================================================= P1-E3  DCA + NNS
def e3_dca():
    r = Run("P1-E3", "Decision curve analysis and number needed to screen per tier",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "Net benefit translates calibration into clinical utility and is the analysis "
            "clinical editors expect. NNS expresses the operational cost per tier.")
    cal = CalibratedClassifierCV(estimator=clf_main, method="isotonic", cv=5).fit(Atr, y_tr)
    p_cal = cal.predict_proba(Ate)[:, 1]

    distal = [c for c in X.columns if c.startswith(("tto_", "sui_", "suicidio_", "repeticion"))]
    sc_d = StandardScaler().fit(Xtr[distal])
    p_distal = LogisticRegression(**C.L1_PARAMS).fit(sc_d.transform(Xtr[distal]), y_tr) \
        .predict_proba(sc_d.transform(Xte[distal]))[:, 1]

    th = np.linspace(0.01, 0.50, 100)
    prev = y_te.mean()
    nb = lambda p: np.array([((p >= t).astype(int) @ y_te) / len(y_te)
                             - (((p >= t).astype(int) @ (1 - y_te)) / len(y_te)) * (t / (1 - t))
                             for t in th])
    nb_model, nb_distal = nb(p_cal), nb(p_distal)
    nb_all = prev - (1 - prev) * (th / (1 - th))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(th, nb_model, color="#c53030", lw=2.4, label="Indirect item-level model (calibrated)")
    ax.plot(th, nb_distal, color="#2b6cb0", ls="-.", lw=1.8, label="Distal risk factors only")
    ax.plot(th, nb_all, color="gray", ls="--", lw=1.4, label="Treat all")
    ax.axhline(0, color="black", lw=1.2, label="Treat none")
    ax.set_xlim(.01, .50); ax.set_ylim(-.05, .22)
    ax.set_xlabel("Risk threshold probability ($p_t$)"); ax.set_ylabel("Net benefit")
    ax.set_title("Clinical decision curve analysis", fontsize=11)
    ax.legend(fontsize=9); ax.grid(ls=":", alpha=.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"results/P1E3_decision_curve.{ext}")
    plt.close(fig)

    tiers = []
    for lo, hi, nm in [(.80, 1.01, "Tier 1 (p>=0.80)"),
                       (.50, .80, "Tier 2 (0.50<=p<0.80)"),
                       (.00, .50, "Tier 3 (p<0.50)")]:
        m = (p_cal >= lo) & (p_cal < hi)
        n, tp = int(m.sum()), int(y_te[m].sum())
        tiers.append({"Tier": nm, "n": n, "Casos reales": tp,
                      "Tier precision": tp / n if n else np.nan,
                      "NNS": n / tp if tp else np.nan})
    dft = pd.DataFrame(tiers)
    print(dft.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    gain = float((nb_model - nb_all)[np.argmin(np.abs(th - 0.19))])
    print(f"\nNet benefit at p_t=0.19 (prevalence): model {nb_model[np.argmin(np.abs(th-.19))]:.4f}"
          f" vs treat-all {nb_all[np.argmin(np.abs(th-.19))]:.4f}  (gain {gain:+.4f})")

    dft.to_csv("results/P1E3_triage_tiers_nns.csv", index=False)
    pd.DataFrame({"threshold": th, "nb_model_calibrated": nb_model,
                  "nb_distal": nb_distal, "nb_treat_all": nb_all}) \
        .to_csv("results/P1E3_net_benefit_curve.csv", index=False)
    r.finding("net_benefit_gain_at_p019", round(gain, 4), "vs treat-all strategy")
    r.finding("nns_tier1", round(float(dft.iloc[0]["NNS"]), 2))
    r.finding("nns_tier2", round(float(dft.iloc[1]["NNS"]), 2))
    r.finding("precision_tier1", round(float(dft.iloc[0]["Tier precision"]), 4))
    r.output("results/P1E3_triage_tiers_nns.csv", "Triage tiers with NNS")
    r.output("results/P1E3_net_benefit_curve.csv", "Curva de beneficio neto")
    r.output("results/P1E3_decision_curve.pdf", "DCA figure for the manuscript")
    r.output("results/P1E3_decision_curve.png")
    r.close()


# ================================================= P1-E4  estabilidad de seleccion
def e4_stability(B=1000):
    r = Run("P1-E4", "Feature-selection stability (1,000 L1 bootstraps)",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "At EPV~1.9 individual coefficients are unstable. Selection frequency "
            "distinguishes a reproducible semantic prior from an anecdotal one.")
    rng = np.random.default_rng(C.RANDOM_STATE)
    cols = list(Xtr.columns)
    counts, signs = np.zeros(len(cols)), np.zeros(len(cols))
    for _ in range(B):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        if len(np.unique(y_tr[idx])) < 2:
            continue
        sc = StandardScaler().fit(Xtr.iloc[idx])
        m = LogisticRegression(**C.L1_PARAMS).fit(sc.transform(Xtr.iloc[idx]), y_tr[idx])
        c = m.coef_.ravel()
        counts += (c != 0); signs += np.sign(c)
    df = pd.DataFrame({"Variable": cols,
                       "Selection frequency": counts / B,
                       "Sign consistency": np.where(counts > 0, np.abs(signs) / np.maximum(counts, 1), np.nan),
                       "Final model coef.": clf_main.coef_.ravel(),
                       "In final model": clf_main.coef_.ravel() != 0}).sort_values(
        "Selection frequency", ascending=False)
    print(df.head(20).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fin = df[df["In final model"]]
    n90 = int((fin["Selection frequency"] >= .90).sum())
    n50 = int((fin["Selection frequency"] >= .50).sum())
    print(f"\nOf the {len(fin)} predictors in the final model: {n90} selected >=90% of the time, "
          f"{n50} >=50%, {len(fin)-n50} below 50%.")
    df.to_csv("results/P1E4_selection_stability.csv", index=False)
    r.finding("n_bootstraps", B)
    r.finding("n_final_predictors", int(len(fin)))
    r.finding("n_stable_90", n90, "selection frequency >= 90%")
    r.finding("n_stable_50", n50, "selection frequency >= 50%")
    r.finding("n_unstable", int(len(fin) - n50), "in the final model but selected < 50% of the time")
    r.output("results/P1E4_selection_stability.csv", "Selection frequency per variable")
    r.close()


# ============================================ P1-E5  bootstrap de optimismo
def e5_optimism(B=200):
    r = Run("P1-E5", "Bootstrap internal validation with optimism correction (Harrell)",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "Collins and Steyerberg regard split-sample validation as statistically inefficient. "
            "The bootstrap uses all 1,539 cases and estimates optimism directly.")
    rng = np.random.default_rng(C.RANDOM_STATE)
    sc_f = StandardScaler().fit(X); A_full = sc_f.transform(X); y_full = y.astype(int).to_numpy()
    m_app = LogisticRegression(**C.L1_PARAMS).fit(A_full, y_full)
    app_auc = roc_auc_score(y_full, m_app.predict_proba(A_full)[:, 1])
    app_pr = average_precision_score(y_full, m_app.predict_proba(A_full)[:, 1])
    o_auc, o_pr = [], []
    for _ in range(B):
        idx = rng.integers(0, len(X), len(X))
        if len(np.unique(y_full[idx])) < 2:
            continue
        sc = StandardScaler().fit(X.iloc[idx])
        m = LogisticRegression(**C.L1_PARAMS).fit(sc.transform(X.iloc[idx]), y_full[idx])
        pb = m.predict_proba(sc.transform(X.iloc[idx]))[:, 1]
        po = m.predict_proba(sc.transform(X))[:, 1]
        o_auc.append(roc_auc_score(y_full[idx], pb) - roc_auc_score(y_full, po))
        o_pr.append(average_precision_score(y_full[idx], pb) - average_precision_score(y_full, po))
    c_auc, c_pr = app_auc - np.mean(o_auc), app_pr - np.mean(o_pr)
    print(f"AUC    aparente {app_auc:.4f} | optimismo {np.mean(o_auc):+.4f} | corregida {c_auc:.4f}")
    print(f"PR AUC aparente {app_pr:.4f} | optimismo {np.mean(o_pr):+.4f} | corregida {c_pr:.4f}")
    print(f"Referencia hold-out 70/30: AUC {roc_auc_score(y_te, p_te):.4f} | "
          f"PR AUC {average_precision_score(y_te, p_te):.4f}")
    pd.DataFrame([{"Metrica": "AUC", "Apparent": app_auc, "Optimism": np.mean(o_auc),
                   "Corrected": c_auc, "Hold-out 70/30": roc_auc_score(y_te, p_te)},
                  {"Metrica": "PR AUC", "Apparent": app_pr, "Optimism": np.mean(o_pr),
                   "Corrected": c_pr, "Hold-out 70/30": average_precision_score(y_te, p_te)}]) \
        .to_csv("results/P1E5_optimism_bootstrap.csv", index=False)
    r.finding("n_bootstraps", B)
    r.finding("auc_corrected", round(float(c_auc), 4))
    r.finding("prauc_corrected", round(float(c_pr), 4))
    r.finding("optimism_auc", round(float(np.mean(o_auc)), 4))
    r.output("results/P1E5_optimism_bootstrap.csv", "Optimism-corrected AUC / PR AUC")
    r.close()


# ================================================ P1-E6  comparador distal
def e6_distal():
    r = Run("P1-E6", "Realistic comparator: distal risk factors vs the indirect model",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "Replaces the Oracle experiment, which is circular because it predicts the SIQ-JR "
            "from SIQ-JR items. Distal factors are the screening that "
            "schools currently use, and therefore the honest comparator.")
    allc = list(X.columns)
    sets = {
        "Distal factors (personal and family history)":
            [c for c in allc if c.startswith(("tto_", "sui_", "suicidio_", "repeticion"))],
        "Distal + demographics":
            [c for c in allc if c in kinds["non_psychometric"]],
        "Classical scales, item level (PHQ-8+GAD-7+BHS)":
            [c for c in allc if c.startswith(("phq", "gad", "beck"))],
        "Indirect item-level model (proposed)": allc,
    }
    df = pd.DataFrame([fit_eval(cols, name=n) for n, cols in sets.items()])
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    d = df.iloc[-1]["Recall"] - df.iloc[0]["Recall"]
    print(f"\nAdvantage of the indirect model over distal factors: dRecall {d:+.4f}, "
          f"dAUC {df.iloc[-1]['AUC']-df.iloc[0]['AUC']:+.4f}")
    df.to_csv("results/P1E6_distal_comparator.csv", index=False)
    r.finding("recall_distal", round(float(df.iloc[0]["Recall"]), 4))
    r.finding("auc_distal", round(float(df.iloc[0]["AUC"]), 4))
    r.finding("recall_proposed", round(float(df.iloc[-1]["Recall"]), 4))
    r.finding("delta_recall", round(float(d), 4))
    r.output("results/P1E6_distal_comparator.csv", "Comparison against distal risk screening")
    r.close()


# ============================================== P1-E7  calibracion avanzada
def e7_calibration():
    r = Run("P1-E7", "Advanced calibration: intercept, slope, ICI and Brier",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "Van Calster requires slope and intercept; the Brier score alone conflates "
            "discrimination with calibration. We target moderate, not strong, calibration.")
    cal = CalibratedClassifierCV(estimator=clf_main, method="isotonic", cv=5).fit(Atr, y_tr)
    p_cal = cal.predict_proba(Ate)[:, 1]

    def metrics(p):
        pc = np.clip(p, 1e-9, 1 - 1e-9)
        lg = np.log(pc / (1 - pc))
        fit = sm.Logit(y_te, sm.add_constant(lg)).fit(disp=0)
        sm_l = sm.nonparametric.lowess(y_te, p, frac=.2, it=0)
        ici = np.mean(np.abs(p - np.interp(p, sm_l[:, 0], sm_l[:, 1])))
        return {"Intercept (calibration-in-the-large)": fit.params[0],
                "Calibration slope": fit.params[1], "ICI": ici,
                "Brier": brier_score_loss(y_te, p)}

    df = pd.DataFrame([{"Model": "L1-LR uncalibrated", **metrics(p_te)},
                       {"Model": "L1-LR + isotonic", **metrics(p_cal)}])
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    b0, b1 = df.iloc[0]["Brier"], df.iloc[1]["Brier"]
    print(f"\nBrier: {b0:.4f} -> {b1:.4f}  ({100*(b1-b0)/b0:+.2f}%)  "
          f"[manuscrito: 0.1205 -> 0.0848, -29.63%]")
    df.to_csv("results/P1E7_calibration_advanced.csv", index=False)
    r.finding("slope_uncalibrated", round(float(df.iloc[0]["Calibration slope"]), 4),
              "1.0 = perfect calibration; <1 indicates overconfidence")
    r.finding("slope_calibrated", round(float(df.iloc[1]["Calibration slope"]), 4))
    r.finding("ici_uncalibrated", round(float(df.iloc[0]["ICI"]), 4))
    r.finding("ici_calibrated", round(float(df.iloc[1]["ICI"]), 4))
    r.finding("brier_improvement_pct", round(float(100 * (b1 - b0) / b0), 2))
    r.output("results/P1E7_calibration_advanced.csv", "Van Calster hierarchy metrics")
    r.close()


# ========================================== P1-E8  label-noise robustness figure
def e8_noise_figure():
    src = "results/E2_label_noise_curves.csv"
    r = Run("P1-E8", "Label-noise robustness figure (carried over from E2)",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS + [src],
            "Result carried over from the discontinued unsupervised analysis: L1-LR discrimination is "
            "invariant to asymmetric label noise up to the 39% documented by "
            "Klimes-Dougan, turning a threat cited in the manuscript into a demonstrated "
            "strength. LIMITATION: noise was injected only in training; the hold-out ground "
            "truth is assumed clean.")
    ENMECH = {"severity-targeted": "Severity-targeted", "uniform random (control)": "Uniform random"}
    d = pd.read_csv(src)
    sup = d[d["Model"] == "L1-LR supervised"]

    # The manuscript figure shows ONLY the supervised model: label-free scores
    # belong to the discontinued unsupervised analysis. Axes are fixed on a wide,
    # common range so that invariance reads as such; a compressed axis would
    # visually exaggerate a drop of four tenths of a point.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    panels = [("PR AUC", "PR AUC on the uncorrupted hold-out"),
              ("F2", "$F_2$ at the operating point")]
    for ax, (met, lab) in zip(axes, panels):
        for mech, col, mk in [("severity-targeted", "#c53030", "o"),
                              ("uniform random (control)", "#4a5568", "s")]:
            s = sup[sup["Mechanism"] == mech].sort_values("Noise")
            ax.plot(s["Noise"] * 100, s[met], marker=mk, ls="-", color=col, lw=2.1, ms=6,
                    label=f"{ENMECH[mech]} noise", zorder=3)
        ref = sup[(sup["Noise"] == 0) & (sup["Mechanism"] == "severity-targeted")][met].iloc[0]
        ax.axhline(ref, color="#2b6cb0", ls=":", lw=1.4,
                   label=f"Noise-free reference = {ref:.3f}")
        ax.axvspan(30, 39, color="#fed7d7", alpha=.45, zorder=0)
        ax.set_ylim(0.50, 0.95)
        ax.set_xlabel("Asymmetric label noise injected in training (%)")
        ax.set_ylabel(lab)
        ax.grid(ls=":", alpha=.6)
        ax.legend(fontsize=8.5, loc="lower left", framealpha=.95)
        fin = sup[(sup["Noise"] == 0.39) & (sup["Mechanism"] == "severity-targeted")][met].iloc[0]
        ax.annotate(f"{100*(fin-ref)/ref:+.1f}%", xy=(39, fin), xytext=(30.5, ref + 0.06),
                    fontsize=9, color="#c53030", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.2))
    fig.suptitle("Robustness of the supervised model to label noise\n"
                 "(shaded band = 39% inconsistency rate reported by Klimes-Dougan et al., 2022)",
                 fontsize=10.5, y=1.05)
    for ext in ("pdf", "png"):
        fig.savefig(f"results/P1E8_label_noise_robustness.{ext}")
    plt.close(fig)

    st = sup[sup["Mechanism"] == "severity-targeted"].sort_values("Noise")
    a, b = st.iloc[0]["PR AUC"], st.iloc[-1]["PR AUC"]
    f2a, f2b = st.iloc[0]["F2"], st.iloc[-1]["F2"]
    print(f"PR AUC 0% -> 39%: {a:.4f} -> {b:.4f}  ({100*(b-a)/a:+.2f}%)")
    print(f"F2     0% -> 39%: {f2a:.4f} -> {f2b:.4f}  ({100*(f2b-f2a)/f2a:+.2f}%)")
    r.finding("prauc_noise_0", round(float(a), 4))
    r.finding("prauc_noise_39", round(float(b), 4))
    r.finding("degradation_prauc_pct", round(float(100 * (b - a) / a), 2))
    r.finding("degradation_f2_pct", round(float(100 * (f2b - f2a) / f2a), 2))
    r.output("results/P1E8_label_noise_robustness.pdf", "Robustness figure for the manuscript")
    r.output("results/P1E8_label_noise_robustness.png")
    r.close()



# ================================== P1-E9  aggregate scale-score comparator
def e9_scale_totals():
    r = Run("P1-E9", "Comparación contra puntajes totales de escala (objeción M1)",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "This is the comparison any psychometrically trained reviewer will make: if three scale sums "
            "match the 108-item model, item-level resolution does not buy accuracy. "
            "It is reported explicitly rather than omitted.")
    scales = {"PHQ-8": "phq", "GAD-7": "gad", "BHS": "beck",
              "CAPE-P15": "cape", "SPSI-R": "rps", "CBT-SQ": "cbt", "ERQ-CA": "erq"}
    tot_tr = pd.DataFrame({k: Xtr[[c for c in X.columns if c.startswith(v)]].sum(axis=1)
                           for k, v in scales.items()})
    tot_te = pd.DataFrame({k: Xte[[c for c in X.columns if c.startswith(v)]].sum(axis=1)
                           for k, v in scales.items()})

    def ev(cols, name):
        sc = StandardScaler().fit(tot_tr[cols])
        m = LogisticRegression(class_weight="balanced", random_state=C.RANDOM_STATE) \
            .fit(sc.transform(tot_tr[cols]), y_tr)
        pr = m.predict_proba(sc.transform(tot_te[cols]))[:, 1]
        pdz = m.predict(sc.transform(tot_te[cols])).astype(int)
        return {"Configuration": name, "n feat": len(cols),
                "Recall": recall_score(y_te, pdz, zero_division=0),
                "F2": fbeta_score(y_te, pdz, beta=2.0, zero_division=0),
                "AUC": roc_auc_score(y_te, pr), "PR AUC": average_precision_score(y_te, pr)}

    rows = [ev(["PHQ-8"], "PHQ-8 total score alone"),
            ev(["BHS"], "BHS total score alone"),
            ev(["PHQ-8", "GAD-7", "BHS"], "PHQ-8 + GAD-7 + BHS total scores"),
            ev(list(scales), "All seven total scores"),
            {"Configuration": "Item-level model (proposed)", "n feat": X.shape[1],
             "Recall": recall_score(y_te, pred_te), "F2": fbeta_score(y_te, pred_te, beta=2.0),
             "AUC": roc_auc_score(y_te, p_te), "PR AUC": average_precision_score(y_te, p_te)}]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    three, item = df.iloc[2], df.iloc[4]
    print(f"\nDelta AUC (item-level minus three totals): {item['AUC']-three['AUC']:+.4f}")
    print(f"Delta PR AUC: {item['PR AUC']-three['PR AUC']:+.4f} | Delta Recall: {item['Recall']-three['Recall']:+.4f}")
    df.to_csv("results/P1E9_scale_totals.csv", index=False)
    r.finding("auc_three_totals", round(float(three["AUC"]), 4))
    r.finding("auc_item_level", round(float(item["AUC"]), 4))
    r.finding("delta_auc", round(float(item["AUC"] - three["AUC"]), 4),
              "zero or negative = item-level resolution does not improve discrimination")
    r.finding("delta_prauc", round(float(item["PR AUC"] - three["PR AUC"]), 4))
    r.output("results/P1E9_scale_totals.csv", "Comparison against aggregate scores")
    r.close()


# ============================== P1-E10  derivation of the triage cut-points
def e10_triage_thresholds():
    from sklearn.model_selection import cross_val_predict
    r = Run("P1-E10", "Derivation of triage cut-points on calibrated probability",
            "p1_experiments.py", C.RANDOM_STATE, INPUTS,
            "The inherited 0.80/0.50 cut-points produced a protocol with Recall 0.614 while the "
            "abstract claimed 0.841. The new cut-points derive from prespecified clinical "
            "criteria applied to out-of-fold predictions on the TRAINING partition, never the hold-out.")
    cal = CalibratedClassifierCV(estimator=clf_main, method="isotonic", cv=5).fit(Atr, y_tr)
    p_cal = cal.predict_proba(Ate)[:, 1]
    p_oof = cross_val_predict(
        CalibratedClassifierCV(estimator=LogisticRegression(**C.L1_PARAMS), method="isotonic", cv=5),
        Atr, y_tr, cv=5, method="predict_proba")[:, 1]

    # Criterion 1: Tier 1 must reach precision >= 0.80 (avoid unnecessary urgent referrals)
    t1 = next(t for t in np.arange(0.01, 1.0, 0.01)
              if precision_score(y_tr, (p_oof >= t).astype(int), zero_division=0) >= 0.80)
    # Criterion 2: Tier 1+2 must reach recall >= 0.90 (screening imperative)
    t2 = next(t for t in np.arange(0.99, 0.0, -0.01)
              if recall_score(y_tr, (p_oof >= t).astype(int), zero_division=0) >= 0.90)
    print(f"Cut-points derived on training: Tier 1 p>={t1:.2f} | Tier 2 p>={t2:.2f}")

    rows, cum_n, cum_tp = [], 0, 0
    for lo, hi, nm in [(t1, 1.01, f"Tier 1: High alert ($p\\ge{t1:.2f}$)"),
                       (t2, t1, f"Tier 2: Elevated (${t2:.2f}\\le p<{t1:.2f}$)"),
                       (0.0, t2, f"Tier 3: Standard ($p<{t2:.2f}$)")]:
        msk = (p_cal >= lo) & (p_cal < hi)
        n, tp = int(msk.sum()), int(y_te[msk].sum())
        if "Tier 3" not in nm:
            cum_n += n; cum_tp += tp
        rows.append({"Tier": nm, "n": n, "True cases": tp,
                     "Tier precision": tp / n if n else np.nan,
                     "Cumulative recall": cum_tp / y_te.sum() if "Tier 3" not in nm else np.nan,
                     "Cumulative NNS": cum_n / cum_tp if ("Tier 3" not in nm and cum_tp) else np.nan})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    flag = (p_cal >= t2).astype(int)
    print(f"\nFull protocol (Tier 1+2): {int(flag.sum())} flagged of {len(y_te)} "
          f"({100*flag.mean():.1f}% of the cohort) | Recall={recall_score(y_te, flag):.3f} "
          f"| Precision={precision_score(y_te, flag):.3f}")
    print(f"Comparison: default 0.5 uncalibrated threshold -> Recall={recall_score(y_te, pred_te):.3f}")
    df.to_csv("results/P1E10_triage_thresholds.csv", index=False)
    r.finding("cut_tier1", round(float(t1), 2), "precision >= 0.80 on training out-of-fold")
    r.finding("cut_tier2", round(float(t2), 2), "recall >= 0.90 on training out-of-fold")
    r.finding("recall_protocol", round(float(recall_score(y_te, flag)), 4))
    r.finding("precision_protocol", round(float(precision_score(y_te, flag)), 4))
    r.finding("pct_cohort_flagged", round(float(100 * flag.mean()), 1))
    r.output("results/P1E10_triage_thresholds.csv", "Triage tiers with cumulative recall")
    r.close()


EXPERIMENTS = {"E1": e1_loso, "E2": e2_criterion, "E3": e3_dca, "E4": e4_stability,
               "E5": e5_optimism, "E6": e6_distal, "E7": e7_calibration, "E8": e8_noise_figure,
               "E9": e9_scale_totals, "E10": e10_triage_thresholds}

if __name__ == "__main__":
    print("Replication check of the canonical pipeline:")
    C.verify()
    which = [a.upper() for a in sys.argv[1:]] or list(EXPERIMENTS)
    for key in which:
        print("\n" + "=" * 78)
        print(f"P1-{key}")
        print("=" * 78)
        EXPERIMENTS[key]()
    print("\n\nTraza en results/AUDIT-LOG.md y results/RUN_MANIFEST.json")
