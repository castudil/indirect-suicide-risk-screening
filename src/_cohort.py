"""
Canonical cohort module. Single source of truth for ingestion, cleaning and
partitioning across ALL experiments. Any script needing the data should import
from here rather than re-implementing the preprocessing.

Includes replication assertions (`verify()`) that fail loudly if the pipeline
stops reproducing the figures published in the manuscript, turning any silent
data or library-version drift into a visible error.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.3
SIQ_OFFSET = 15          # the SIQ-JR is scored 1-7 per item; subtracted to anchor at 0
SIQ_THRESHOLD = 31       # clinical cut-off for high suicidal risk

# Final model hyperparameters, from the Bayesian optimisation
# (results/best_hyperparameters.json, replicated in notebook2.ipynb).
L1_PARAMS = dict(C=0.04935326607350617, penalty='l1', solver='liblinear',
                 class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)

PSYCHOMETRIC_PREFIXES = ['phq', 'gad', 'cape', 'rps', 'beck', 'cbt', 'erq', 'erc', 'er_']
DISTRESS_PREFIXES = ['phq', 'gad', 'cape', 'beck']          # higher = worse, unambiguously
SCALE_PREFIXES = {'PHQ-8': 'phq', 'GAD-7': 'gad', 'CAPE-P15': 'cape',
                  'SPSI-R': 'rps', 'BHS': 'beck', 'CBT-SQ': 'cbt', 'ERQ-CA': 'erq'}

_DROP_TOTALS = ["phq9_total_0", "gad7_total_0", "cape-p15_total_0", "rps_total_0",
                "beck_total_0", "cape-p15_pi_0", "cape-p15_be_0", "cape-p15_pa_0",
                "cbt_ac_0", "cbt_rc_0", "erc_rc_0", "er_es_0", "gad7_8_0"]

# Figures published in the manuscript, used as the replication test.
EXPECTED = {"n_total": 1539, "n_train": 1077, "n_test": 462,
            "n_severe": 293, "n_severe_test": 88, "n_schools": 21,
            "n_nonzero_coefs": 35}


def load_cohort(csv_path="datos.csv"):
    """Return (X, y, siq, clusters) after complete-case cleaning.

    `clusters` retains the school identifier, which the original pipeline
    discarded and which is required for leave-one-school-out validation.
    """
    df = pd.read_csv(csv_path, delimiter=",", encoding="latin1")
    df.columns = df.columns.str.strip()

    clusters_raw = df["establecimiento"].copy()
    df = df.drop(["id", "fecha_0", "curso", "establecimiento", "grupo"], axis=1)
    df = df.drop(_DROP_TOTALS, axis=1)
    # The administered CBT-SQ omits items 13 and 15 of the original instrument;
    # renamed to preserve traceability with the source numbering.
    df = df.rename(columns={"cbt_13_0": "cbt_14_0", "cbt_14_0": "cbt_16_0"})
    df = df.replace(",", ".", regex=True).replace("#¡NULO!", float("nan"))
    df = df.drop([f"siq_{i}_0" for i in range(1, 16)], axis=1)
    df = df.astype(float).dropna()

    clusters = clusters_raw.loc[df.index].astype(str)
    X = df.drop("siq_total_0", axis=1)
    siq = df["siq_total_0"] - SIQ_OFFSET
    y = siq >= SIQ_THRESHOLD
    return X, y, siq, clusters


def split(X, y, siq, clusters):
    """Stratified 70/30 split at participant level (as in the manuscript)."""
    return train_test_split(X, y, siq, clusters, test_size=TEST_SIZE,
                            stratify=y, random_state=RANDOM_STATE)


def columns_by_kind(X):
    psych = [c for c in X.columns
             if any(c.lower().startswith(p) for p in PSYCHOMETRIC_PREFIXES)]
    distress = [c for c in psych
                if any(c.lower().startswith(p) for p in DISTRESS_PREFIXES)]
    other = [c for c in X.columns if c not in psych]
    return {"psychometric": psych, "distress": distress, "non_psychometric": other}


def verify(verbose=True):
    """Replication test. Raises AssertionError on any deviation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y, siq, clusters = load_cohort()
    Xtr, Xte, ytr, yte, _, _, ctr, _ = split(X, y, siq, clusters)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(**L1_PARAMS).fit(scaler.transform(Xtr), ytr.astype(int))
    nz = int((clf.coef_.ravel() != 0).sum())

    got = {"n_total": len(X), "n_train": len(Xtr), "n_test": len(Xte),
           "n_severe": int(y.sum()), "n_severe_test": int(yte.sum()),
           "n_schools": int(clusters.nunique()), "n_nonzero_coefs": nz}
    failures = {k: (v, EXPECTED[k]) for k, v in got.items() if v != EXPECTED[k]}
    if verbose:
        for k, v in got.items():
            mark = "OK " if v == EXPECTED[k] else "XX "
            print(f"  [{mark}] {k:20s} got={v:<6} expected={EXPECTED[k]}")
    assert not failures, f"Replication failure: {failures}"
    return got


if __name__ == "__main__":
    print("Replication test of the canonical pipeline:")
    verify()
    print("\nAll figures replicate the manuscript.")
