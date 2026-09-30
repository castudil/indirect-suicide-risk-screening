"""Empirical verification of the mapping between data columns and item content.

The interpretability claim of this study rests on being able to state, item by
item, which clinical content drives the prediction. That claim is only as good
as the correspondence between each column of the source file and the item text
the adolescent actually read. This script verifies that correspondence from the
data itself, without trusting the column labels.

Two independent checks are applied to the ERQ-CA:

1. Clean-anchor test. Each item is correlated with two anchors built from items
   that belong to the same subscale under *both* candidate scoring keys
   (suppression: 2, 4, 6; reappraisal: 1, 3, 5, 7, 10). Items 8 and 9, the ones
   under question, are excluded from both anchors, so no part-whole inflation
   can drive the result and the anchors are valid whichever key is correct.

2. Total reconstruction. The precomputed subscale totals are rebuilt by summing
   items under each candidate key and compared case by case for exact equality.

The SPSI-R check verifies whether the stored total reverses the three positively
worded items before summing.

Run: python verify_item_coding.py
"""

import re

import numpy as np
import pandas as pd

from _cohort import read_source


def load_raw(csv_path="datos.csv"):
    """Load the source file untouched.

    `_cohort.load_cohort` drops the precomputed subscale totals, which are
    exactly what the reconstruction test needs, so the raw file is read here.
    """
    df = read_source(csv_path)
    return df.replace(",", ".", regex=True).replace("#¡NULO!", float("nan")).astype(float, errors="ignore")

ERQ = [f"erq_{i}_0" for i in range(1, 11)]
SUPPRESSION_ANCHOR = [2, 4, 6]
REAPPRAISAL_ANCHOR = [1, 3, 5, 7, 10]

# Candidate scoring keys for the ERQ.
CA_KEY = {"suppression": [2, 4, 6, 8], "reappraisal": [1, 3, 5, 7, 9, 10]}
ADULT_KEY = {"suppression": [2, 4, 6, 9], "reappraisal": [1, 3, 5, 7, 8, 10]}

SPSIR = [f"rps_{i}_0" for i in range(1, 26)]
SPSIR_POSITIVE_ITEMS = [4, 5, 12]


def erq_anchor_test(df):
    d = df[ERQ].dropna()
    suppression = d[[f"erq_{i}_0" for i in SUPPRESSION_ANCHOR]].mean(axis=1)
    reappraisal = d[[f"erq_{i}_0" for i in REAPPRAISAL_ANCHOR]].mean(axis=1)
    print(f"\nERQ clean-anchor test (n = {len(d)})")
    print(f"  {'item':<8}{'r(suppression)':>16}{'r(reappraisal)':>16}  behaves as")
    for i in range(1, 11):
        x = d[f"erq_{i}_0"]
        rs, rr = x.corr(suppression), x.corr(reappraisal)
        label = "suppression" if rs > rr else "reappraisal"
        flag = "  <-- under question" if i in (8, 9) else ""
        print(f"  erq_{i:<4}{rs:>16.3f}{rr:>16.3f}  {label}{flag}")


def erq_reconstruction_test(df):
    print("\nERQ total reconstruction (exact agreement, case by case)")
    for column, subscale in [("er_es_0", "suppression"), ("erc_rc_0", "reappraisal")]:
        if column not in df.columns:
            print(f"  {column}: absent")
            continue
        sub = df[ERQ + [column]].dropna()
        for key, name in [(CA_KEY, "ERQ-CA"), (ADULT_KEY, "adult ERQ")]:
            items = key[subscale]
            total = sub[[f"erq_{i}_0" for i in items]].sum(axis=1)
            agreement = np.mean(np.isclose(total, sub[column])) * 100
            print(f"  {column} vs {name} key {str(items):<22} {agreement:6.1f}%")


def spsir_scoring_test(df):
    sub = df[SPSIR + ["rps_total_0"]].dropna()
    raw = sub[SPSIR].sum(axis=1)
    agreement = np.mean(np.isclose(raw, sub["rps_total_0"])) * 100
    print(f"\nSPSI-R total vs raw unreversed sum of 25 items: {agreement:.1f}% exact")
    print("  item-rest correlations for the positively worded items")
    for i in SPSIR_POSITIVE_ITEMS:
        x = sub[f"rps_{i}_0"]
        print(f"    rps_{i:<4}{x.corr(raw - x):>8.3f}   range [{int(x.min())}, {int(x.max())}]")


# Item families, as they are spelled in the source file.
FAMILIES = {
    "PHQ-8": (r"phq9_\d+_0", "phq9_total_0"),
    "GAD-7": (r"gad7_\d+_0", "gad7_total_0"),
    "BHS": (r"beck_\d+_0", "beck_total_0"),
    "SPSI-R": (r"rps_\d+_0", "rps_total_0"),
    "CAPE-P15": (r"cape-?p?e?15_\d+_\d+", "cape-p15_total_0"),
    "CBT-SQ": (r"cbt_\d+_0", None),
    "ERQ-CA": (r"erq_\d+_0", None),
    "SIQ-JR": (r"siq_\d+_0", "siq_total_0"),
}


def _members(df, pattern):
    return sorted((c for c in df.columns if re.fullmatch(pattern, c)),
                  key=lambda c: int(re.search(r"_(\d+)_", c).group(1)))


def instrument_sweep(df):
    """Reconstruct every stored total and check that response formats are homogeneous.

    A total that cannot be rebuilt from its items, or a family mixing response
    ranges, means the stored aggregate does not measure what its name claims.
    """
    print("\nInstrument sweep")
    print(f"  {'instrument':<11}{'items':>6}  {'numbering':<12}{'total = raw sum':>17}  response ranges")
    for name, (pattern, total) in FAMILIES.items():
        items = _members(df, pattern)
        nums = [int(re.search(r"_(\d+)_", c).group(1)) for c in items]
        numbering = f"{min(nums)}-{max(nums)}"
        if sorted(nums) != list(range(min(nums), max(nums) + 1)):
            numbering += "!"
        ranges = sorted({(df[c].min(), df[c].max()) for c in items})
        shown = ", ".join(f"{int(a)}-{int(b)}" for a, b in ranges)
        if total is None or total not in df.columns:
            agreement = "     n/a"
        else:
            sub = df[items + [total]].dropna()
            agreement = f"{np.mean(np.isclose(sub[items].sum(axis=1), sub[total])) * 100:7.1f}%"
        flag = "  <-- heterogeneous" if len(ranges) > 1 else ""
        print(f"  {name:<11}{len(items):>6}  {numbering:<12}{agreement:>17}  {shown}{flag}")


def gad7_scoring_test(df):
    """The GAD-7 severity score has seven items; the eighth is functional impairment.

    A 7-item scale scored 0-3 cannot exceed 21, so the observed maximum settles
    whether the stored total includes the impairment item.
    """
    items = [f"gad7_{i}_0" for i in range(1, 9)]
    sub = df[items + ["gad7_total_0"]].dropna()
    print("\nGAD-7 scoring")
    for k in (7, 8):
        total = sub[items[:k]].sum(axis=1)
        agreement = np.mean(np.isclose(total, sub["gad7_total_0"])) * 100
        print(f"  stored total vs sum of items 1-{k}: {agreement:5.1f}%"
              f"   max = {total.max():.0f}   mean = {total.mean():.2f}")
    print("  the GAD-7 severity score has a theoretical maximum of 21")


if __name__ == "__main__":
    df = load_raw()
    erq_anchor_test(df)
    erq_reconstruction_test(df)
    spsir_scoring_test(df)
    gad7_scoring_test(df)
    instrument_sweep(df)
