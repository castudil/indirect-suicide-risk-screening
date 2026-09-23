# Indirect screening for adolescent suicidal risk — analysis code

Analysis code for:

> Martínez, I., Astudillo, C., & Núñez, D. *Detecting high suicidal risk in adolescents without
> asking about suicide: development and internal validation of a brief indirect screener.*
> (under review)

This repository contains **only the analysis code and the aggregate results it produces**. It
contains no participant data.

## Data availability

The individual-level data are not redistributed here. They come from the baseline phase of a
cluster-randomised trial in Chilean secondary schools (ANID/FONDECYT Regular 1210093;
ClinicalTrials.gov NCT05229302) and comprise sensitive mental health measures from minors,
including suicidal ideation. Access is available from the Faculty of Psychology, Universidad de
Talca, upon reasonable request and subject to ethics approval.

Scripts expect a file `datos.csv` in the working directory, semicolon-separated, Latin-1 encoded.
`src/questionnaire.py` documents every variable and the wording of every item, so the expected
structure is fully specified even without the data.

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python _cohort.py          # replication check: must print 7 OK lines before anything else
python p1_experiments.py   # all experiments; writes to ../results/
```

`_cohort.py` is the single source of truth for ingestion, cleaning and partitioning. Its `verify()`
asserts seven published figures (N, partition sizes, case counts, number of schools, and the 35
non-zero coefficients of the final model) and **fails loudly** if any drifts. Run it first.

## Audit trail

Every experiment records its run through `src/_audit.py`: the script and its SHA-256, the random
seed, package versions, hashes of inputs and of each artefact produced, and the numerical findings
it claims. Two files carry this:

- `results/AUDIT-LOG.md` — human-readable, one entry per run
- `results/RUN_MANIFEST.json` — machine-readable

To confirm that no result file has changed since it was generated:

```bash
cd src && python _audit.py verify
```

## Layout

```
src/
  _cohort.py                      canonical cohort: ingestion, split, replication check
  _audit.py                       audit trail (hashes, manifest, log); `verify` subcommand
  questionnaire.py                variable dictionary and item wording
  p1_experiments.py               the ten reported experiments (E1–E10)
  e2_label_noise_robustness.py    label-noise injection; source of the robustness result
  make_supplementary.py           generates the supplementary tables
results/                          aggregate outputs only; no individual-level data
```

## A note on language

All code and comments are in English. Two files are deliberate exceptions and should not be
translated: `src/questionnaire.py` and `results/selected_features_with_odds_ratios.csv` carry the
verbatim Spanish wording of the items as administered to participants. Rendering them in English
would misrepresent what was actually asked. The English renderings used in the manuscript tables
are kept in `make_supplementary.py`.

The exploratory Jupyter notebooks used during development are not included. They are working
documents superseded by the modules above and are available from the corresponding author. The
hyperparameters selected by the Bayesian search they performed are recorded in
`results/best_hyperparameters.json` and in `_cohort.L1_PARAMS`.

## Experiments

| ID | What it does |
|---|---|
| P1-E1 | Leave-one-school-out cross-validation across 21 institutions |
| P1-E2 | Criterion-contamination sensitivity (removing BHS, BHS+CAPE) |
| P1-E3 | Decision curve analysis and number needed to screen |
| P1-E4 | Bootstrap selection stability (1,000 refits) |
| P1-E5 | Optimism-corrected bootstrap internal validation |
| P1-E6 | Comparison against distal risk-factor screening |
| P1-E7 | Calibration intercept, slope, ICI and Brier score |
| P1-E8 | Label-noise robustness figure |
| P1-E9 | Comparison against aggregate scale scores |
| P1-E10 | Derivation of the triage tier boundaries |

## Citation

See `CITATION.cff`.

## Licence

MIT (`LICENSE`).
