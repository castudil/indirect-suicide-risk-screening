"""
Generates paper/supplementary.tex from the audited artefacts.

S1  Exclusion protocol: every variable removed from the predictor matrix, with its rationale.
S2  Complete inventory of the 35 active predictors, with beta, OR [95% CI], significance
    and bootstrap selection frequency.

English renderings of the item content are kept in this file so that any
revision of the wording is versioned alongside the code that produces the table.

Usage:  python make_supplementary.py
"""

import pandas as pd
from _audit import Run

OUT = "../paper/supplementary.tex"

# ------------------------------------------------------------ English renderings
EN = {
    "PHQ8_Item_1": "Feeling down, depressed, irritable or hopeless",
    "PHQ8_Item_2": "Little interest or pleasure in doing things",
    "PHQ8_Item_5": "Poor appetite, weight loss, or overeating",
    "PHQ8_Item_6": "Feeling bad about oneself; feeling like a failure or having let others down",
    "PHQ8_Item_8": "Psychomotor retardation or agitation noticeable to others",
    "GAD7_Item_2": "Not being able to stop or control worrying",
    "CAPE15_Item_1": "Feeling that some people are not what they seem to be",
    "CAPE15_Item_3": "Feeling as if one is being persecuted in some way",
    "CAPE15_Item_5": "Feeling that people look at one oddly because of one's appearance",
    "CAPE15_Item_8": "Feeling as if one's thoughts were not one's own",
    "CAPE15_Item_9": "Thoughts so intense that others might overhear them",
    "CAPE15_Item_10": "Thoughts echoing or repeating in one's mind",
    "CAPE15_Item_11": "Feeling under the control of an external force or power",
    "CAPE15_Item_12": "Feeling that a double has replaced a family member or acquaintance",
    "CAPE15_Item_15": "Seeing objects, people or animals that others cannot see",
    "SPSIR_Item_5": "Belief that one's problems can be solved",
    "SPSIR_Item_6": "Waiting for problems to resolve themselves before acting",
    "BHS_Item_2": "Might as well give up, since nothing can be improved on one's own",
    "BHS_Item_3": "When things go badly, it helps to know they cannot stay that way forever",
    "BHS_Item_6": "Expecting to obtain what one cares about in the future",
    "BHS_Item_7": "The future appears dark",
    "BHS_Item_9": "Unable to make things change, with no reason to believe one ever will",
    "BHS_Item_11": "Everything ahead looks more unpleasant than pleasant",
    "BHS_Item_14": "Things do not work out the way one would want",
    "BHS_Item_16": "Never getting what one wants, so wanting anything is pointless",
    "BHS_Item_18": "The future seems vague and uncertain",
    "ERQ_Item_3": "Thinking about something else in order to feel less bad",
    "ERQ_Item_8": "Being careful not to show negative feelings",
    "CBTSQ_Item_4": "Motivating oneself by doing things",
    "CBTSQ_Item_7": "Communicating one's needs effectively",
    "CBTSQ_Item_14": "Noticing negative thought patterns as they occur",
    "BIO_Patient_Prior_Treatment": "Personal history of psychiatric or psychological treatment",
    "BIO_Family_Prior_Treatment": "Family history of psychiatric or psychological treatment",
    "BIO_Family_Attempt": "Family history of suicide attempt",
    "BIO_Family_Death": "Family bereavement due to suicide",
}
INSTRUMENT = {"PHQ8": "PHQ-8", "GAD7": "GAD-7", "CAPE15": "CAPE-P15", "SPSIR": "SPSI-R",
              "BHS": "BHS", "ERQ": "ERQ-CA", "CBTSQ": "CBT-SQ", "BIO": "Contextual"}

# Fixes two labels that the source file left as raw column names.
RELABEL = {"cape-p15_15_0": "CAPE15_Item_15", "cbt_14_0": "CBTSQ_Item_14"}
# Label -> raw column map, needed to join with selection stability.
RAW = {"PHQ8_Item_%d" % i: "phq9_%d_0" % i for i in range(1, 9)}
RAW |= {"GAD7_Item_%d" % i: "gad7_%d_0" % i for i in range(1, 8)}
RAW |= {"CAPE15_Item_%d" % i: "cape-p15_%d_0" % i for i in range(1, 16)}
RAW |= {"SPSIR_Item_%d" % i: "rps_%d_0" % i for i in range(1, 26)}
RAW |= {"BHS_Item_%d" % i: "beck_%d_0" % i for i in range(1, 21)}
RAW |= {"ERQ_Item_%d" % i: "erq_%d_0" % i for i in range(1, 11)}
RAW |= {"CBTSQ_Item_%d" % i: "cbt_%d_0" % i for i in range(1, 17)}
RAW |= {"CAPE15_Item_1": "cape-p15_1_2", "CAPE15_Item_13": "cape-pe15_13_0",
        "CBTSQ_Item_14": "cbt_14_0",
        "BIO_Patient_Prior_Treatment": "tto_previo",
        "BIO_Family_Prior_Treatment": "tto_familiar_previo",
        "BIO_Family_Attempt": "sui_familia", "BIO_Family_Death": "suicidio_familiar"}

# ----------------------------------------------------------------- S1 source data
SIQ_ITEMS = [
    "I thought it would be better if I were not alive",
    "I thought about killing myself",
    "I thought about how I would kill myself",
    "I thought about when I would kill myself",
    "I thought about people dying",
    "I thought about death",
    "I thought about what to write in a suicide note",
    "I thought about writing a will",
    "I thought about telling people I plan to kill myself",
    "I thought about how people would feel if I killed myself",
    "I wished I were dead",
    "I thought that killing myself would solve my problems",
    "I thought others would be happier if I were dead",
    "I wished I had never been born",
    "I thought no one cared whether I lived or died",
]
AGGREGATES = [
    ("phq9\\_total\\_0", "PHQ-8 total score (items 1--8)", "Aggregate of constituent items"),
    ("gad7\\_total\\_0", "GAD-7 total score", "Aggregate of constituent items"),
    ("cape-p15\\_total\\_0", "CAPE-P15 total score", "Aggregate of constituent items"),
    ("rps\\_total\\_0", "SPSI-R total score", "Aggregate of constituent items"),
    ("beck\\_total\\_0", "BHS total score", "Aggregate of constituent items"),
    ("cape-p15\\_pi\\_0", "CAPE-P15 paranoid ideation subscale", "Aggregate of constituent items"),
    ("cape-p15\\_be\\_0", "CAPE-P15 bizarre experiences subscale", "Aggregate of constituent items"),
    ("cape-p15\\_pa\\_0", "CAPE-P15 perceptual anomalies subscale", "Aggregate of constituent items"),
    ("cbt\\_ac\\_0", "CBT-SQ behavioural activation subscale", "Aggregate of constituent items"),
    ("cbt\\_rc\\_0", "CBT-SQ cognitive restructuring subscale", "Aggregate of constituent items"),
    ("erc\\_rc\\_0", "ERQ-CA cognitive reappraisal subscale", "Aggregate of constituent items"),
    ("er\\_es\\_0", "ERQ-CA expressive suppression subscale", "Aggregate of constituent items"),
]
ADMIN = [
    ("id", "Participant identifier", "Administrative; no clinical content"),
    ("fecha\\_0", "Assessment date", "Administrative; no clinical content"),
    ("curso", "Class group", "Administrative; no clinical content"),
    ("establecimiento", "School identifier",
     "Excluded from the predictor matrix; retained solely as the grouping variable for leave-one-school-out validation"),
    ("grupo", "Trial allocation arm",
     "Excluded to prevent the model from encoding trial design rather than clinical state"),
    ("gad7\\_8\\_0", "Eighth GAD-7 column",
     "Not part of the validated seven-item GAD-7 scoring; excluded"),
]


def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def build():
    orr = pd.read_csv("results/selected_features_with_odds_ratios.csv")
    orr["Variable"] = orr["Variable"].replace(RELABEL)
    stab = pd.read_csv("results/P1E4_selection_stability.csv").set_index("Variable")

    L = []
    A = L.append
    A(r"\documentclass[11pt,a4paper]{article}")
    A(r"\usepackage[margin=2.2cm]{geometry}")
    A(r"\usepackage{booktabs}\usepackage{longtable}\usepackage{array}\usepackage{amsmath}\usepackage{amsbsy}\usepackage{graphicx}")
    A(r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}")
    A(r"\renewcommand{\arraystretch}{1.12}")
    A(r"\title{Supplementary Material\\[4pt]\large Detecting high suicidal risk in adolescents"
      r" without asking about suicide: development and internal validation of a brief indirect screener}")
    A(r"\author{Mart\'inez, N\'u\~nez \& Astudillo}\date{}")
    A(r"\begin{document}\maketitle")

    # ---------------------------------------------------------------- Tabla S1
    A(r"\section*{Supplementary Table S1: Variable exclusion protocol}")
    A(r"Every variable removed from the predictor matrix, with the rationale for its removal. "
      r"Block A enforces the strictly indirect design; Block B removes derived scores that would "
      r"reintroduce collinearity with their constituent items; Block C removes administrative "
      r"fields. The analytical matrix retained after this protocol contains \textbf{108 atomic "
      r"features} (99 psychometric items and 9 contextual indicators).")
    A(r"\begin{longtable}{@{}p{3.1cm}p{7.4cm}p{5.6cm}@{}}")
    A(r"\toprule \textbf{Variable} & \textbf{Content} & \textbf{Rationale for exclusion} \\ \midrule")
    A(r"\endfirsthead \toprule \textbf{Variable} & \textbf{Content} & \textbf{Rationale} \\ \midrule \endhead")
    A(r"\multicolumn{3}{@{}l}{\textbf{Block A. Direct markers of suicidal ideation}} \\[2pt]")
    for i, txt in enumerate(SIQ_ITEMS, 1):
        A(f"\\texttt{{siq\\_{i}\\_0}} & ``{esc(txt)}'' & Direct suicidality item; constitutes the outcome \\\\")
    A(r"\texttt{siq\_total\_0} & SIQ-JR total score & Defines the outcome ($\ge 31$); never a predictor \\")
    A(r"\texttt{(not collected)} & PHQ-9 Item 9 (thoughts of being better off dead or of self-harm) & "
      r"Never administered: the battery applied the eight-item PHQ-8, which omits this item by "
      r"design. No direct suicidality response exists in the source data \\[3pt]")
    A(r"\multicolumn{3}{@{}l}{\textbf{Block B. Aggregate and subscale scores}} \\[2pt]")
    for v, c, r_ in AGGREGATES:
        A(f"\\texttt{{{v}}} & {c} & {r_}; retained at item level instead \\\\")
    A(r"\\[3pt]\multicolumn{3}{@{}l}{\textbf{Block C. Administrative and design variables}} \\[2pt]")
    for v, c, r_ in ADMIN:
        A(f"\\texttt{{{v}}} & {c} & {r_} \\\\")
    A(r"\bottomrule \end{longtable}")
    A(r"\noindent\footnotesize \textbf{Notes.} The administered CBT-SQ omits items 13 and 15 of the "
      r"original 16-item instrument; the remaining 14 items were renumbered to preserve "
      r"correspondence with the source instrument, so administered items 13 and 14 are reported "
      r"here as original items 14 and 16. The depression module applied was the PHQ-8; the "
      r"\texttt{phq9} column prefix in the source data reflects local naming convention rather "
      r"than administration of the ninth item. \normalsize")

    # ---------------------------------------------------------------- Tabla S2
    A(r"\newpage\section*{Supplementary Table S2: Complete inventory of active predictors}")
    A(r"All 35 predictors retained by the $L_1$ penalty, ranked by absolute standardized "
      r"coefficient. Odds ratios and 95\% confidence intervals derive from a 1{,}000-iteration "
      r"non-parametric bootstrap over the selected feature space. \textbf{Selection frequency} is "
      r"the proportion of 1{,}000 independent bootstrap refits of the complete pipeline in which "
      r"the feature received a non-zero coefficient; sign consistency was 100\% for every feature "
      r"with a frequency above 0.50. Features below the 0.50 threshold are reported for "
      r"completeness but should not be interpreted as reproducible findings.")
    A(r"\begin{longtable}{@{}p{3.0cm}p{6.0cm}rlcrl@{}}")
    hdr = (r"\toprule \textbf{Variable} & \textbf{Item content} & $\boldsymbol{\beta}$ & "
           r"\textbf{OR [95\% CI]} & $\boldsymbol{p<.05}$ & \textbf{Sel.\ freq.} & \textbf{Source} \\ \midrule")
    A(hdr + r" \endfirsthead " + hdr + r" \endhead")
    miss = []
    for _, r_ in orr.iterrows():
        v = r_["Variable"]
        raw = RAW.get(v)
        f = stab.loc[raw, "Selection frequency"] if raw in stab.index else None
        if f is None:
            miss.append(v)
        content = EN.get(v)
        if content is None:
            miss.append(v + " (sin traduccion)")
            content = str(r_["Description"])[:70]
        star = "" if (f is not None and f >= 0.50) else r"$^{\dagger}$"
        A(f"\\texttt{{{esc(v)}}}{star} & {esc(content)} & {r_['Coefficient']:.3f} & "
          f"{r_['Odds_Ratio']:.2f} [{r_['OR_CI_95_low']:.2f}--{r_['OR_CI_95_high']:.2f}] & "
          f"{'Yes' if r_['Significativo_95']=='Sí' else 'No'} & "
          f"{f:.3f} & {INSTRUMENT[v.split('_')[0]]} \\\\")
    A(r"\bottomrule \end{longtable}")
    A(r"\noindent\footnotesize $^{\dagger}$ Selected in fewer than 50\% of bootstrap resamples. "
      r"\normalsize")

    # ------------------------------------------------- Tables and figures S3-S6
    A(r"\newpage\section*{Supplementary Table S3: Cross-validation on the training partition}")
    A(r"Five-fold stratified cross-validation, reported for completeness alongside the three "
      r"validation procedures in the main text. Values are mean $\pm$ SD across folds.")
    A(r"\begin{center}\small\begin{tabular}{@{}lccc@{}}\toprule")
    A(r"\textbf{Model} & \textbf{Recall} & \textbf{$F_2$-Score} & \textbf{PR AUC} \\ \midrule")
    for m, r1, r2, r3 in [(r"\textbf{Logistic Regression ($L_1$)}", "0.902 \\pm 0.034", "0.798 \\pm 0.032", "0.731 \\pm 0.025"),
                          ("Random Forest", "0.893 \\pm 0.025", "0.787 \\pm 0.022", "0.714 \\pm 0.036"),
                          ("SVM", "0.878 \\pm 0.034", "0.783 \\pm 0.037", "0.727 \\pm 0.039"),
                          ("XGBoost", "0.859 \\pm 0.039", "0.780 \\pm 0.037", "0.739 \\pm 0.034"),
                          ("Gradient Boosting", "0.615 \\pm 0.024", "0.629 \\pm 0.022", "0.695 \\pm 0.023"),
                          ("KNN", "0.541 \\pm 0.084", "0.559 \\pm 0.074", "0.586 \\pm 0.038")]:
        A(f"{m} & ${r1}$ & ${r2}$ & ${r3}$ \\\\")
    A(r"\bottomrule\end{tabular}\end{center}")

    A(r"\section*{Supplementary Table S4: Performance across stepped feature subsets}")
    A(r"Hold-out performance ($N=462$) as the feature set is pruned by absolute standardized "
      r"coefficient. Discrimination is preserved down to 20 predictors, which is the basis for the "
      r"short-form instrument discussed in the main text.")
    A(r"\begin{center}\small\begin{tabular}{@{}lccccc@{}}\toprule")
    A(r"$n_{\text{features}}$ & \textbf{Precision} & \textbf{Recall} & \textbf{$F_2$} & "
      r"\textbf{Accuracy} & \textbf{AUC} \\ \midrule")
    for row in ["30 & 0.548 & 0.841 & 0.760 & 0.838 & 0.914", "28 & 0.540 & 0.841 & 0.757 & 0.833 & 0.914",
                "26 & 0.544 & 0.841 & 0.758 & 0.836 & 0.913", "24 & 0.533 & 0.830 & 0.746 & 0.829 & 0.912",
                "22 & 0.541 & 0.830 & 0.750 & 0.833 & 0.911", r"\textbf{20} & 0.522 & 0.818 & 0.735 & 0.823 & 0.911",
                "18 & 0.537 & 0.818 & 0.741 & 0.831 & 0.908", "16 & 0.533 & 0.818 & 0.739 & 0.829 & 0.905",
                "15 & 0.522 & 0.795 & 0.720 & 0.823 & 0.901"]:
        A(row + r" \\")
    A(r"\bottomrule\end{tabular}\end{center}")

    A(r"\newpage\section*{Supplementary Figure S1: Robustness to label noise}")
    A(r"\begin{center}\includegraphics[width=\linewidth]{resources/label_noise_robustness.pdf}\end{center}")
    A(r"Positive training labels were flipped to negative at increasing rates under two mechanisms, "
      r"with 100 replicates per level; evaluation was always against the uncorrupted hold-out. The "
      r"shaded band marks the 39\% inconsistency rate reported for retrospective self-report. "
      r"Discrimination (left) is invariant; the operating point (right) degrades by less than four percent.")

    A(r"\section*{Supplementary Figure S2: Reliability diagrams}")
    A(r"\begin{center}\includegraphics[width=\linewidth]{resources/calibration_comparison_all_models.pdf}\end{center}")
    A(r"Uncalibrated (top row) versus isotonic-calibrated (bottom row) risk distributions across the "
      r"three primary supervised classifiers. The dashed diagonal denotes perfect calibration. "
      r"Numerical calibration metrics are reported in the main text.")


    A(r"\newpage\section*{Supplementary Table S5: Calibration metrics}")
    A(r"Calibration of the $L_1$ logistic regression on the hold-out set before and after isotonic "
      r"regression, following the hierarchy of Van Calster et al. An intercept of 0 and a slope of 1 "
      r"denote perfect calibration.")
    A(r"\begin{center}\small\begin{tabular}{@{}lcc@{}}\toprule")
    A(r"\textbf{Metric} & \textbf{Uncalibrated} & \textbf{Isotonic} \\ \midrule")
    for m, u, c in [("Calibration intercept (in the large)", "$-1.465$", r"$\mathbf{-0.067}$"),
                    ("Calibration slope", "1.104", "0.886"),
                    ("Integrated Calibration Index (ICI)", "0.152", r"$\mathbf{0.027}$"),
                    ("Brier score", "0.1205", r"$\mathbf{0.0848}$")]:
        A(f"{m} & {u} & {c} \\\\")
    A(r"\bottomrule\end{tabular}\end{center}")

    A(r"\section*{Supplementary Figure S3: Leave-one-school-out validation}")
    A(r"\begin{center}\includegraphics[width=0.85\linewidth]{resources/loso_validation_by_school.pdf}\end{center}")
    A(r"Cross-validation across 21 educational institutions. Each point is the AUC obtained on a "
      r"school entirely excluded from training; point size is proportional to the number of "
      r"participants in that school. The size-weighted mean (0.910) closely matches the "
      r"individual-level hold-out estimate (0.915).")

    A(r"\newpage\section*{Supplementary Figure S4: Clinical decision curve analysis}")
    A(r"\begin{center}\includegraphics[width=0.9\linewidth]{resources/decision_curve_analysis.pdf}\end{center}")
    A(r"Net benefit on the hold-out set. The calibrated item-level model dominates both default "
      r"strategies across the full range of clinically plausible thresholds. Screening on distal "
      r"risk factors alone provides essentially no net benefit relative to treating all.")

    A(r"\end{document}")

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return miss, len(orr)


if __name__ == "__main__":
    r = Run("SUPPL", "Generation of the supplementary tables",
            "make_supplementary.py", 42,
            ["results/selected_features_with_odds_ratios.csv",
             "results/P1E4_selection_stability.csv", "questionnaire.py"],
            "S1 documents the exclusion protocol; S2 the complete inventory of 35 predictors "
            "with bootstrap selection frequency added.")
    miss, n = build()
    print(f"Wrote {OUT} with {n} predictors in S2.")
    print("Variables without a join or rendering:", miss or "none ✓")
    r.finding("n_predictores_S2", n)
    r.finding("variables_sin_cruce", miss or "none")
    r.output(OUT, "Supplementary tables")
    r.close()
