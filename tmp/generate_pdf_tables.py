"""
Generates results/aggregated_metrics_table.pdf with F1, BAC, AUC summary tables
and a full Table 4 with all Algorithm × Threshold combinations.
Usage: python3 generate_pdf_tables.py
"""

import glob
import pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ---------------------------------
# Load data
# ---------------------------------

files = sorted(glob.glob("results/loo_metrics_seed_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.columns = ["Dataset", "Algorithm", "Threshold", "NumFeatures", "F1", "BAC", "AUC"]

alg_map = {"DecisionTree": "DT", "KNN": "KNN", "LogisticRegression": "LR",
           "NaiveBayes": "NB", "RandomForest": "RF", "SVM_Linear": "SVM-Lin",
           "SVM_RBF": "SVM-RBF", "XGBoost": "XGBoost"}
ds_map  = {"Baseline": "Baseline", "Ela_metadataset": "ELA", "Combined": "Combined"}
alg_order = ["DT", "KNN", "LR", "NB", "RF", "SVM-Lin", "SVM-RBF", "XGBoost"]
ds_order  = ["Baseline", "ELA", "Combined"]
thr_order = [0.80, 0.85, 0.90, 0.95]

df["Algorithm"] = df["Algorithm"].map(alg_map)
df["Dataset"]   = df["Dataset"].map(ds_map)
df["Threshold"] = df["Threshold"].astype(float)

# Aggregate across seeds only (keep threshold as dimension)
agg_full = df.groupby(["Dataset", "Algorithm", "Threshold"])[["F1", "BAC", "AUC"]].agg(["mean", "std"]).round(4)
agg_full.columns = ["F1_mean", "F1_std", "BAC_mean", "BAC_std", "AUC_mean", "AUC_std"]
agg_full = agg_full.reset_index()

# Aggregate across seeds AND thresholds (for tables 1-3)
agg = df.groupby(["Dataset", "Algorithm"])[["F1", "BAC", "AUC"]].agg(["mean", "std"]).round(4)
agg.columns = ["F1_mean", "F1_std", "BAC_mean", "BAC_std", "AUC_mean", "AUC_std"]
agg = agg.reset_index()

def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"

for m in ["F1", "BAC", "AUC"]:
    agg[m] = agg.apply(lambda r: fmt(r[f"{m}_mean"], r[f"{m}_std"]), axis=1)

# ---------------------------------
# Colors & styles
# ---------------------------------

HEADER_COLOR    = colors.HexColor("#2c3e50")
SUBHEADER_COLOR = colors.HexColor("#4a6fa5")
ALT_ROW_COLOR   = colors.HexColor("#eaf1fb")
BEST_COLOR      = colors.HexColor("#d4efdf")
WHITE           = colors.white

MAIN_TITLE = ("Supplementary Material of the paper\nExploratory Landscape Analysis as "
              "Meta-Features for Hyperparameter Tuning Recommendation: A Study on SVMs")

styles = getSampleStyleSheet()
title_style = ParagraphStyle("title", parent=styles["Title"],
                              fontSize=14, spaceAfter=4, alignment=TA_CENTER,
                              textColor=colors.HexColor("#2c3e50"))
section_style = ParagraphStyle("section", parent=styles["Heading2"],
                                fontSize=11, spaceBefore=10, spaceAfter=4,
                                textColor=SUBHEADER_COLOR)
caption_style = ParagraphStyle("caption", parent=styles["Normal"],
                                fontSize=8, spaceAfter=8, alignment=TA_CENTER,
                                textColor=colors.grey)

# ---------------------------------
# Table style builders
# ---------------------------------

def make_table_style(nrows, best_coords, fontsize=8.5, header_rows=1):
    base = [
        ("BACKGROUND",    (0, 0), (-1, header_rows - 1), HEADER_COLOR),
        ("TEXTCOLOR",     (0, 0), (-1, header_rows - 1), WHITE),
        ("FONTNAME",      (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, header_rows - 1), fontsize),
        ("ALIGN",         (0, 0), (-1, header_rows - 1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, header_rows - 1), 5),
        ("TOPPADDING",    (0, 0), (-1, header_rows - 1), 5),
        ("FONTNAME",  (0, header_rows), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, header_rows), (-1, -1), fontsize),
        ("ALIGN",     (0, header_rows), (-1, -1), "CENTER"),
        ("FONTNAME",  (0, header_rows), (1, -1),  "Helvetica-Bold"),
        ("TOPPADDING",    (0, header_rows), (-1, -1), 3),
        ("BOTTOMPADDING", (0, header_rows), (-1, -1), 3),
        ("GRID",      (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 1.2, WHITE),
    ]
    for i in range(header_rows, nrows):
        if (i - header_rows) % 2 == 1:
            base.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW_COLOR))
    for (r, c) in best_coords:
        base.append(("BACKGROUND", (c, r), (c, r), BEST_COLOR))
        base.append(("FONTNAME",   (c, r), (c, r), "Helvetica-BoldOblique"))
    return TableStyle(base)

# ---------------------------------
# Tables 1-3: aggregated (alg × dataset)
# ---------------------------------

def build_metric_table(metric_col):
    pivot = agg.pivot(index="Algorithm", columns="Dataset", values=metric_col).reindex(index=alg_order)[ds_order]
    means = agg.pivot(index="Algorithm", columns="Dataset", values=f"{metric_col}_mean").reindex(index=alg_order)[ds_order]
    header = ["Algorithm"] + ds_order
    rows, best_coords = [header], set()
    for r_idx, alg in enumerate(alg_order):
        rows.append([alg] + [pivot.loc[alg, ds] for ds in ds_order])
        for c_idx, ds in enumerate(ds_order):
            if means.loc[alg, ds] == means[ds].max():
                best_coords.add((r_idx + 1, c_idx + 1))
    return rows, best_coords


# ---------------------------------
# Build PDF
# ---------------------------------

out_path = "results/aggregated_metrics_table.pdf"
doc = SimpleDocTemplate(out_path, pagesize=landscape(A4),
                        leftMargin=1.2*cm, rightMargin=1.2*cm,
                        topMargin=1.8*cm, bottomMargin=1.8*cm)

col_widths_123 = [2.8*cm, 6.2*cm, 6.2*cm, 6.2*cm]

summary_metrics = [
    ("F1",  "Table 1 — Weighted F1-Score (mean ± std across 10 seeds and 4 correlation thresholds)"),
    ("BAC", "Table 2 — Balanced Accuracy (mean ± std across 10 seeds and 4 correlation thresholds)"),
    ("AUC", "Table 3 — AUC (mean ± std across 10 seeds and 4 correlation thresholds)"),
]

story = []

# Tables 1-3
for i, (m_col, caption) in enumerate(summary_metrics):
    rows, best = build_metric_table(m_col)
    t = Table(rows, colWidths=col_widths_123, repeatRows=1, splitByRow=False)
    t.setStyle(make_table_style(len(rows), best))
    block = KeepTogether([
        Paragraph(MAIN_TITLE, title_style),
        Paragraph("Green cells indicate the best value per meta-dataset column.", caption_style),
        Spacer(1, 0.4*cm),
        Paragraph(caption, section_style),
        Spacer(1, 0.2*cm),
        t,
    ])
    story.append(block)
    story.append(PageBreak())

# Table 4 — stacked: Meta-dataset | Algorithm | τ | F1 | BAC | AUC
def build_full_table_stacked():
    header = ["Meta-dataset", "Algorithm", "τ", "F1", "BAC", "AUC"]
    rows = [header]

    # best mean per (dataset, metric)
    best_means = {}
    for ds in ds_order:
        for m in ["F1", "BAC", "AUC"]:
            sub = agg_full[agg_full["Dataset"] == ds][f"{m}_mean"]
            best_means[(ds, m)] = sub.max()

    best_coords = set()
    r_idx = 1

    for ds in ds_order:
        for alg in alg_order:
            for thr in thr_order:
                sub = agg_full[(agg_full["Algorithm"] == alg) &
                               (agg_full["Dataset"]   == ds)  &
                               (agg_full["Threshold"] == thr)]
                row = [ds, alg, f"{thr:.2f}"]
                for c_idx, m in enumerate(["F1", "BAC", "AUC"]):
                    mean_val = sub[f"{m}_mean"].values[0]
                    std_val  = sub[f"{m}_std"].values[0]
                    row.append(fmt(mean_val, std_val))
                    col = 3 + c_idx
                    if round(mean_val, 4) == round(best_means[(ds, m)], 4):
                        best_coords.add((r_idx, col))
                rows.append(row)
                r_idx += 1

    return rows, best_coords

rows4, best4 = build_full_table_stacked()

cw4 = [3.2*cm, 2.2*cm, 1.2*cm, 6.0*cm, 6.0*cm, 6.0*cm]

t4 = Table(rows4, colWidths=cw4, repeatRows=1)
style4 = make_table_style(len(rows4), best4, fontsize=8.0, header_rows=1)

style4.add("ALIGN", (0, 1), (-1, -1), "CENTER")

# Separator lines between meta-datasets
block_size = len(alg_order) * len(thr_order)
for ds_idx in range(1, len(ds_order)):
    sep_row = 1 + ds_idx * block_size
    style4.add("LINEABOVE", (0, sep_row), (-1, sep_row), 1.5, colors.HexColor("#2c3e50"))

# Alternate shading by algorithm block within each dataset
for ds_idx in range(len(ds_order)):
    ds_start = 1 + ds_idx * block_size
    for alg_idx in range(len(alg_order)):
        if alg_idx % 2 == 1:
            row_start = ds_start + alg_idx * len(thr_order)
            row_end   = row_start + len(thr_order) - 1
            style4.add("BACKGROUND", (0, row_start), (-1, row_end),
                       colors.HexColor("#f0f4f8"))

t4.setStyle(style4)

story.append(Paragraph(MAIN_TITLE, title_style))
story.append(Paragraph(
    "Table 4 — Full results for all Meta-dataset × Algorithm × Correlation Threshold combinations "
    "(mean ± std across 10 seeds). Green cells indicate the best value per meta-dataset × metric.",
    caption_style))
story.append(Spacer(1, 0.3*cm))
story.append(t4)

doc.build(story)
print(f"PDF saved to: {out_path}")
