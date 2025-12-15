#!/usr/bin/env python3
# coding: utf-8

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


file_name = input("Enter the input Excel file name (on your Desktop, e.g., p1_analysis.xlsx): ").strip()

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, file_name)

print(f"\n✓ Looking for file: {file_path}")
if not os.path.exists(file_path):
    print(f"\n❌ Error: File not found at {file_path}")
    print("Please make sure the file is on your Desktop and the name is correct.")
    raise SystemExit(1)

# Load all sheets
data = pd.read_excel(file_path, sheet_name=None)

# Standardize column names in every sheet (removes accidental leading/trailing spaces)
for k in list(data.keys()):
    df = data[k]
    df.columns = [str(c).strip() for c in df.columns]
    data[k] = df

print(f"Loaded {len(data)} sheets: {', '.join(list(data.keys())[:10])}{'...' if len(data) > 10 else ''}")

# Pick male/female sheets by name (exact matches or contains)
sheet_names = list(data.keys())

def find_sheet(name_candidates):
    for cand in name_candidates:
        for s in sheet_names:
            if s == cand:
                return s
    for cand in name_candidates:
        for s in sheet_names:
            if cand.lower() in s.lower():
                return s
    return None

male_sheet = find_sheet(["male", "0_male", "1_male", "2_male", "3_male", "male_text"])
female_sheet = find_sheet(["female", "0_female", "1_female", "2_female", "3_female", "female_text"])

if male_sheet is None or female_sheet is None:
    print("\n❌ Could not auto-detect male/female sheets.")
    print("Sheets found:")
    for s in sheet_names:
        print(f"  - {s}")
    print("\nRename your sheets to include 'male' and 'female', or adjust the find_sheet() candidates.")
    raise SystemExit(1)

print(f"\n✓ Using sheets:")
print(f"  male:   {male_sheet}")
print(f"  female: {female_sheet}")

male_df = data[male_sheet].copy()
female_df = data[female_sheet].copy()

# Decide metric columns from intersection of columns
exclude_cols = {"paragraph"}  # word_count removed intentionally
common_cols = [c for c in male_df.columns if c in female_df.columns]
metric_cols = [c for c in common_cols if c not in exclude_cols and not str(c).endswith("_count")]

if not metric_cols:
    print("\n❌ No metric columns found to test.")
    print(f"Common columns were: {common_cols}")
    raise SystemExit(1)

print(f"\n✓ Found {len(metric_cols)} metric columns to test")


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna()
    y = y.dropna()
    nx = x.shape[0]
    ny = y.shape[0]
    if nx <= 1 or ny <= 1:
        return 0.0
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    denom_df = nx + ny - 2
    if denom_df <= 0:
        return 0.0
    pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / denom_df)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


results = []
alpha = 0.05

for col in metric_cols:
    m = male_df[col].dropna()
    f = female_df[col].dropna()

    n_m = int(m.shape[0])
    n_f = int(f.shape[0])

    if n_m <= 1 or n_f <= 1:
        continue

    # Welch t-test (safer when variances differ)
    t_stat, p_val = stats.ttest_ind(m, f, equal_var=False)

    mean_m = float(m.mean())
    mean_f = float(f.mean())
    diff_m_minus_f = mean_m - mean_f

    results.append({
        "metric": col,
        "male_mean": mean_m,
        "female_mean": mean_f,
        "difference_male_minus_female": float(diff_m_minus_f),
        "percent_difference_vs_female": float((diff_m_minus_f / mean_f) * 100) if mean_f != 0 else 0.0,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d_male_minus_female": cohens_d(m, f),
        "male_n": n_m,
        "female_n": n_f,
        "significant_p<=0.05": bool(p_val <= alpha),
    })

results_df = pd.DataFrame(results)
if results_df.empty:
    print("\nNo metrics had enough data to test.")
    raise SystemExit(0)

# Sort by significance then absolute effect size
results_df["abs_d"] = results_df["cohens_d_male_minus_female"].abs()
results_df = results_df.sort_values(["significant_p<=0.05", "abs_d", "p_value"], ascending=[False, False, True]).drop(columns=["abs_d"])

# Master tables
mean_table = pd.DataFrame({
    "male_mean": [float(male_df[c].mean()) for c in metric_cols],
    "female_mean": [float(female_df[c].mean()) for c in metric_cols],
}, index=metric_cols)

std_table = pd.DataFrame({
    "male_std": [float(male_df[c].std(ddof=1)) for c in metric_cols],
    "female_std": [float(female_df[c].std(ddof=1)) for c in metric_cols],
}, index=metric_cols)

n_table = pd.DataFrame({
    "male_n": [int(male_df[c].dropna().shape[0]) for c in metric_cols],
    "female_n": [int(female_df[c].dropna().shape[0]) for c in metric_cols],
}, index=metric_cols)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(desktop_path, f"male_vs_female_differences_{timestamp}.xlsx")

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    mean_table.round(4).to_excel(writer, sheet_name="1_MASTER_Means")
    std_table.round(4).to_excel(writer, sheet_name="2_MASTER_StdDev")
    n_table.to_excel(writer, sheet_name="3_MASTER_N")
    results_df.round(6).to_excel(writer, sheet_name="4_TTEST_Results", index=False)

print("\n✓ Excel file created successfully!")
print(f"✓ Location: {output_file}")

sig_count = int(results_df["significant_p<=0.05"].sum())
print("\nSummary:")
print(f"  Metrics tested: {len(results_df)}")
print(f"  Significant (p <= 0.05): {sig_count}")
print("  Sheets:")
print("    1_MASTER_Means, 2_MASTER_StdDev, 3_MASTER_N, 4_TTEST_Results")
