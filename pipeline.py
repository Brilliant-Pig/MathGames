import os
import sys
import re
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pdfminer.high_level import extract_text
from lifelines import WeibullAFTFitter
from lifelines.utils import datetimes_to_durations

# ----------------------------
# Utilities
# ----------------------------

WORKDIR = "/workspace"
DATA_DIR = "d:\\MyFile\\比赛\\数学建模\\compition"
PDF_PATHS = [
	os.path.join(DATA_DIR, "C题.pdf"),
	os.path.join(WORKDIR, "C题.pdf"),
]
XLSX_PATHS = [
	os.path.join(DATA_DIR, "附件.xlsx"),
	os.path.join(WORKDIR, "附件.xlsx"),
]
OUT_DIR = os.path.join(WORKDIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class RiskWeights:
	alpha: float = 1.0
	beta: float = 0.5
	lambda_cost: float = 0.2

dataclass
class RiskSettings:
	limit_week: float = 28.0
	delta_margin: float = 0.0
	alpha0: float = 0.05


# ----------------------------
# PDF extraction
# ----------------------------

def extract_problem3_text(pdf_paths: List[str]) -> str:
	text = ""
	for p in pdf_paths:
		if os.path.exists(p):
			try:
				text = extract_text(p)
				break
			except Exception as e:
				print(f"Warning: failed to extract text from {p}: {e}")
	if not text:
		return ""
	# crude split to find 问题三 section
	pattern = r"问题\s*三[\s\S]*?(?=问题\s*[一二四五六七八九]|$)"
	m = re.search(pattern, text)
	return m.group(0) if m else text


# ----------------------------
# Data loading
# ----------------------------

def find_first_existing(paths: List[str]) -> Optional[str]:
	for p in paths:
		if os.path.exists(p):
			return p
	return None


def load_excel_head(path: str, n: int = 5) -> Tuple[List[str], pd.DataFrame]:
	df_iter = pd.read_excel(path, sheet_name=None, engine="openpyxl")
	# choose the first sheet by default
	sheet_name = list(df_iter.keys())[0]
	df = df_iter[sheet_name]
	cols = list(df.columns)
	return cols, df.head(n)


# ----------------------------
# Interval-censoring construction
# ----------------------------

def construct_interval_censoring(df: pd.DataFrame,
								   id_col: str,
								   gest_week_col: str,
								   y_col: str,
								   threshold: float = 4.0) -> pd.DataFrame:
	"""
	Given longitudinal rows per subject, build interval-censored targets.
	Returns a subject-level dataframe with columns: lower, upper, event, last_time.
	- left-censored: lower=0, upper=first_positive_time, event=1
	- interval-censored: lower=last_negative_time, upper=first_positive_time, event=1
	- right-censored: lower=last_time_observed, upper=+inf (np.inf), event=0
	"""
	needed = [id_col, gest_week_col, y_col]
	for c in needed:
		if c not in df.columns:
			raise ValueError(f"Missing column: {c}")
	# keep only rows with gest_week and y
	df2 = df.dropna(subset=[gest_week_col, y_col]).copy()
	# subject-level aggregation
	records = []
	for sid, g in df2.groupby(id_col):
		g = g.sort_values(gest_week_col)
		weeks = g[gest_week_col].to_numpy()
		ys = g[y_col].to_numpy()

		first_pos_idx = None
		last_neg_idx = None
		for i, v in enumerate(ys):
			if v < threshold:
				last_neg_idx = i
			if v >= threshold and first_pos_idx is None:
				first_pos_idx = i
				break

		if first_pos_idx is None:
			# never positive -> right-censored
			lower = float(weeks[-1])
			upper = np.inf
			event = 0
			last_time = float(weeks[-1])
			records.append({"id": sid, "lower": lower, "upper": upper, "event": event, "last_time": last_time})
			continue

		if last_neg_idx is None:
			# first observation already >= threshold -> left-censored
			lower = 0.0
			upper = float(weeks[first_pos_idx])
			event = 1
			last_time = float(weeks[first_pos_idx])
			records.append({"id": sid, "lower": lower, "upper": upper, "event": event, "last_time": last_time})
			continue

		# interval censored
		lower = float(weeks[last_neg_idx])
		upper = float(weeks[first_pos_idx])
		event = 1
		last_time = float(weeks[first_pos_idx])
		records.append({"id": sid, "lower": lower, "upper": upper, "event": event, "last_time": last_time})

	return pd.DataFrame.from_records(records)


# ----------------------------
# Fitting AFT with interval censoring
# ----------------------------

def fit_interval_weibull_aft(target_df: pd.DataFrame,
							   cov_df: pd.DataFrame,
							   id_col: str,
							   covariates: List[str]) -> WeibullAFTFitter:
	"""
	lifelines' WeibullAFTFitter supports interval_censoring via fit_interval_censoring.
	Requires dataframe with columns: 'lower_bound', 'upper_bound' for times.
	"""
	df = target_df.merge(cov_df[[id_col] + covariates], left_on="id", right_on=id_col, how="left")
	df = df.drop(columns=[id_col])
	# lifelines API: fit_interval_censoring(df, lower_bound_col, upper_bound_col, event_col=None, ...)
	aft = WeibullAFTFitter()
	aft.fit_interval_censoring(df,
						lower_bound_col="lower",
						upper_bound_col="upper",
						show_progress=False)
	return aft, df


# ----------------------------
# BMI grouping and t* selection
# ----------------------------

def compute_group_tstar(aft: WeibullAFTFitter,
						 group_df: pd.DataFrame,
						 covariates: List[str],
						 bmi_col: str,
						 weights: RiskWeights,
						 settings: RiskSettings) -> pd.DataFrame:
	"""
	For each BMI group, compute risk curve and t* that minimizes R(t).
	Here we approximate:
	- P(Y_true >= 4% at t) is proxied by P(T <= t) from AFT for the event 'time to threshold'.
	- Late penalty uses P(T > limit_week)
	"""
	# default coarse BMI bins; can be optimized later
	bins = [float('-inf'), 20.0, 24.0, 28.0, 32.0, 36.0, float('inf')]
	labels = ["<20", "20-24", "24-28", "28-32", "32-36", ">=36"]
	df = group_df.copy()
	df["bmi_group"] = pd.cut(df[bmi_col].astype(float), bins=bins, labels=labels, right=False)

	t_grid = np.linspace(10.0, 30.0, 161)  # 10 to 30 weeks by 0.125 week
	rows = []
	for grp, g in df.groupby("bmi_group"):
		if len(g) == 0:
			continue
		# obtain survival/cdf curves using mean covariates of the group
		mean_row = g[covariates].mean(numeric_only=True)
		newX = pd.DataFrame([mean_row.values], columns=covariates)

		cdf_vals = []
		for t in t_grid:
			cdf_t = float(1.0 - aft.predict_survival_function(newX, times=[t]).values[0, 0])
			cdf_vals.append(cdf_t)
		cdf_vals = np.clip(np.array(cdf_vals), 0.0, 1.0)

		p_reach = cdf_vals  # proxy for P(Y_true>=4%)
		p_late = float(aft.predict_survival_function(newX, times=[settings.limit_week]).values[0, 0])

		# simplistic expected re-test cost: proportional to failure prob at t
		risk_curve = weights.alpha * (1.0 - p_reach) + weights.beta * p_late + weights.lambda_cost * (1.0 - p_reach)
		idx = int(np.argmin(risk_curve))
		t_star = float(t_grid[idx])

		rows.append({
			"bmi_group": str(grp),
			"n": int(len(g)),
			"t_star_week": t_star,
			"p_reach_at_t_star": float(p_reach[idx]),
			"p_late_at_limit": float(p_late)
		})

	return pd.DataFrame(rows)


# ----------------------------
# Plotting
# ----------------------------

def plot_group_curves(aft: WeibullAFTFitter,
					  group_df: pd.DataFrame,
					  covariates: List[str],
					  bmi_col: str,
					  out_png: str,
					  settings: RiskSettings) -> None:
	sns.set(style="whitegrid")
	bins = [float('-inf'), 20.0, 24.0, 28.0, 32.0, 36.0, float('inf')]
	labels = ["<20", "20-24", "24-28", "28-32", "32-36", ">=36"]
	df = group_df.copy()
	df["bmi_group"] = pd.cut(df[bmi_col].astype(float), bins=bins, labels=labels, right=False)

	t_grid = np.linspace(10.0, 30.0, 161)
	plt.figure(figsize=(10, 6))
	for grp, g in df.groupby("bmi_group"):
		if len(g) == 0:
			continue
		mean_row = g[covariates].mean(numeric_only=True)
		newX = pd.DataFrame([mean_row.values], columns=covariates)
		cdf_vals = []
		for t in t_grid:
			cdf_t = float(1.0 - aft.predict_survival_function(newX, times=[t]).values[0, 0])
			cdf_vals.append(cdf_t)
		cdf_vals = np.clip(np.array(cdf_vals), 0.0, 1.0)
		plt.plot(t_grid, cdf_vals, label=f"BMI {grp} (n={len(g)})")

	plt.axvline(settings.limit_week, color="red", linestyle="--", label=f"limit={settings.limit_week}w")
	plt.xlabel("Gestational age (weeks)")
	plt.ylabel("P(reach >= 4%) approx. = P(T<=t)")
	plt.title("Group-wise probability to reach threshold by gestational age")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_png, dpi=160)
	plt.close()


# ----------------------------
# Main runner
# ----------------------------

def main():
	# 1) Extract problem 3 text
	pdf_path = find_first_existing(PDF_PATHS)
	problem3_txt = extract_problem3_text(PDF_PATHS) if pdf_path else ""
	with open(os.path.join(OUT_DIR, "problem3.txt"), "w", encoding="utf-8") as f:
		f.write(problem3_txt or "[No PDF found or extraction failed]")

	# 2) Inspect Excel
	xlsx_path = find_first_existing(XLSX_PATHS)
	if not xlsx_path:
		print("Excel file not found. Please place 附件.xlsx under /workspace or original path.")
		return 1
	cols, head_df = load_excel_head(xlsx_path)
	head_df.to_csv(os.path.join(OUT_DIR, "excel_head.csv"), index=False)
	with open(os.path.join(OUT_DIR, "excel_columns.json"), "w", encoding="utf-8") as f:
		json.dump(cols, f, ensure_ascii=False, indent=2)

	# 3) Load full Excel, basic cleaning
	df_iter = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
	sheet_name = list(df_iter.keys())[0]
	df = df_iter[sheet_name]

	# Column name assumptions mapping (adjust to actual columns):
	# ID: 'ID' or '病例号' etc.; Gestational week: 'J'; Y value: 'V'; BMI: 'K'
	id_col_candidates = ["ID", "样本ID", "编号", "case_id", "id"]
	gest_col_candidates = ["J", "孕周", "孕周(周)"]
	y_col_candidates = ["V", "Y浓度", "Y比例", "Y%"]
	bmi_col_candidates = ["K", "BMI"]

	def pick_col(cands):
		for c in cands:
			if c in df.columns:
				return c
		return None

	id_col = pick_col(id_col_candidates)
	gest_col = pick_col(gest_col_candidates)
	y_col = pick_col(y_col_candidates)
	bmi_col = pick_col(bmi_col_candidates)

	if id_col is None or gest_col is None or y_col is None or bmi_col is None:
		print("Required columns not found. Please adjust column mappings in pipeline.py")
		print({"id_col": id_col, "gest_col": gest_col, "y_col": y_col, "bmi_col": bmi_col})
		return 2

	# Keep rows with male fetus indication: assume male if U or V present and > 0; remove rows with missing Y
	df = df.copy()
	# Standardize potential percentage scale: if Y in [0,1], convert to %
	if df[y_col].dropna().between(0, 1).mean() > 0.9 and df[y_col].max() <= 1.0:
		df[y_col] = df[y_col] * 100.0

	# 4) Build interval-censored targets per subject
	interval_df = construct_interval_censoring(df, id_col=id_col, gest_week_col=gest_col, y_col=y_col, threshold=4.0)
	interval_df.to_csv(os.path.join(OUT_DIR, "interval_targets.csv"), index=False)

	# 5) Prepare covariates (BMI and simple QC proxies if available)
	covariates = [bmi_col]
	additional_covs = [c for c in ["年龄", "C", "D", "E", "L", "M", "N", "P", "AA"] if c in df.columns]
	covariates.extend(additional_covs)
	cov_df = df.groupby(id_col)[covariates].mean(numeric_only=True).reset_index()

	# 6) Fit Weibull AFT with interval censoring
	aft, aft_df = fit_interval_weibull_aft(interval_df, cov_df, id_col=id_col, covariates=covariates)
	aft.print_summary()
	with open(os.path.join(OUT_DIR, "aft_summary.txt"), "w", encoding="utf-8") as f:
		try:
			f.write(str(aft.summary))
		except Exception:
			f.write("[aft summary not available]")

	# 7) Group by BMI and compute t*
	weights = RiskWeights(alpha=1.0, beta=0.5, lambda_cost=0.2)
	settings = RiskSettings(limit_week=28.0, delta_margin=0.0, alpha0=0.05)
	group_results = compute_group_tstar(aft, cov_df.rename(columns={id_col: "id"}), covariates=covariates, bmi_col=bmi_col, weights=weights, settings=settings)
	group_results.to_csv(os.path.join(OUT_DIR, "group_tstar.csv"), index=False)

	# 8) Plot group curves
	plot_group_curves(aft, cov_df.rename(columns={id_col: "id"}), covariates=covariates, bmi_col=bmi_col, out_png=os.path.join(OUT_DIR, "group_curves.png"), settings=settings)

	print("Done. See outputs/ for artifacts.")
	return 0


if __name__ == "__main__":
	sys.exit(main())