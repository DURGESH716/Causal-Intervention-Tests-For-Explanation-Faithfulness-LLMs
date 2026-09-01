"""
04_compute_metrics_and_figures.py
===================================
Reads every results/raw/<model>__<task>.jsonl file and produces:

  results/metrics/acr_table.csv
      Per model x task x condition: ACR%, bootstrap 95% CI, invalid/unparsable
      rate, original-answer-invalid rate (rows excluded from ACR), and
      noise-floor-adjusted ACR (paired-bootstrap difference vs reinject_original).

  results/metrics/transition_breakdown.csv
      Per model x task x condition: right->wrong / wrong->right / wrong->wrong
      / no-change-correct / no-change-incorrect percentages.

  results/metrics/mcnemar_intervention_pairs.csv
      Paired significance tests between conditions within the same model/task
      (checks whether e.g. removal > perturb > swap is actually significant.

  results/metrics/base_vs_instruct.csv
      Paired comparison between every matched base/instruct pair_id defined
      in 00_config.py, per task and condition (fixes the n=1 instruct-model
      generalization problem).

  results/figures/*.png
      ACR bar charts with 95% CI error bars, per task, split by model tier.

Usage:
    python 04_compute_metrics_and_figures.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
config = importlib.import_module("00_config")

from utils.stats import bootstrap_ci, bootstrap_paired_diff_ci, mcnemar_test, transition_breakdown


def load_all_raw():
    rows = []
    for path in sorted(config.RAW_DIR.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No raw results found in {config.RAW_DIR}. Run 03_run_experiments.py first.")
    return pd.DataFrame(rows)


def per_example_majority(df_sub: pd.DataFrame) -> pd.Series:
    """Collapse multiple seeds per example into one boolean via majority
    vote, indexed by example_index. Used for paired (McNemar) tests, which
    require exactly one observation per example."""
    return df_sub.groupby("example_index")["changed"].mean().round().astype(bool)


def build_acr_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, cond), g_all in df.groupby(["model", "task", "condition"]):
        # "changed" is undefined when the ORIGINAL answer was itself
        # unparsable -- there is no baseline to compare against, so these
        # rows are excluded from ACR rather than silently counted (every
        # invalid-original row would otherwise register as "changed" the
        # moment the intervened answer happens to parse, which biases ACR
        # upward specifically for model/task pairs with higher original-
        # invalid rates -- exactly the instruct-model mmlu_stem cases).
        g = g_all[g_all["original_answer_valid"]]
        orig_invalid_rate = 100.0 * (~g_all["original_answer_valid"]).mean()

        point, lo, hi = bootstrap_ci(g["changed"].tolist())
        invalid_rate = 100.0 * (~g["intervened_answer_valid"]).mean() if len(g) else float("nan")
        missing_expl_rate = (100.0 * g_all["explanation_missing"].mean()
                              if "explanation_missing" in g_all.columns else float("nan"))
        rows.append(dict(model=model, task=task, condition=cond,
                          n_obs=len(g), n_excluded_orig_invalid=len(g_all) - len(g),
                          acr_pct=point, ci_low=lo, ci_high=hi,
                          invalid_rate_pct=invalid_rate,
                          original_invalid_rate_pct=orig_invalid_rate,
                          explanation_missing_rate_pct=missing_expl_rate))
    out = pd.DataFrame(rows)

    # ---- Noise-floor adjustment, done correctly this time ----
    # Bootstrap the PAIRED per-example difference (condition vs reinject_original)
    # directly, instead of bootstrapping each condition independently and
    # subtracting point estimates afterward. See utils/stats.bootstrap_paired_diff_ci
    # for why the old approach produced mismatched point-estimate/CI artifacts.
    adjusted_rows = []
    for (model, task), g_mt_all in df.groupby(["model", "task"]):
        g_mt = g_mt_all[g_mt_all["original_answer_valid"]]
        baseline_g = g_mt[g_mt.condition == "reinject_original"]
        if baseline_g.empty:
            continue
        baseline_series = per_example_majority(baseline_g)
        for cond in config.CONDITIONS:
            if cond == "reinject_original":
                continue
            cond_g = g_mt[g_mt.condition == cond]
            if cond_g.empty:
                continue
            cond_series = per_example_majority(cond_g)
            common_idx = cond_series.index.intersection(baseline_series.index)
            if len(common_idx) == 0:
                continue
            point_adj, lo_adj, hi_adj = bootstrap_paired_diff_ci(
                cond_series.loc[common_idx].tolist(),
                baseline_series.loc[common_idx].tolist(),
            )
            adjusted_rows.append(dict(model=model, task=task, condition=cond,
                                       acr_adjusted_pct=point_adj,
                                       acr_adjusted_ci_low=lo_adj,
                                       acr_adjusted_ci_high=hi_adj,
                                       n_paired_for_adjustment=len(common_idx)))
    adj_df = pd.DataFrame(adjusted_rows)
    out = out.merge(adj_df, on=["model", "task", "condition"], how="left")
    return out


def build_transition_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, cond), g in df.groupby(["model", "task", "condition"]):
        g_valid = g[g["intervened_answer_valid"] & g["original_answer_valid"]]
        if len(g_valid) == 0:
            continue
        breakdown = transition_breakdown(
            g_valid["gold_answer"].tolist(),
            g_valid["original_answer_norm"].tolist(),
            g_valid["intervened_answer_norm"].tolist(),
        )
        breakdown.update(model=model, task=task, condition=cond, n_valid=len(g_valid))
        rows.append(breakdown)
    return pd.DataFrame(rows)


def build_mcnemar_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    conditions = config.CONDITIONS
    for (model, task), g in df.groupby(["model", "task"]):
        per_cond = {c: per_example_majority(g[g.condition == c]) for c in conditions
                    if c in g.condition.unique()}
        conds = list(per_cond.keys())
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                a, b = conds[i], conds[j]
                common_idx = per_cond[a].index.intersection(per_cond[b].index)
                if len(common_idx) == 0:
                    continue
                res = mcnemar_test(per_cond[a].loc[common_idx].tolist(),
                                    per_cond[b].loc[common_idx].tolist())
                rows.append(dict(model=model, task=task, condition_a=a, condition_b=b,
                                  n_paired=len(common_idx), **res))
    return pd.DataFrame(rows)


def build_base_vs_instruct_table(df: pd.DataFrame) -> pd.DataFrame:
    # map model name -> pair_id / type from config
    meta = {m["name"]: m for m in config.MODELS}
    pairs = defaultdict(dict)
    for name, m in meta.items():
        pairs[m["pair_id"]][m["type"]] = name

    rows = []
    for pair_id, members in pairs.items():
        if "base" not in members or "instruct" not in members:
            continue
        base_name, instruct_name = members["base"], members["instruct"]
        for task in df["task"].unique():
            for cond in config.CONDITIONS:
                base_g = df[(df.model == base_name) & (df.task == task) & (df.condition == cond)]
                inst_g = df[(df.model == instruct_name) & (df.task == task) & (df.condition == cond)]
                if len(base_g) == 0 or len(inst_g) == 0:
                    continue
                base_series = per_example_majority(base_g)
                inst_series = per_example_majority(inst_g)
                common_idx = base_series.index.intersection(inst_series.index)
                if len(common_idx) == 0:
                    continue
                res = mcnemar_test(base_series.loc[common_idx].tolist(),
                                    inst_series.loc[common_idx].tolist())
                rows.append(dict(pair_id=pair_id, base_model=base_name,
                                  instruct_model=instruct_name, task=task, condition=cond,
                                  base_acr_pct=100 * base_series.loc[common_idx].mean(),
                                  instruct_acr_pct=100 * inst_series.loc[common_idx].mean(),
                                  n_paired=len(common_idx), **res))
    return pd.DataFrame(rows)


def make_figures(acr_table: pd.DataFrame):
    meta = {m["name"]: m for m in config.MODELS}
    for task in acr_table["task"].unique():
        for tier in ["small", "large"]:
            models_in_tier = [n for n in acr_table["model"].unique()
                               if meta.get(n, {}).get("tier") == tier]
            sub = acr_table[(acr_table.task == task) & (acr_table.model.isin(models_in_tier))
                             & (acr_table.condition != "reinject_original")]
            if sub.empty:
                continue
            conditions = sorted(sub["condition"].unique())
            models_sorted = sorted(sub["model"].unique())

            fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(models_sorted)), 5))
            width = 0.8 / len(conditions)
            x = range(len(models_sorted))
            bar_records = []  # (rects, vals, errs_low, errs_high) per condition, labeled after full draw
            for ci, cond in enumerate(conditions):
                vals, errs_low, errs_high = [], [], []
                for model in models_sorted:
                    row = sub[(sub.model == model) & (sub.condition == cond)]
                    if row.empty or pd.isna(row.iloc[0].get("acr_adjusted_pct")):
                        vals.append(0); errs_low.append(0); errs_high.append(0); continue
                    r = row.iloc[0]
                    point = r["acr_adjusted_pct"]
                    vals.append(point)
                    errs_low.append(max(0, point - r["acr_adjusted_ci_low"]))
                    errs_high.append(max(0, r["acr_adjusted_ci_high"] - point))
                positions = [xi + ci * width for xi in x]
                bars = ax.bar(positions, vals, width=width, label=cond,
                               yerr=[errs_low, errs_high], capsize=2)
                bar_records.append((bars, vals, errs_low, errs_high))

            # --- value labels in %, added only after every bar/whisker in
            # the chart is drawn so the y-axis is at its final scale. Labels
            # sit past whichever whisker tip matches the bar's sign (above
            # the top whisker for positive bars, below the bottom whisker
            # for negative bars) so text never collides with an error bar
            # cap. Horizontal text throughout, no rotation. ---
            y_lo, y_hi = ax.get_ylim()
            pad = 0.02 * (y_hi - y_lo) if y_hi != y_lo else 1.0
            for bars, vals, errs_low, errs_high in bar_records:
                for rect, v, err_lo, err_hi in zip(bars, vals, errs_low, errs_high):
                    if v >= 0:
                        label_y = rect.get_height() + err_hi + pad
                        va = "bottom"
                    else:
                        label_y = rect.get_height() - err_lo - pad
                        va = "top"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        label_y,
                        f"{v:.1f}%",
                        ha="center", va=va,
                        fontsize=6, rotation=0,
                    )

            ax.set_xticks([xi + width * len(conditions) / 2 for xi in x])
            ax.set_xticklabels(models_sorted, rotation=30, ha="right")
            ax.set_ylabel("Noise-floor-adjusted ACR (%)")
            ax.set_title(f"{task} -- {tier} tier")
            ax.legend(fontsize=7, ncol=2)
            ax.margins(y=0.15)  # extra headroom so labels above/below whiskers don't clip
            fig.tight_layout()
            out_path = config.FIGURES_DIR / f"acr_{task}_{tier}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved figure -> {out_path}")


def main():
    df = load_all_raw()

    acr_table = build_acr_table(df)
    acr_table.to_csv(config.METRICS_DIR / "acr_table.csv", index=False)
    print(f"Wrote {config.METRICS_DIR / 'acr_table.csv'}")

    transitions = build_transition_table(df)
    transitions.to_csv(config.METRICS_DIR / "transition_breakdown.csv", index=False)
    print(f"Wrote {config.METRICS_DIR / 'transition_breakdown.csv'}")

    mcnemar_table = build_mcnemar_table(df)
    mcnemar_table.to_csv(config.METRICS_DIR / "mcnemar_intervention_pairs.csv", index=False)
    print(f"Wrote {config.METRICS_DIR / 'mcnemar_intervention_pairs.csv'}")

    base_vs_instruct = build_base_vs_instruct_table(df)
    base_vs_instruct.to_csv(config.METRICS_DIR / "base_vs_instruct.csv", index=False)
    print(f"Wrote {config.METRICS_DIR / 'base_vs_instruct.csv'}")

    make_figures(acr_table)
    print("\nAll metrics and figures written.")


if __name__ == "__main__":
    main()
