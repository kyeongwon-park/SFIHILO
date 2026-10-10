from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FuncFormatter, PercentFormatter

try:
    from scipy.stats import ttest_rel as _scipy_ttest_rel
except Exception:
    _scipy_ttest_rel = None


ROOT = Path(__file__).resolve().parents[1]


METHOD_ORDER = ["random", "mutual_information", "boundary_lookahead"]
METHOD_LABELS = {
    "random": "Random",
    "mutual_information": "Mutual Information",
    "boundary_lookahead": "Boundary Lookahead",
}
METHOD_STYLE = {
    "random": dict(color="#4B5563", marker="o", linestyle="-"),
    "mutual_information": dict(color="#2F80ED", marker="o", linestyle="-"),
    "boundary_lookahead": dict(color="#FF6B35", marker="o", linestyle="-"),
}
RECENCY_STYLE = {
    "unweighted_recency": dict(color="#4B5563", marker="s", linestyle="-", label="Unweighted"),
    "estimated_recency": dict(color="#009E73", marker="s", linestyle="-", label="Learned Decay"),
    "oracle_recency": dict(color="#CC79A7", marker="s", linestyle="-", label="Ground-truth Decay"),
}
REWARD_MODEL_STYLE = {
    "gp_rff_reward_model": dict(color="#6A51A3", marker="o", linestyle="-", label="GP Reward Model"),
    "linear_reward_model": dict(color="#8C6D1F", marker="o", linestyle="-", label="Linear Reward Model"),
}
FEATURE_COLORS = {
    "missing_features": "#56B4E9",
    "additional_features": "#E69F00",
}

METRIC_ARRAY_KEY = {
    "preference_error_reward_gap_bin": "err_test",
    "preference_error_balanced": "err_balanced",
    "forward_rmse": "forward_rmse",
    "current_preference_error_balanced": "current_err_balanced",
}

PANEL_TITLES = {
    "A": "A. Learning curves by querying method",
    "B": "B. Effects of precision requirements on sample efficiency",
    "C": "C. Effects of dimensionality on sample efficiency",
    "D": "D. Effects of dimensionality on forward-model error",
    "E": "E. Effects of noise on sample efficiency",
    "F": "F. Effects of reward-model class\non steady-state accuracy",
    "G": "G. Effects of recency weighting\nunder nonstationarities",
    "H": "H. Effects of missing features\non preference error",
    "I": "I. Effects of additional features\non preference error",
}

FIGSIZE_FINAL = (12.4, 11.4)
TITLE_BAND_INCHES = 0.52
FIGSIZE_FINAL_TITLED = (FIGSIZE_FINAL[0], FIGSIZE_FINAL[1] + 3.0 * TITLE_BAND_INCHES)
TOP_WIDTH_RATIOS = [2.70, 2.85]
X_MAX = 1000


@dataclass
class ResultBundle:
    name: str
    path: Path
    rows: list[dict]
    metadata: dict
    arrays: dict[str, np.ndarray]

    @property
    def ok(self) -> bool:
        return bool(self.rows) or bool(self.arrays)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def load_bundle(results_dir: Path, name: str) -> ResultBundle:
    path = results_dir / name
    return ResultBundle(
        name=name,
        path=path,
        rows=_load_rows(path / "summary.csv"),
        metadata=_load_json(path / "metadata.json"),
        arrays=_load_arrays(path / "results.npz"),
    )


def _to_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _method_key_from_label(bundle: ResultBundle, label: str) -> str | None:
    want = str(label).strip().lower()
    for method in bundle.metadata.get("methods", []):
        if str(method.get("label", "")).strip().lower() == want:
            return str(method.get("key", ""))
    key = want.replace(" ", "_")
    return key if key in METHOD_LABELS else None


def _available_dimensions(bundle: ResultBundle) -> list[int]:
    dims = set()
    for row in bundle.rows:
        d = _to_int(row.get("d"))
        if d is not None:
            dims.add(d)
    for key in bundle.arrays:
        if key.startswith("d"):
            digits = []
            for ch in key[1:]:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            if digits:
                dims.add(int("".join(digits)))
    return sorted(dims)


def _select_dimension(bundle: ResultBundle, preferred: int) -> int | None:
    dims = _available_dimensions(bundle)
    if not dims:
        return None
    return int(preferred) if int(preferred) in dims else int(dims[0])


def _summary_series(bundle: ResultBundle, *, d: int, method: str, metric: str, tau=None):
    rows = []
    for row in bundle.rows:
        if _to_int(row.get("d")) != int(d):
            continue
        if str(row.get("method", "")).strip() != str(method):
            continue
        if str(row.get("metric", "")).strip() != str(metric):
            continue
        row_tau = row.get("tau", "")
        if tau is None:
            if str(row_tau).strip() != "":
                continue
        else:
            if not math.isclose(_to_float(row_tau), float(tau), rel_tol=0.0, abs_tol=1e-12):
                continue
        qc = _to_int(row.get("qc"))
        if qc is None:
            continue
        rows.append((qc, _to_float(row.get("mean")), _to_float(row.get("ci95"), 0.0)))
    rows.sort(key=lambda x: x[0])
    if not rows:
        return np.array([]), np.array([]), np.array([])
    q = np.asarray([x[0] for x in rows], dtype=float)
    mean = np.asarray([x[1] for x in rows], dtype=float)
    ci = np.asarray([x[2] for x in rows], dtype=float)
    return q, mean, ci


def _raw_prefix(d: int, method_key: str, condition_key: str | None = None) -> str:
    if condition_key:
        return f"d{int(d)}_{condition_key}_{method_key}"
    return f"d{int(d)}_{method_key}"


def _raw_metric(bundle: ResultBundle, *, d: int, method_key: str, metric: str, condition_key: str | None = None):
    array_key = METRIC_ARRAY_KEY.get(metric, metric)
    prefix = _raw_prefix(d, method_key, condition_key)
    q_key = f"{prefix}_qcs"
    y_key = f"{prefix}_{array_key}"
    if q_key not in bundle.arrays or y_key not in bundle.arrays:
        return np.array([]), np.empty((0, 0))
    return np.asarray(bundle.arrays[q_key], dtype=float), np.asarray(bundle.arrays[y_key], dtype=float)


def _raw_tau_list(bundle: ResultBundle, *, d: int, method_key: str, condition_key: str | None = None):
    key = f"{_raw_prefix(d, method_key, condition_key)}_tau_list"
    if key not in bundle.arrays:
        return np.array([])
    return np.asarray(bundle.arrays[key], dtype=float)


def _clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_size(10)


def _blank_panel(ax, title: str, message: str):
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        wrap=True,
    )


def _finite_xy(qcs, ys):
    q = np.asarray(qcs, dtype=float).reshape(-1)
    y = np.asarray(ys, dtype=float).reshape(-1)
    n = min(q.size, y.size)
    if n == 0:
        return np.array([]), np.array([])
    q, y = q[:n], y[:n]
    ok = np.isfinite(q) & np.isfinite(y)
    return q[ok], y[ok]


def _threshold_query(qcs, ys, target: float = 0.10) -> float:
    q, y = _finite_xy(qcs, ys)
    if q.size == 0:
        return np.nan
    order = np.argsort(q)
    q, y = q[order], y[order]
    if y[0] <= float(target):
        return 1.0 if float(q[0]) <= 0.0 else float(q[0])
    for idx in range(1, q.size):
        y0, y1 = float(y[idx - 1]), float(y[idx])
        if y0 > float(target) and y1 <= float(target):
            q0, q1 = float(q[idx - 1]), float(q[idx])
            if y1 == y0:
                return q1
            frac = (float(target) - y0) / (y1 - y0)
            return float(q0 + np.clip(frac, 0.0, 1.0) * (q1 - q0))
    return np.nan


def _fit_threshold_query(qcs, ys, target: float = 0.10) -> float:
    q, y = _finite_xy(qcs, ys)
    if q.size < 2:
        return np.nan

    q_fit, y_fit = q, y
    if q_fit.size < 2:
        return np.nan
    x = np.log1p(q_fit)
    try:
        slope, intercept = np.polyfit(x, y_fit, 1)
    except np.linalg.LinAlgError:
        return np.nan
    if not np.isfinite(slope):
        return np.nan
    if float(intercept) <= float(target):
        return 1.0
    if slope >= 0.0:
        return np.nan
    x_cross = (float(target) - float(intercept)) / float(slope)
    if not np.isfinite(x_cross):
        return np.nan
    q_hat = float(np.expm1(x_cross))
    if not np.isfinite(q_hat):
        return np.nan
    return max(0.0, q_hat)


def _threshold_values_from_matrix(qcs, mat, target: float):
    q = np.asarray(qcs, dtype=float).reshape(-1)
    y = np.asarray(mat, dtype=float)
    if y.ndim != 2 or q.size == 0:
        return np.array([])
    n_q = min(q.size, y.shape[0])
    q, y = q[:n_q], y[:n_q, :]
    valid_cols = np.all(np.isfinite(y), axis=0)
    y = y[:, valid_cols]
    if y.size == 0:
        return np.array([])
    vals = np.asarray([_threshold_query(q, y[:, run_idx], target) for run_idx in range(y.shape[1])], dtype=float)
    missing = np.where(~np.isfinite(vals))[0]
    for run_idx in missing:
        q_fit = _fit_threshold_query(q, y[:, run_idx], target)
        if np.isfinite(q_fit):
            vals[run_idx] = float(q_fit)
    return vals[np.isfinite(vals)]


def _trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    return np.sum((xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1]) * 0.5)


def _auc_values(qcs, mat):
    q = np.asarray(qcs, dtype=float).reshape(-1)
    y = np.asarray(mat, dtype=float)
    if y.ndim != 2 or q.size < 2:
        return np.array([])
    n_q = min(q.size, y.shape[0])
    q, y = q[:n_q], y[:n_q, :]
    vals = []
    span = float(np.nanmax(q) - np.nanmin(q))
    if span <= 0:
        return np.array([])
    for run_idx in range(y.shape[1]):
        yy = y[:, run_idx]
        ok = np.isfinite(q) & np.isfinite(yy)
        if np.count_nonzero(ok) >= 2:
            vals.append(float(_trapezoid(yy[ok], q[ok]) / span))
    return np.asarray(vals, dtype=float)


def _mean_value(values) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def _relative_change_label(mean: float, baseline: float) -> str:
    if not np.isfinite(mean) or not np.isfinite(baseline) or baseline == 0.0:
        return "n/a"
    return f"{100.0 * (mean - baseline) / baseline:+.1f}%"


def _holm_adjust_pvalues(pvalues):
    pvals = np.asarray(pvalues, dtype=float).reshape(-1)
    out = np.full(pvals.shape, np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(pvals))[0]
    if valid_idx.size == 0:
        return out
    valid = pvals[valid_idx]
    order = np.argsort(valid)
    ordered = valid[order]
    m = ordered.size
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for i, p in enumerate(ordered):
        running = max(running, (m - i) * float(p))
        adjusted[i] = min(1.0, running)
    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    out[valid_idx] = restored
    return out


def _paired_t_pvalue(vals_a, vals_b, log_values=False):
    if _scipy_ttest_rel is None:
        return np.nan
    a = np.asarray(vals_a, dtype=float).reshape(-1)
    b = np.asarray(vals_b, dtype=float).reshape(-1)
    n = min(a.size, b.size)
    if n < 2:
        return np.nan
    a, b = a[:n], b[:n]
    valid = np.isfinite(a) & np.isfinite(b)
    if log_values:
        valid &= (a > 0.0) & (b > 0.0)
    if np.count_nonzero(valid) < 2:
        return np.nan
    a, b = a[valid], b[valid]
    if log_values:
        a, b = np.log(a), np.log(b)
    if np.all(np.abs(a - b) <= 1e-15):
        return 1.0
    try:
        return float(_scipy_ttest_rel(a, b).pvalue)
    except Exception:
        return np.nan


def _pvalue_to_star_text(pvalue):
    if not np.isfinite(pvalue):
        return None
    if pvalue < 1e-3:
        return "***"
    if pvalue < 1e-2:
        return "**"
    if pvalue < 5e-2:
        return "*"
    return None


def _pairwise_stats(values_by_cat_group, categories, groups):
    pairwise = defaultdict(list)
    preferred_pairs = [
        ("mutual_information", "random"),
        ("boundary_lookahead", "random"),
        ("boundary_lookahead", "mutual_information"),
    ]
    seen = set()
    pairs = []
    for a, b in preferred_pairs:
        if a in groups and b in groups and (a, b) not in seen:
            pairs.append((a, b))
            seen.add((a, b))
    for a, b in combinations(groups, 2):
        if (a, b) not in seen and (b, a) not in seen:
            pairs.append((a, b))
            seen.add((a, b))
    for cat in categories:
        rows = []
        for group_a, group_b in pairs:
            vals_a = values_by_cat_group.get(cat, {}).get(group_a, [])
            vals_b = values_by_cat_group.get(cat, {}).get(group_b, [])
            p_t = _paired_t_pvalue(vals_a, vals_b, log_values=False)
            p_t_log = _paired_t_pvalue(vals_a, vals_b, log_values=True)
            rows.append({"group_a": group_a, "group_b": group_b, "pvalue_t": p_t, "pvalue_t_log": p_t_log})
        if rows:
            holm_t = _holm_adjust_pvalues([row["pvalue_t"] for row in rows])
            holm_t_log = _holm_adjust_pvalues([row["pvalue_t_log"] for row in rows])
            for row, p_t, p_t_log in zip(rows, holm_t, holm_t_log):
                row["pvalue_t_holm"] = float(p_t) if np.isfinite(p_t) else np.nan
                row["pvalue_t_log_holm"] = float(p_t_log) if np.isfinite(p_t_log) else np.nan
            pairwise[cat] = rows
    return pairwise


def _draw_pairwise_significance_brackets(ax, categories, groups, base, offsets, pairwise_by_cat, position="bottom", pvalue_key="pvalue_t_holm"):
    if not pairwise_by_cat:
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    offset_map = {group: offsets[idx] for idx, group in enumerate(groups)}

    for cat_idx, cat in enumerate(categories):
        sig_results = []
        for res in pairwise_by_cat.get(cat, []):
            if res["group_a"] not in offset_map or res["group_b"] not in offset_map:
                continue
            star_txt = _pvalue_to_star_text(res.get(pvalue_key, np.nan))
            if not star_txt:
                continue
            x1 = base[cat_idx] + offset_map[res["group_a"]]
            x2 = base[cat_idx] + offset_map[res["group_b"]]
            if x2 < x1:
                x1, x2 = x2, x1
            sig_results.append({"star_txt": star_txt, "x1": float(x1), "x2": float(x2)})
        if not sig_results:
            continue

        sig_results.sort(key=lambda item: (item["x2"] - item["x1"], item["x1"], item["x2"]))
        levels = []
        for item in sig_results:
            for level_idx, placed_items in enumerate(levels):
                if any(not (item["x2"] < placed["x1"] or item["x1"] > placed["x2"]) for placed in placed_items):
                    continue
                item["level_idx"] = level_idx
                placed_items.append(item)
                break
            else:
                item["level_idx"] = len(levels)
                levels.append([item])

        n_levels = max(item["level_idx"] for item in sig_results) + 1
        if position == "top":
            y_levels = [0.83, 0.89, 0.95][:n_levels] if n_levels <= 3 else np.linspace(0.80, 0.96, n_levels).tolist()
            height, text_pad = 0.018, 0.012
        else:
            y_levels = [0.18, 0.12, 0.06][:n_levels] if n_levels <= 3 else np.linspace(0.24, 0.06, n_levels).tolist()
            height, text_pad = 0.018, 0.012

        for item in sig_results:
            x1, x2 = item["x1"], item["x2"]
            y = y_levels[item["level_idx"]]
            if position == "top":
                ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], transform=trans, color="#444444", linewidth=0.85, clip_on=False)
                ax.text(0.5 * (x1 + x2), y + height + text_pad, item["star_txt"], transform=trans, ha="center", va="bottom", fontsize=7.5, color="#222222", clip_on=False)
            else:
                ax.plot([x1, x1, x2, x2], [y, y - height, y - height, y], transform=trans, color="#444444", linewidth=0.85, clip_on=False)
                ax.text(0.5 * (x1 + x2), y - height - text_pad, item["star_txt"], transform=trans, ha="center", va="top", fontsize=7.5, color="#222222", clip_on=False)


def _boxplot(ax, data, positions, colors, width=0.18, showfliers=True, label_texts=None, label_fontsize=6.8):
    cleaned = []
    cleaned_pos = []
    cleaned_colors = []
    cleaned_labels = []
    if label_texts is None:
        label_texts = [None] * len(data)
    for values, pos, color, label_text in zip(data, positions, colors, label_texts):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            cleaned.append(values)
            cleaned_pos.append(pos)
            cleaned_colors.append(color)
            cleaned_labels.append(label_text)
    if not cleaned:
        return False
    bp = ax.boxplot(
        cleaned,
        positions=cleaned_pos,
        widths=width,
        patch_artist=True,
        showmeans=True,
        showfliers=showfliers,
        medianprops=dict(color="#333333", linewidth=1.3),
        meanprops=dict(marker="s", markerfacecolor="#111111", markeredgecolor="#111111", markersize=3.4),
        whiskerprops=dict(color="#333333", linewidth=1.0),
        capprops=dict(color="#333333", linewidth=1.0),
        flierprops=dict(marker="o", markersize=2.8, markerfacecolor="none", markeredgecolor="#333333", markeredgewidth=0.7, alpha=0.62),
    )
    for patch, color in zip(bp["boxes"], cleaned_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#111111")
        patch.set_linewidth(1.3)
        patch.set_alpha(0.86)
    lower_whiskers = []
    upper_whiskers = []
    for idx in range(len(cleaned)):
        lower_whiskers.append(float(np.nanmin(bp["whiskers"][2 * idx].get_ydata())))
        upper_whiskers.append(float(np.nanmax(bp["whiskers"][2 * idx + 1].get_ydata())))
    ax._box_whisker_bounds = (lower_whiskers, upper_whiskers)
    for pos, label_text, upper in zip(cleaned_pos, cleaned_labels, upper_whiskers):
        if not label_text:
            continue
        ax.annotate(
            label_text,
            xy=(pos, upper),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color="#222222",
            clip_on=False,
        )
    return True


def _set_threshold_log_axis(ax, ymin=1.0):
    lower, upper = getattr(ax, "_box_whisker_bounds", ([], []))
    positive_upper = [float(v) for v in upper if np.isfinite(v) and float(v) > 0.0]
    if not positive_upper:
        return
    ymax = max(float(ax.get_ylim()[1]), max(positive_upper) * 1.32, float(ymin) * 1.1)
    ax.set_yscale("log")
    ax.set_ylim(float(ymin), ymax)


def _natural_unit_ymax(max_value, floor=0.10):
    if not np.isfinite(max_value) or max_value <= 0.0:
        return float(floor)
    padded = min(1.0, max(float(floor), float(max_value) * 1.05))
    step = 0.025 if padded <= 0.15 else 0.05
    return float(min(1.0, np.ceil(padded / step) * step))


def _group_offsets(n_groups: int):
    width = 0.78 / max(1, int(n_groups))
    offsets = (np.arange(int(n_groups), dtype=float) - (int(n_groups) - 1) / 2.0) * width
    return offsets, width * 0.88


def _plot_line(
    ax,
    q,
    mean,
    ci,
    *,
    label,
    color,
    marker,
    linestyle,
    percent=False,
    linewidth=1.75,
    markersize=3.3,
    alpha=0.15,
):
    if q.size == 0 or not np.any(np.isfinite(mean)):
        return False
    order = np.argsort(q)
    q, mean, ci = q[order], mean[order], ci[order]
    ax.plot(q, mean, color=color, marker=marker, linestyle=linestyle, linewidth=linewidth, markersize=markersize, label=label)
    ci = np.where(np.isfinite(ci), ci, 0.0)
    lo = mean - ci
    hi = mean + ci
    if percent:
        lo = np.clip(lo, 0.0, 1.0)
        hi = np.clip(hi, 0.0, 1.0)
    ax.fill_between(q, lo, hi, color=color, alpha=alpha, linewidth=0.0)
    return True


def _observed_threshold_crossing(qcs, ys, target: float):
    q, y = _finite_xy(qcs, ys)
    if q.size == 0:
        return np.nan
    order = np.argsort(q)
    q, y = q[order], y[order]
    if y[0] <= target:
        return float(q[0])
    for idx in range(1, q.size):
        y0, y1 = float(y[idx - 1]), float(y[idx])
        if y0 > target and y1 <= target:
            q0, q1 = float(q[idx - 1]), float(q[idx])
            if y1 == y0:
                return q1
            frac = (float(target) - y0) / (y1 - y0)
            return float(q0 + np.clip(frac, 0.0, 1.0) * (q1 - q0))
    return np.nan


def draw_panel_a(ax, stationary: ResultBundle, d: int, target_error: float):
    d = _select_dimension(stationary, d)
    if d is None:
        _blank_panel(ax, PANEL_TITLES["A"], "Missing stationary result folder.")
        return False
    used = False
    max_y_seen = float(target_error)
    for key in METHOD_ORDER:
        label = METHOD_LABELS[key]
        q, mean, ci = _summary_series(stationary, d=d, method=label, metric="preference_error_balanced")
        plotted = _plot_line(ax, q, mean, ci, label=label, percent=True, **METHOD_STYLE[key])
        used |= plotted
        if plotted:
            high = np.clip(mean + np.where(np.isfinite(ci), ci, 0.0), 0.0, 1.0)
            visible = np.isfinite(high) & (np.asarray(q, dtype=float) <= float(X_MAX))
            if np.any(visible):
                max_y_seen = max(max_y_seen, float(np.nanmax(high[visible])))
            x_cross = _observed_threshold_crossing(q, mean, target_error)
            if np.isfinite(x_cross) and x_cross <= float(X_MAX):
                ax.scatter(
                    [x_cross],
                    [target_error],
                    s=34,
                    marker="o",
                    facecolor="white",
                    edgecolor=METHOD_STYLE[key]["color"],
                    linewidth=1.25,
                    zorder=10,
                )
    if not used:
        _blank_panel(ax, PANEL_TITLES["A"], "No balanced-error learning curves found.")
        return False
    ax.axhline(target_error, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Preference prediction error rate (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(0, float(X_MAX))
    ax.set_ylim(0.0, _natural_unit_ymax(max_y_seen, floor=max(0.10, float(target_error))))
    ax.legend(frameon=False, fontsize=7.8, loc="upper right", handlelength=1.45, handletextpad=0.45, borderaxespad=0.2, labelspacing=0.28)
    _clean_axis(ax)
    return True


def draw_panel_b(ax, stationary: ResultBundle, d: int, target_error: float):
    d = _select_dimension(stationary, d)
    if d is None:
        _blank_panel(ax, PANEL_TITLES["B"], "Missing stationary result folder.")
        return False
    categories = None
    all_data, positions, colors, label_texts = [], [], [], []
    offsets, box_width = _group_offsets(len(METHOD_ORDER))
    entries = []
    values_by_cat_group = defaultdict(dict)
    for method_idx, key in enumerate(METHOD_ORDER):
        qcs, arr = _raw_metric(stationary, d=d, method_key=key, metric="preference_error_reward_gap_bin")
        taus = _raw_tau_list(stationary, d=d, method_key=key)
        if arr.ndim != 3 or taus.size == 0:
            continue
        tau_indices = list(range(1, min(taus.size, arr.shape[1])))
        if categories is None:
            categories = [_reward_gap_label(taus, idx) for idx in tau_indices]
        for cat_idx, tau_idx in enumerate(tau_indices):
            vals = _threshold_values_from_matrix(qcs, arr[:, tau_idx, :], target_error)
            entries.append((cat_idx, key, vals, float(cat_idx + 1) + offsets[method_idx], METHOD_STYLE[key]["color"]))
            values_by_cat_group[cat_idx][key] = vals
    baseline_by_cat = {}
    for cat_idx, key, vals, _, _ in entries:
        if key == "random":
            baseline_by_cat[cat_idx] = _mean_value(vals)
    for cat_idx, key, vals, pos, color in entries:
        all_data.append(vals)
        positions.append(pos)
        colors.append(color)
        label_texts.append("0.0%" if key == "random" else _relative_change_label(_mean_value(vals), baseline_by_cat.get(cat_idx, np.nan)))
    if not _boxplot(ax, all_data, positions, colors, width=box_width, showfliers=True, label_texts=label_texts):
        _blank_panel(ax, PANEL_TITLES["B"], "No reward-gap-bin raw arrays found.")
        return False
    _set_threshold_log_axis(ax, ymin=1.0)
    ax.set_xlabel("Evaluation Subset")
    ax.set_ylabel("Queries to 10% error")
    ax.set_xticks(np.arange(len(categories or []), dtype=float) + 1.0)
    ax.set_xticklabels(categories or [], fontsize=8)
    ax.set_xlim(0.4, max(1.0, float(len(categories or []))) + 0.6)
    base = np.arange(len(categories or []), dtype=float) + 1.0
    _draw_pairwise_significance_brackets(
        ax,
        list(range(len(categories or []))),
        METHOD_ORDER,
        base,
        offsets,
        _pairwise_stats(values_by_cat_group, list(range(len(categories or []))), METHOD_ORDER),
        position="bottom",
        pvalue_key="pvalue_t_holm",
    )
    _add_method_legend(ax, METHOD_ORDER, patch=True)
    _clean_axis(ax)
    return True


def _reward_gap_label(taus, idx: int) -> str:
    lo = float(taus[idx])
    if idx + 1 < len(taus):
        hi = float(taus[idx + 1])
        return rf"$\Delta r \in [{lo:g}, {hi:g})$"
    return rf"$\Delta r \in [{lo:g}, \infty)$"


def draw_panel_c(ax, stationary: ResultBundle, d_list: list[int], target_error: float):
    all_data, positions, colors, label_texts = [], [], [], []
    offsets, box_width = _group_offsets(len(METHOD_ORDER))
    dims = [d for d in d_list if d in _available_dimensions(stationary)]
    if not dims:
        dims = _available_dimensions(stationary)
    entries = []
    values_by_cat_group = defaultdict(dict)
    for d_idx, d in enumerate(dims):
        for method_idx, key in enumerate(METHOD_ORDER):
            qcs, arr = _raw_metric(stationary, d=d, method_key=key, metric="preference_error_balanced")
            vals = _threshold_values_from_matrix(qcs, arr, target_error)
            entries.append((d_idx, key, vals, float(d_idx + 1) + offsets[method_idx], METHOD_STYLE[key]["color"]))
            values_by_cat_group[d_idx][key] = vals
    baseline_by_dim = {}
    for d_idx, key, vals, _, _ in entries:
        if key == "random":
            baseline_by_dim[d_idx] = _mean_value(vals)
    for d_idx, key, vals, pos, color in entries:
        all_data.append(vals)
        positions.append(pos)
        colors.append(color)
        label_texts.append("0.0%" if key == "random" else _relative_change_label(_mean_value(vals), baseline_by_dim.get(d_idx, np.nan)))
    if not _boxplot(ax, all_data, positions, colors, width=box_width, showfliers=True, label_texts=label_texts):
        _blank_panel(ax, PANEL_TITLES["C"], "No balanced-error raw arrays found.")
        return False
    _set_threshold_log_axis(ax, ymin=1.0)
    ax.set_xlabel("Outcome dimensions")
    ax.set_ylabel("Queries to 10% error")
    ax.set_xticks(np.arange(len(dims), dtype=float) + 1.0)
    ax.set_xticklabels([f"d={d}" for d in dims])
    ax.set_xlim(0.4, max(1.0, float(len(dims))) + 0.6)
    base = np.arange(len(dims), dtype=float) + 1.0
    _draw_pairwise_significance_brackets(
        ax,
        list(range(len(dims))),
        METHOD_ORDER,
        base,
        offsets,
        _pairwise_stats(values_by_cat_group, list(range(len(dims))), METHOD_ORDER),
        position="bottom",
        pvalue_key="pvalue_t_holm",
    )
    _clean_axis(ax)
    return True


def draw_panel_d(ax, stationary: ResultBundle, d_list: list[int]):
    all_data, positions, colors, label_texts = [], [], [], []
    offsets, box_width = _group_offsets(len(METHOD_ORDER))
    dims = [d for d in d_list if d in _available_dimensions(stationary)]
    if not dims:
        dims = _available_dimensions(stationary)
    entries = []
    values_by_cat_group = defaultdict(dict)
    for d_idx, d in enumerate(dims):
        for method_idx, key in enumerate(METHOD_ORDER):
            qcs, arr = _raw_metric(stationary, d=d, method_key=key, metric="forward_rmse")
            vals = 1000.0 * _auc_values(qcs, arr)
            entries.append((d_idx, key, vals, float(d_idx + 1) + offsets[method_idx], METHOD_STYLE[key]["color"]))
            values_by_cat_group[d_idx][key] = vals
    baseline_by_dim = {}
    for d_idx, key, vals, _, _ in entries:
        if key == "random":
            baseline_by_dim[d_idx] = _mean_value(vals)
    for d_idx, key, vals, pos, color in entries:
        all_data.append(vals)
        positions.append(pos)
        colors.append(color)
        label_texts.append("0.0%" if key == "random" else _relative_change_label(_mean_value(vals), baseline_by_dim.get(d_idx, np.nan)))
    if not _boxplot(ax, all_data, positions, colors, width=box_width, showfliers=True, label_texts=label_texts):
        _blank_panel(ax, PANEL_TITLES["D"], "No forward-RMSE raw arrays found.")
        return False
    ax.set_xlabel("Outcome dimensions")
    ax.set_ylabel("Forward RMSE AUC")
    ax.set_xticks(np.arange(len(dims), dtype=float) + 1.0)
    ax.set_xticklabels([f"d={d}" for d in dims])
    ax.set_xlim(0.4, max(1.0, float(len(dims))) + 0.6)
    ax.set_ylim(bottom=0.0)
    base = np.arange(len(dims), dtype=float) + 1.0
    _draw_pairwise_significance_brackets(
        ax,
        list(range(len(dims))),
        METHOD_ORDER,
        base,
        offsets,
        _pairwise_stats(values_by_cat_group, list(range(len(dims))), METHOD_ORDER),
        position="top",
        pvalue_key="pvalue_t_holm",
    )
    _clean_axis(ax)
    return True


def draw_panel_e(ax, noisy: ResultBundle, noise_free: ResultBundle, d: int, target_error: float):
    if not noise_free.ok:
        _blank_panel(
            ax,
            PANEL_TITLES["E"],
            "Noise-free stationary results not found.\nRun configs/main_paper_noise_free.json to populate this panel.",
        )
        return False
    d_noisy = _select_dimension(noisy, d)
    d_clean = _select_dimension(noise_free, d)
    if d_noisy is None or d_clean is None:
        _blank_panel(ax, PANEL_TITLES["E"], "Missing noisy or noise-free stationary results.")
        return False
    all_data, positions, colors, label_texts = [], [], [], []
    offsets, box_width = _group_offsets(len(METHOD_ORDER))
    bundles = [("Off", noise_free, d_clean), ("On", noisy, d_noisy)]
    entries = []
    values_by_cat_group = defaultdict(dict)
    for cond_idx, (_, bundle, d_cur) in enumerate(bundles):
        for method_idx, key in enumerate(METHOD_ORDER):
            qcs, arr = _raw_metric(bundle, d=d_cur, method_key=key, metric="preference_error_balanced")
            vals = _threshold_values_from_matrix(qcs, arr, target_error)
            entries.append((cond_idx, key, vals, float(cond_idx + 1) + offsets[method_idx], METHOD_STYLE[key]["color"]))
            values_by_cat_group[cond_idx][key] = vals
    baseline_by_cond = {}
    for cond_idx, key, vals, _, _ in entries:
        if key == "random":
            baseline_by_cond[cond_idx] = _mean_value(vals)
    for cond_idx, key, vals, pos, color in entries:
        all_data.append(vals)
        positions.append(pos)
        colors.append(color)
        label_texts.append("0.0%" if key == "random" else _relative_change_label(_mean_value(vals), baseline_by_cond.get(cond_idx, np.nan)))
    if not _boxplot(ax, all_data, positions, colors, width=box_width, showfliers=True, label_texts=label_texts):
        _blank_panel(ax, PANEL_TITLES["E"], "No noise-comparison raw arrays found.")
        return False
    _set_threshold_log_axis(ax, ymin=10.0)
    ax.set_xlabel("Noise condition")
    ax.set_ylabel("Queries to 10% error")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Noise OFF", "Noise ON"])
    ax.set_xlim(0.4, 2.6)
    base = np.arange(2, dtype=float) + 1.0
    _draw_pairwise_significance_brackets(
        ax,
        [0, 1],
        METHOD_ORDER,
        base,
        offsets,
        _pairwise_stats(values_by_cat_group, [0, 1], METHOD_ORDER),
        position="bottom",
        pvalue_key="pvalue_t_holm",
    )
    _clean_axis(ax)
    return True


def draw_panel_f(ax, reward_model: ResultBundle, target_error: float):
    d = _select_dimension(reward_model, 10)
    if d is None:
        _blank_panel(ax, PANEL_TITLES["F"], "Missing reward-model-class result folder.")
        return False
    used = False
    for key in ["gp_rff_reward_model", "linear_reward_model"]:
        st = REWARD_MODEL_STYLE[key]
        if key == "gp_rff_reward_model":
            method_name = "Boundary Lookahead (GP-RFF Laplace Reward Model)"
        else:
            method_name = "Boundary Lookahead (Linear Laplace Reward Model)"
        q, mean, ci = _summary_series(reward_model, d=d, method=method_name, metric="preference_error_balanced")
        if q.size == 0:
            q, mean, ci = _summary_series(reward_model, d=d, method=st["label"], metric="preference_error_balanced")
        used |= _plot_line(ax, q, mean, ci, label=st["label"], color=st["color"], marker=st["marker"], linestyle=st["linestyle"], percent=True)
    if not used:
        _blank_panel(ax, PANEL_TITLES["F"], "No reward-model-class balanced-error curves found.")
        return False
    ax.axhline(target_error, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, float(X_MAX))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Preference prediction error rate (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    max_y = float(target_error)
    for key in ["gp_rff_reward_model", "linear_reward_model"]:
        st = REWARD_MODEL_STYLE[key]
        method_name = "Boundary Lookahead (GP-RFF Laplace Reward Model)" if key == "gp_rff_reward_model" else "Boundary Lookahead (Linear Laplace Reward Model)"
        q, mean, ci = _summary_series(reward_model, d=d, method=method_name, metric="preference_error_balanced")
        if q.size == 0:
            q, mean, ci = _summary_series(reward_model, d=d, method=st["label"], metric="preference_error_balanced")
        if q.size:
            high = np.clip(mean + np.where(np.isfinite(ci), ci, 0.0), 0.0, 1.0)
            visible = np.isfinite(high) & (np.asarray(q, dtype=float) <= float(X_MAX))
            if np.any(visible):
                max_y = max(max_y, float(np.nanmax(high[visible])))
    ax.set_ylim(0.0, _natural_unit_ymax(max_y, floor=max(0.10, float(target_error))))
    ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.5, handletextpad=0.45, borderaxespad=0.2, labelspacing=0.3)
    _clean_axis(ax)
    return True


def draw_panel_g(ax, nonstationarity: ResultBundle):
    d = _select_dimension(nonstationarity, 8)
    if d is None:
        _blank_panel(ax, PANEL_TITLES["G"], "Missing nonstationarity result folder.")
        return False
    used = False
    ymin_seen = np.inf
    ymax_seen = -np.inf
    for key, st in RECENCY_STYLE.items():
        source_label = _label_for_method_key(nonstationarity, key, st["label"])
        q, mean, ci = _summary_series(nonstationarity, d=d, method=source_label, metric="forward_rmse")
        if q.size:
            keep = np.asarray(q, dtype=float) > 0.0
            q = np.asarray(q, dtype=float)[keep]
            mean = np.asarray(mean, dtype=float)[keep]
            ci = np.asarray(ci, dtype=float)[keep]
        if q.size:
            ci_plot = np.where(np.isfinite(ci), ci, 0.0)
            low = mean - ci_plot
            high = mean + ci_plot
            finite_low = low[np.isfinite(low)]
            finite_high = high[np.isfinite(high)]
            if finite_low.size:
                ymin_seen = min(ymin_seen, float(np.nanmin(finite_low)))
            if finite_high.size:
                ymax_seen = max(ymax_seen, float(np.nanmax(finite_high)))
        used |= _plot_line(ax, q, mean, ci, label=st["label"], color=st["color"], marker=st["marker"], linestyle=st["linestyle"])
    if not used:
        _blank_panel(ax, PANEL_TITLES["G"], "No nonstationarity forward-RMSE curves found.")
        return False
    ax.set_xlim(0, float(X_MAX))
    ax.set_xlabel("Number of queries")
    ax.set_ylabel("Forward RMSE")
    if np.isfinite(ymin_seen) and np.isfinite(ymax_seen) and ymax_seen > ymin_seen:
        ypad = max((float(ymax_seen) - float(ymin_seen)) * 0.10, 1e-6)
        ax.set_ylim(float(ymin_seen) - ypad, float(ymax_seen) + ypad)
    else:
        ax.autoscale(axis="y")
    ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.5, handletextpad=0.45, borderaxespad=0.2, labelspacing=0.3)
    _clean_axis(ax)
    return True


def _label_for_method_key(bundle: ResultBundle, key: str, fallback: str) -> str:
    for method in bundle.metadata.get("methods", []):
        if str(method.get("key", "")) == key:
            return str(method.get("label", fallback))
    return fallback


def draw_panel_feature(ax, feature_bundle: ResultBundle, condition_type: str, title_key: str, y_max=None):
    rows = []
    for row in feature_bundle.rows:
        if str(row.get("condition_type", "")).strip() != condition_type:
            continue
        if str(row.get("metric", "")).strip() != "preference_error_balanced":
            continue
        if str(row.get("method", "")).strip() != "Boundary Lookahead":
            continue
        rows.append(row)
    if not rows:
        _blank_panel(ax, PANEL_TITLES[title_key], "Missing feature-misspecification results.")
        return False

    by_cond = {}
    for row in rows:
        key = row.get("condition", "")
        qc = _to_int(row.get("qc"), -1)
        if key not in by_cond or qc > by_cond[key][0]:
            by_cond[key] = (qc, row)

    xs, means, cis = [], [], []
    for _, row in sorted(by_cond.values(), key=lambda item: _to_float(item[1].get("n_misspecified"), 0.0)):
        n_miss = _to_float(row.get("n_misspecified"), 0.0)
        reference_dim = _to_float(row.get("d"), row.get("reward_outcome_dim", row.get("total_outcome_dim", 1.0)))
        xs.append(100.0 * n_miss / max(reference_dim, 1.0))
        means.append(_to_float(row.get("mean")) * 100.0)
        cis.append(_to_float(row.get("ci95"), 0.0) * 100.0)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(means, dtype=float)
    ci = np.asarray(cis, dtype=float)
    color = FEATURE_COLORS[condition_type]
    ax.plot(x, y, color=color, marker="o", linewidth=2.2, markersize=5.2, markeredgewidth=1.0)
    ax.fill_between(x, np.maximum(0.0, y - ci), y + ci, color=color, alpha=0.15, linewidth=0.0)
    ax.set_xlabel("Missing feature ratio (%)" if condition_type == "missing_features" else "Additional feature ratio (%)")
    ax.set_ylabel("Preference error after 500 queries (%)")
    if y_max is not None and np.isfinite(y_max):
        ax.set_ylim(0.0, float(y_max))
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _pos: f"{val:.2f}".rstrip("0").rstrip(".")))
    _clean_axis(ax)
    return True


def _add_method_legend(ax, method_keys, patch=False):
    handles = []
    labels = []
    for key in method_keys:
        st = METHOD_STYLE[key]
        if patch:
            handle = Patch(facecolor=st["color"], edgecolor=st["color"], alpha=0.86, label=METHOD_LABELS[key])
        else:
            handle = plt.Line2D([0], [0], color=st["color"], marker=st["marker"], linestyle=st["linestyle"], label=METHOD_LABELS[key])
        handles.append(handle)
        labels.append(METHOD_LABELS[key])
    ax.legend(handles, labels, frameon=False, fontsize=7.8, loc="upper right", handlelength=1.2, handletextpad=0.38, borderaxespad=0.15, labelspacing=0.25)


def _add_final_panel_titles(fig, axes):
    for key, title in PANEL_TITLES.items():
        ax = axes.get(key)
        if ax is None:
            continue
        ax.set_title("")
        bbox = ax.get_position()
        fig.text(
            bbox.x0,
            bbox.y1 + 0.009,
            title,
            ha="left",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="#111111",
            linespacing=1.08,
        )


def _base_final_axes_positions():
    fig = plt.figure(figsize=FIGSIZE_FINAL)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.88], hspace=0.30)
    top = outer[0].subgridspec(1, 2, width_ratios=TOP_WIDTH_RATIOS, wspace=0.22)
    mid = outer[1].subgridspec(1, 3, width_ratios=[1.81, 1.81, 1.62], wspace=0.28)
    bottom = outer[2].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.30)
    axes = {
        "A": fig.add_subplot(top[0, 0]),
        "B": fig.add_subplot(top[0, 1]),
        "C": fig.add_subplot(mid[0, 0]),
        "D": fig.add_subplot(mid[0, 1]),
        "E": fig.add_subplot(mid[0, 2]),
        "F": fig.add_subplot(bottom[0, 0]),
        "G": fig.add_subplot(bottom[0, 1]),
        "H": fig.add_subplot(bottom[0, 2]),
        "I": fig.add_subplot(bottom[0, 3]),
    }
    fig.subplots_adjust(left=0.062, right=0.992, bottom=0.068, top=0.965)
    positions = {label: ax.get_position().frozen() for label, ax in axes.items()}
    plt.close(fig)
    return positions


def _final_axes_with_titles(fig):
    base_w, base_h = FIGSIZE_FINAL
    new_h = FIGSIZE_FINAL_TITLED[1]
    base_positions = _base_final_axes_positions()
    row_shift = {
        "A": 2.0 * TITLE_BAND_INCHES,
        "B": 2.0 * TITLE_BAND_INCHES,
        "C": TITLE_BAND_INCHES,
        "D": TITLE_BAND_INCHES,
        "E": TITLE_BAND_INCHES,
        "F": 0.0,
        "G": 0.0,
        "H": 0.0,
        "I": 0.0,
    }
    axes = {}
    for label, bbox in base_positions.items():
        y0_inches = bbox.y0 * base_h + row_shift.get(label, 0.0)
        axes[label] = fig.add_axes([
            bbox.x0,
            y0_inches / new_h,
            bbox.width,
            bbox.height * base_h / new_h,
        ])
    return axes


def _feature_ymax(feature_bundle: ResultBundle):
    highs = []
    for condition_type in ("missing_features", "additional_features"):
        for row in feature_bundle.rows:
            if str(row.get("condition_type", "")).strip() != condition_type:
                continue
            if str(row.get("metric", "")).strip() != "preference_error_balanced":
                continue
            if str(row.get("method", "")).strip() != "Boundary Lookahead":
                continue
            highs.append(100.0 * (_to_float(row.get("mean")) + _to_float(row.get("ci95"), 0.0)))
    if not highs:
        return None
    return float(np.nanmax(highs)) * 1.10


def build_composite(args, bundles: dict[str, ResultBundle]):
    fig = plt.figure(figsize=FIGSIZE_FINAL_TITLED)
    axes = _final_axes_with_titles(fig)
    feature_ymax = _feature_ymax(bundles["feature"])
    draw_panel_a(axes["A"], bundles["stationary"], args.d, args.target_error)
    draw_panel_b(axes["B"], bundles["stationary"], args.d, args.target_error)
    draw_panel_c(axes["C"], bundles["stationary"], args.rmse_d_list, args.target_error)
    draw_panel_d(axes["D"], bundles["stationary"], args.rmse_d_list)
    draw_panel_e(axes["E"], bundles["stationary"], bundles["noise_free"], args.d, args.target_error)
    draw_panel_f(axes["F"], bundles["reward_model"], args.target_error)
    draw_panel_g(axes["G"], bundles["nonstationarity"])
    draw_panel_feature(axes["H"], bundles["feature"], "missing_features", "H", y_max=feature_ymax)
    draw_panel_feature(axes["I"], bundles["feature"], "additional_features", "I", y_max=feature_ymax)
    _add_final_panel_titles(fig, axes)
    return fig


def save_figure(fig, outdir: Path, stem: str, formats: list[str]):
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")


def save_panels(args, bundles: dict[str, ResultBundle]):
    panel_defs = [
        ("figure2_panel_a_learning_curves", "A", lambda ax: draw_panel_a(ax, bundles["stationary"], args.d, args.target_error), (4.8, 3.6)),
        ("figure2_panel_b_evaluation_subset", "B", lambda ax: draw_panel_b(ax, bundles["stationary"], args.d, args.target_error), (5.2, 3.6)),
        ("figure2_panel_c_dimensionality_sample_efficiency", "C", lambda ax: draw_panel_c(ax, bundles["stationary"], args.rmse_d_list, args.target_error), (4.2, 3.6)),
        ("figure2_panel_d_dimensionality_forward_error", "D", lambda ax: draw_panel_d(ax, bundles["stationary"], args.rmse_d_list), (4.2, 3.6)),
        ("figure2_panel_e_noise", "E", lambda ax: draw_panel_e(ax, bundles["stationary"], bundles["noise_free"], args.d, args.target_error), (4.2, 3.6)),
        ("figure2_panel_f_reward_model_class", "F", lambda ax: draw_panel_f(ax, bundles["reward_model"], args.target_error), (4.2, 3.6)),
        ("figure2_panel_g_recency_weighting", "G", lambda ax: draw_panel_g(ax, bundles["nonstationarity"]), (4.2, 3.6)),
        ("figure2_panel_h_missing_features", "H", lambda ax: draw_panel_feature(ax, bundles["feature"], "missing_features", "H"), (4.2, 3.6)),
        ("figure2_panel_i_additional_features", "I", lambda ax: draw_panel_feature(ax, bundles["feature"], "additional_features", "I"), (4.2, 3.6)),
    ]
    for stem, title_key, draw, figsize in panel_defs:
        fig, ax = plt.subplots(figsize=figsize)
        draw(ax)
        ax.set_title(PANEL_TITLES[title_key], loc="left", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, args.output_dir, stem, args.formats)
        plt.close(fig)


def _parse_d_list(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [4, 8, 12]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot the paper Figure 2 panels from SFIHILO result folders.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="Directory containing experiment result subfolders.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures", help="Directory for generated figures.")
    parser.add_argument("--stationary", default="main_paper", help="Stationary main-paper result folder name.")
    parser.add_argument("--stationary-noise-free", default="main_paper_noise_free", help="Noise-free stationary result folder name.")
    parser.add_argument("--nonstationarity", default="nonstationarity", help="Nonstationarity result folder name.")
    parser.add_argument("--feature-misspecification", default="feature_misspecification", help="Feature-misspecification result folder name.")
    parser.add_argument("--reward-model-class", default="reward_model_class_ablation", help="Reward-model-class ablation result folder name.")
    parser.add_argument("--d", type=int, default=8, help="Primary dimension used for Figure 2 panels A, B, and E.")
    parser.add_argument("--rmse-d-list", type=_parse_d_list, default=[4, 8, 12], help="Comma-separated dimensions for panels C and D.")
    parser.add_argument("--target-error", type=float, default=0.10, help="Target preference error rate for sample-efficiency panels.")
    parser.add_argument("--panels", action="store_true", help="Also save each panel as a separate figure.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output formats, for example: pdf png")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.results_dir = args.results_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    bundles = {
        "stationary": load_bundle(args.results_dir, args.stationary),
        "noise_free": load_bundle(args.results_dir, args.stationary_noise_free),
        "nonstationarity": load_bundle(args.results_dir, args.nonstationarity),
        "feature": load_bundle(args.results_dir, args.feature_misspecification),
        "reward_model": load_bundle(args.results_dir, args.reward_model_class),
    }
    for key, bundle in bundles.items():
        if not bundle.ok:
            print(f"[WARN] Missing or empty result folder for {key}: {bundle.path}")

    fig = build_composite(args, bundles)
    save_figure(fig, args.output_dir, "figure2", args.formats)
    plt.close(fig)
    if args.panels:
        save_panels(args, bundles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
