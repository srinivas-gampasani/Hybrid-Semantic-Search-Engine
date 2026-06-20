"""
Visualization — Hybrid Search Engine
======================================
Generates all proof plots from real pipeline runs:
  1. Method comparison bar chart (MRR, NDCG)
  2. Per-query NDCG heatmap
  3. Score distribution scatter plot
  4. A/B test results chart
  5. Pipeline dashboard
"""

import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "BM25 Sparse":        "#E74C3C",
    "Dense Only":         "#3498DB",
    "Hybrid RRF":         "#2ECC71",
    "Hybrid + Reranker":  "#F39C12",
}


def plot_method_comparison(eval_results: dict):
    """Grouped bar chart comparing MRR and NDCG across retrieval methods."""
    methods = list(eval_results.keys())
    mrr     = [eval_results[m].mrr_at_10   for m in methods]
    ndcg5   = [eval_results[m].ndcg_at_5   for m in methods]
    ndcg10  = [eval_results[m].ndcg_at_10  for m in methods]
    p5      = [eval_results[m].precision_at_5 for m in methods]

    x     = np.arange(len(methods))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - 1.5*width, mrr,    width, label='MRR@10',    color='#3498DB', alpha=0.92)
    b2 = ax.bar(x - 0.5*width, ndcg5,  width, label='NDCG@5',    color='#2ECC71', alpha=0.92)
    b3 = ax.bar(x + 0.5*width, ndcg10, width, label='NDCG@10',   color='#E74C3C', alpha=0.92)
    b4 = ax.bar(x + 1.5*width, p5,     width, label='P@5',        color='#F39C12', alpha=0.92)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title("Retrieval Method Comparison: MRR / NDCG / Precision",
                 fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel("Retrieval Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    out = PLOT_DIR / "method_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", out)
    return str(out)


def plot_per_query_ndcg_heatmap(eval_results: dict, queries: List[dict]):
    """Heatmap of NDCG@10 scores per query per method."""
    methods   = list(eval_results.keys())
    query_ids = [q["query_id"] for q in queries]
    short_q   = [q["query"][:30] + "..." for q in queries]

    # Build matrix  [methods x queries]
    matrix = np.zeros((len(methods), len(queries)))
    for i, method in enumerate(methods):
        per_q = {pq.query_id: pq.ndcg_at_10
                 for pq in eval_results[method].per_query}
        for j, q in enumerate(queries):
            matrix[i, j] = per_q.get(q["query_id"], 0.0)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='YlGn', vmin=0, vmax=1)

    ax.set_xticks(range(len(queries)))
    ax.set_xticklabels(short_q, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)

    for i in range(len(methods)):
        for j in range(len(queries)):
            val = matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='NDCG@10')
    ax.set_title("NDCG@10 Heatmap — Per Query per Method", fontsize=13,
                 fontweight='bold', pad=12)

    out = PLOT_DIR / "ndcg_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", out)
    return str(out)


def plot_dense_vs_sparse_scores(dense_results, sparse_results, query: str):
    """Scatter: dense score vs BM25 score for each document in top results."""
    doc_ids = list({r.doc_id for r in dense_results} |
                   {r.doc_id for r in sparse_results})

    d_map = {r.doc_id: r.score for r in dense_results}
    s_map = {r.doc_id: r.score for r in sparse_results}

    # Normalize BM25 scores 0-1
    s_vals = list(s_map.values())
    s_min, s_max = min(s_vals) if s_vals else 0, max(s_vals) if s_vals else 1
    s_range = s_max - s_min if s_max != s_min else 1

    xs, ys, labels = [], [], []
    for doc_id in doc_ids:
        d = d_map.get(doc_id, 0.0)
        s = (s_map.get(doc_id, 0.0) - s_min) / s_range
        xs.append(d)
        ys.append(s)
        labels.append(doc_id)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(xs, ys, c='#3498DB', s=80, alpha=0.8, edgecolors='white', linewidths=1)

    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color='#2C3E50')

    ax.set_xlabel("Dense (Cosine Similarity)", fontsize=11)
    ax.set_ylabel("BM25 Score (normalized)", fontsize=11)
    ax.set_title(f'Dense vs Sparse Scores\nQuery: "{query[:55]}"',
                 fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')

    out = PLOT_DIR / "dense_vs_sparse_scores.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", out)
    return str(out)


def plot_ab_test_results(ab_result):
    """Bar chart showing A/B test metric comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (metric, a_val, b_val, label) in zip(axes, [
        ("MRR@10",  ab_result.mrr_a,  ab_result.mrr_b,  "MRR@10"),
        ("NDCG@10", ab_result.ndcg_a, ab_result.ndcg_b, "NDCG@10"),
    ]):
        bars = ax.bar(
            [f"A\n{ab_result.variant_a}", f"B\n{ab_result.variant_b}"],
            [a_val, b_val],
            color=["#E74C3C", "#2ECC71"], width=0.45, alpha=0.9
        )
        for bar, val in zip(bars, [a_val, b_val]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        improvement = (b_val - a_val) / a_val * 100 if a_val > 0 else 0
        sig = "✓ Significant" if ab_result.significant else "✗ Not sig."
        ax.set_title(f"{label}\n{improvement:+.1f}%  {sig}", fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(a_val, b_val) * 1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle(f"A/B Test: {ab_result.experiment_name}",
                 fontsize=13, fontweight='bold', y=1.01)
    out = PLOT_DIR / "ab_test_results.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", out)
    return str(out)


def plot_search_dashboard(eval_results: dict, ab_result=None):
    """High-level dashboard: KPI tiles + method bar + winner callout."""
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor('#F4F6F9')

    methods = list(eval_results.keys())
    best_method = max(methods, key=lambda m: eval_results[m].ndcg_at_10)
    best = eval_results[best_method]

    # ── KPI tiles ──────────────────────────────────────────────────────────
    kpis = [
        ("Methods Tested",   len(methods),               "#3498DB"),
        ("Best MRR@10",      f"{best.mrr_at_10:.3f}",    "#2ECC71"),
        ("Best NDCG@10",     f"{best.ndcg_at_10:.3f}",   "#E74C3C"),
        ("Best P@5",         f"{best.precision_at_5:.3f}","#F39C12"),
        ("Best Recall@10",   f"{best.recall_at_10:.3f}", "#9B59B6"),
        ("Winner",           best_method.split()[-1],    "#1ABC9C"),
    ]
    n_kpis = len(kpis)
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_axes([0.02 + i*(0.96/n_kpis), 0.73, 0.96/n_kpis - 0.015, 0.22])
        ax.set_facecolor(color)
        ax.text(0.5, 0.58, str(value), transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='white', ha='center', va='center')
        ax.text(0.5, 0.18, label, transform=ax.transAxes,
                fontsize=9, color='white', alpha=0.9, ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    # ── NDCG@10 method bars ────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.06, 0.08, 0.52, 0.55])
    ax_bar.set_facecolor('#F4F6F9')
    ndcg_vals = [eval_results[m].ndcg_at_10 for m in methods]
    bar_colors = ['#2ECC71' if m == best_method else '#5DADE2' for m in methods]
    bars = ax_bar.bar(methods, ndcg_vals, color=bar_colors, width=0.5, alpha=0.92)
    for bar, val in zip(bars, ndcg_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title("NDCG@10 by Retrieval Method", fontsize=12, fontweight='bold', pad=8)
    ax_bar.set_ylabel("NDCG@10", fontsize=11)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bar.tick_params(axis='x', labelsize=9)

    # ── Radar / spider for all metrics ────────────────────────────────────
    ax_r = fig.add_axes([0.62, 0.06, 0.36, 0.6], polar=True)
    metric_labels = ['MRR@10', 'NDCG@5', 'NDCG@10', 'P@5', 'R@10']
    angles = np.linspace(0, 2*np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]

    radar_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    for idx, method in enumerate(methods):
        agg = eval_results[method]
        vals = [agg.mrr_at_10, agg.ndcg_at_5, agg.ndcg_at_10,
                agg.precision_at_5, agg.recall_at_10]
        vals += vals[:1]
        c = radar_colors[idx % len(radar_colors)]
        ax_r.plot(angles, vals, 'o-', linewidth=2, color=c, label=method, alpha=0.85)
        ax_r.fill(angles, vals, alpha=0.08, color=c)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(metric_labels, fontsize=9)
    ax_r.set_ylim(0, 1)
    ax_r.set_title("Metrics Radar Chart", fontsize=11, fontweight='bold', pad=18)
    ax_r.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax_r.set_facecolor('#F4F6F9')

    fig.suptitle("Hybrid Semantic Search Engine — Evaluation Dashboard",
                 fontsize=16, fontweight='bold', y=0.98)

    out = PLOT_DIR / "search_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved: %s", out)
    return str(out)


def plot_rank_shift(hybrid_results, reranked_results, query: str):
    """Show how reranking shifts document positions."""
    hybrid_map  = {r.doc_id: r.rank for r in hybrid_results}
    rerank_map  = {r.doc_id: r.final_rank for r in reranked_results}
    all_ids     = list(rerank_map.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    for doc_id in all_ids:
        h_rank = hybrid_map.get(doc_id, len(all_ids) + 1)
        r_rank = rerank_map[doc_id]
        color = '#2ECC71' if r_rank < h_rank else '#E74C3C' if r_rank > h_rank else '#95A5A6'
        ax.annotate("", xy=(1, r_rank), xytext=(0, h_rank),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        ax.text(-0.05, h_rank, doc_id, ha='right', va='center', fontsize=8)
        ax.text(1.05, r_rank, doc_id, ha='left', va='center', fontsize=8)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(len(all_ids) + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Hybrid RRF', 'After Reranking'], fontsize=11)
    ax.set_ylabel("Rank Position", fontsize=11)
    ax.set_title(f'Rank Shift After Cross-Encoder Reranking\nQuery: "{query[:50]}"',
                 fontsize=12, fontweight='bold', pad=10)

    green_patch = mpatches.Patch(color='#2ECC71', label='Moved Up')
    red_patch   = mpatches.Patch(color='#E74C3C', label='Moved Down')
    gray_patch  = mpatches.Patch(color='#95A5A6', label='Unchanged')
    ax.legend(handles=[green_patch, red_patch, gray_patch], loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = PLOT_DIR / "rank_shift_reranking.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", out)
    return str(out)
