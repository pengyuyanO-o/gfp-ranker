"""Evaluation metrics: regression + ranking."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import load_config, setup_logger, get_output_dir, parse_args_config


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """NDCG@k. y_true = ground-truth relevance (brightness), y_score = predicted score."""
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    pred_order = np.argsort(-y_score)[:k]
    ideal_order = np.argsort(-y_true)[:k]

    def dcg(order):
        gains = y_true[order]
        # shift to ≥ 0
        gains = gains - gains.min() + 1e-8
        discounts = np.log2(np.arange(2, len(order) + 2))
        return (gains / discounts).sum()

    dcg_pred = dcg(pred_order)
    dcg_ideal = dcg(ideal_order)
    return dcg_pred / max(dcg_ideal, 1e-10)


def hit_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Hit@k: fraction of top-k predicted that are in true top-10%."""
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    top10pct = max(1, int(n * 0.10))
    true_top_idx = set(np.argsort(-y_true)[:top10pct])
    pred_top_idx = set(np.argsort(-y_score)[:k])
    return len(pred_top_idx & true_top_idx) / k


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, k_list=(10, 20)) -> dict:
    n = len(y_true)
    metrics = {
        "n": n,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Pearson": pearsonr(y_true, y_pred)[0] if n > 2 else float("nan"),
        "Spearman": spearmanr(y_true, y_pred)[0] if n > 2 else float("nan"),
    }
    for k in k_list:
        metrics[f"NDCG@{k}"] = ndcg_at_k(y_true, y_pred, k)
        metrics[f"Hit@{k}"] = hit_at_k(y_true, y_pred, k)

    # Top-10 predicted quality
    kk = min(10, n)
    if kk > 0:
        top10_pred_idx = np.argsort(-y_pred)[:kk]
        dataset_mean = y_true.mean()
        top10_true_mean = y_true[top10_pred_idx].mean()
        top10_true_max = y_true[top10_pred_idx].max()
        metrics["Top10_true_mean_brightness"] = top10_true_mean
        metrics["Top10_true_max_brightness"] = top10_true_max
        metrics["Top10_enrichment"] = top10_true_mean / max(abs(dataset_mean), 1e-10)

    return metrics


def main():
    args = parse_args_config("Aggregate metrics from prediction CSVs")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("evaluate", log_dir)
    results_dir = get_output_dir(cfg, "outputs", "results")

    pred_files = list(results_dir.glob("predictions_*.csv"))
    if not pred_files:
        logger.warning("No prediction files found in outputs/results/")
        return

    all_rows = []
    for pf in pred_files:
        try:
            df = pd.read_csv(pf)
        except Exception as e:
            logger.warning(f"Cannot read {pf}: {e}")
            continue
        if "delta_log_brightness" not in df.columns or "pred_delta_log_brightness" not in df.columns:
            logger.warning(f"Missing prediction columns in {pf}, skipping")
            continue
        y_true = df["delta_log_brightness"].values
        y_pred = df["pred_delta_log_brightness"].values
        metrics = compute_metrics(y_true, y_pred)
        name = pf.stem.replace("predictions_", "")
        row = {"model_split_seed": name}
        row.update(metrics)
        all_rows.append(row)
        logger.info(f"{name}:  Spearman={metrics['Spearman']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}  Hit@10={metrics['Hit@10']:.4f}")

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary.to_csv(results_dir / "metrics_summary.csv", index=False)
        logger.info(f"Metrics summary → {results_dir}/metrics_summary.csv")

        best = summary.sort_values("Spearman", ascending=False).iloc[0]
        logger.info(f"\nBest model: {best['model_split_seed']}")
        logger.info(f"  Spearman={best['Spearman']:.4f}  NDCG@10={best['NDCG@10']:.4f}  Hit@10={best['Hit@10']:.4f}")
        if "Top10_enrichment" in best:
            logger.info(f"  Top10_enrichment={best['Top10_enrichment']:.4f}")


if __name__ == "__main__":
    main()
