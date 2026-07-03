"""Train Ridge, ElasticNet, and optional LightGBM regressors."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from .evaluate import compute_metrics
from .features import build_single_prior, physchem_delta, single_prior_features, load_esm2_features
from .utils import load_config, setup_logger, get_output_dir, timer, parse_args_config

warnings.filterwarnings("ignore")


def load_split_data(df, split_json_path, X_all, meta_df):
    split = json.loads(Path(split_json_path).read_text())
    sid_to_row = {str(s): i for i, s in enumerate(meta_df["sample_id"])}

    def get_idx(id_list):
        idxs = [sid_to_row[str(s)] for s in id_list if str(s) in sid_to_row]
        return idxs

    train_idx = get_idx(split["train"])
    val_idx = get_idx(split.get("val", []))
    test_idx = get_idx(split["test"])
    return train_idx, val_idx, test_idx


def train_sklearn(X_train, y_train, X_val, y_val, X_test, y_test, model_name, model):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.astype(np.float64))
    X_va = scaler.transform(X_val.astype(np.float64)) if len(X_val) else X_val
    X_te = scaler.transform(X_test.astype(np.float64))

    model.fit(X_tr, y_train)
    y_pred_test = model.predict(X_te)
    metrics = compute_metrics(y_test, y_pred_test)
    return model, scaler, y_pred_test, metrics


def main():
    args = parse_args_config("Train regression baselines")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("train_regression", log_dir)
    results_dir = get_output_dir(cfg, "outputs", "results")
    ckpt_dir = get_output_dir(cfg, "outputs", "checkpoints")
    feat_dir = Path(cfg["_project_dir"]) / "outputs" / "features"
    split_dir = Path(cfg["_project_dir"]) / "outputs" / "processed" / "splits"

    # Load features
    feat_file = feat_dir / "features_all.npz"
    if not feat_file.exists():
        logger.error("features_all.npz not found. Run src.features first.")
        sys.exit(1)

    data = np.load(feat_file)
    X_all = data["X"].astype(np.float32)
    meta = pd.read_csv(feat_dir / "feature_metadata.csv")
    df_full = pd.read_csv(Path(cfg["_project_dir"]) / "outputs" / "processed" / "gfp_processed.csv")
    logger.info(f"Loaded features: {X_all.shape}, meta: {len(meta)}")

    # LightGBM availability
    lgbm_ok = False
    try:
        import lightgbm as lgb
        lgbm_ok = True
        logger.info("LightGBM available")
    except ImportError:
        logger.warning("LightGBM not available, skipping LGBM baselines")

    seeds = cfg["splits"]["seeds"][:1]  # regression: one seed per split type is enough
    split_types = ["random", "mutation_count", "leave_position"]

    for split_type in split_types:
        for seed in seeds:
            split_file = split_dir / f"{split_type}_seed{seed}.json"
            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                continue

            train_idx, val_idx, test_idx = load_split_data(df_full, split_file, X_all, meta)
            if not train_idx or not test_idx:
                logger.warning(f"Empty split for {split_type} seed={seed}")
                continue

            X_tr = X_all[train_idx]
            y_tr = meta.iloc[train_idx]["delta_log_brightness"].values
            X_va = X_all[val_idx] if val_idx else X_all[:0]
            y_va = meta.iloc[val_idx]["delta_log_brightness"].values if val_idx else np.array([])
            X_te = X_all[test_idx]
            y_te = meta.iloc[test_idx]["delta_log_brightness"].values

            logger.info(f"\n=== {split_type} seed={seed}: train={len(X_tr)} val={len(X_va)} test={len(X_te)} ===")

            tag = f"{split_type}_seed{seed}"

            # Replace NaN/inf
            X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
            X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
            X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

            for model_name, model in [
                ("ridge", Ridge(alpha=1.0)),
                ("elasticnet", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)),
            ]:
                with timer(logger, f"{model_name} {tag}"):
                    try:
                        _, _, y_pred, metrics = train_sklearn(X_tr, y_tr, X_va, y_va, X_te, y_te, model_name, model)
                        logger.info(f"  {model_name}: Spearman={metrics['Spearman']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}")

                        test_meta = meta.iloc[test_idx].copy()
                        test_meta["pred_delta_log_brightness"] = y_pred
                        test_meta["pred_log_brightness"] = test_meta["log_brightness"] + (y_pred - test_meta["delta_log_brightness"])
                        test_meta["rank_pred"] = pd.Series(y_pred).rank(ascending=False).values
                        test_meta["rank_true"] = pd.Series(y_te).rank(ascending=False).values
                        test_meta.to_csv(results_dir / f"predictions_{model_name}_{tag}.csv", index=False)
                    except Exception as e:
                        logger.warning(f"  {model_name} failed: {e}")

            if lgbm_ok:
                import lightgbm as lgb
                with timer(logger, f"lgbm_reg {tag}"):
                    try:
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr.astype(np.float64))
                        X_te_s = scaler.transform(X_te.astype(np.float64))
                        lgbm_reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                                      num_leaves=31, random_state=seed,
                                                      n_jobs=4, verbose=-1)
                        lgbm_reg.fit(X_tr_s, y_tr)
                        y_pred = lgbm_reg.predict(X_te_s)
                        metrics = compute_metrics(y_te, y_pred)
                        logger.info(f"  lgbm_reg: Spearman={metrics['Spearman']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}")
                        test_meta = meta.iloc[test_idx].copy()
                        test_meta["pred_delta_log_brightness"] = y_pred
                        test_meta["pred_log_brightness"] = test_meta["log_brightness"] + (y_pred - test_meta["delta_log_brightness"])
                        test_meta["rank_pred"] = pd.Series(y_pred).rank(ascending=False).values
                        test_meta["rank_true"] = pd.Series(y_te).rank(ascending=False).values
                        test_meta.to_csv(results_dir / f"predictions_lgbm_reg_{tag}.csv", index=False)
                    except Exception as e:
                        logger.warning(f"  lgbm_reg failed: {e}")

    logger.info("Regression training complete.")


if __name__ == "__main__":
    main()
