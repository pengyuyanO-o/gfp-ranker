"""Train MLP ranker with Huber + pairwise ranking loss."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import compute_metrics
from .features import build_single_prior
from .models import MLPRanker, CombinedLoss
from .utils import load_config, setup_logger, get_output_dir, timer, parse_args_config


def load_split_data(meta, split_json_path, X_all):
    split = json.loads(Path(split_json_path).read_text())
    sid_to_row = {str(s): i for i, s in enumerate(meta["sample_id"])}

    def get_idx(id_list):
        return [sid_to_row[str(s)] for s in id_list if str(s) in sid_to_row]

    return get_idx(split["train"]), get_idx(split.get("val", [])), get_idx(split["test"])


def encode_gfp_types(gfp_types: np.ndarray) -> torch.Tensor:
    unique = sorted(set(gfp_types))
    mapping = {t: i for i, t in enumerate(unique)}
    return torch.tensor([mapping[t] for t in gfp_types], dtype=torch.long)


def train_one_seed(X_all, meta, split_type, seed, lambda_rank, cfg, device, results_dir, ckpt_dir, logger):
    split_dir = Path(cfg["_project_dir"]) / "outputs" / "processed" / "splits"
    split_file = split_dir / f"{split_type}_seed{seed}.json"
    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return

    train_idx, val_idx, test_idx = load_split_data(meta, split_file, X_all)
    if not train_idx or not test_idx:
        logger.warning(f"Empty split: {split_type} seed={seed}")
        return

    X_tr = torch.tensor(np.nan_to_num(X_all[train_idx].astype(np.float32)), dtype=torch.float32)
    X_va = torch.tensor(np.nan_to_num(X_all[val_idx].astype(np.float32)), dtype=torch.float32) if val_idx else None
    X_te = torch.tensor(np.nan_to_num(X_all[test_idx].astype(np.float32)), dtype=torch.float32)

    y_tr = torch.tensor(meta.iloc[train_idx]["delta_log_brightness"].values, dtype=torch.float32)
    y_va = torch.tensor(meta.iloc[val_idx]["delta_log_brightness"].values, dtype=torch.float32) if val_idx else None
    y_te = meta.iloc[test_idx]["delta_log_brightness"].values

    gfp_tr = encode_gfp_types(meta.iloc[train_idx]["GFP_type"].values)
    gfp_va = encode_gfp_types(meta.iloc[val_idx]["GFP_type"].values) if val_idx else None
    gfp_te = encode_gfp_types(meta.iloc[test_idx]["GFP_type"].values)

    # Top-20% brightness mask for train
    top20_thresh = np.percentile(meta.iloc[train_idx]["Brightness"].values, 80)
    top20_mask_tr = torch.tensor(
        meta.iloc[train_idx]["Brightness"].values >= top20_thresh, dtype=torch.bool
    )

    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    wd = train_cfg["weight_decay"]
    epochs = train_cfg["epochs"]
    patience = train_cfg["early_stopping_patience"]
    dropout = train_cfg["dropout"]

    input_dim = X_tr.shape[1]
    torch.manual_seed(seed)
    model = MLPRanker(input_dim, hidden_dim=512, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CombinedLoss(lambda_rank=lambda_rank)

    # DataLoader
    dataset = TensorDataset(X_tr, y_tr, gfp_tr, top20_mask_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_spearman = -1.0
    best_epoch = 0
    patience_count = 0
    ckpt_name = f"mlp_ranker_{split_type}_seed{seed}_lambda{lambda_rank:.1f}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for Xb, yb, gtb, mb in loader:
            Xb, yb, gtb, mb = Xb.to(device), yb.to(device), gtb.to(device), mb.to(device)
            pred = model(Xb)
            loss, reg_loss, rank_loss = criterion(pred, yb, gtb, mb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validation
        if val_idx and X_va is not None:
            model.eval()
            with torch.no_grad():
                pred_va = model(X_va.to(device)).cpu().numpy()
            from scipy.stats import spearmanr
            val_sp = spearmanr(y_va.numpy(), pred_va)[0]
            if val_sp > best_val_spearman:
                best_val_spearman = val_sp
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "val_spearman": val_sp}, ckpt_path)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info(f"Early stop at epoch {epoch}, best={best_epoch} val_sp={best_val_spearman:.4f}")
                    break

        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"  epoch={epoch:4d} loss={total_loss/max(n_batches,1):.4f} val_sp={best_val_spearman:.4f}")

    # Load best
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    else:
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_te.to(device)).cpu().numpy()

    metrics = compute_metrics(y_te, y_pred)
    tag = f"{split_type}_seed{seed}_lambda{lambda_rank:.1f}"
    logger.info(f"  [{tag}] Spearman={metrics['Spearman']:.4f}  NDCG@10={metrics.get('NDCG@10',0):.4f}  Hit@10={metrics.get('Hit@10',0):.4f}")

    # Save predictions
    test_meta = meta.iloc[test_idx].copy()
    test_meta["pred_delta_log_brightness"] = y_pred
    test_meta["pred_log_brightness"] = test_meta["log_brightness"] + (y_pred - test_meta["delta_log_brightness"])
    test_meta["rank_pred"] = pd.Series(y_pred).rank(ascending=False).values
    test_meta["rank_true"] = pd.Series(y_te).rank(ascending=False).values
    test_meta.to_csv(results_dir / f"predictions_mlp_ranker_{tag}.csv", index=False)

    return metrics


def main():
    args = parse_args_config("Train MLP ranker")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("train_ranker", log_dir)
    results_dir = get_output_dir(cfg, "outputs", "results")
    ckpt_dir = get_output_dir(cfg, "outputs", "checkpoints")
    feat_dir = Path(cfg["_project_dir"]) / "outputs" / "features"

    if not cfg["environment"]["use_gpu"]:
        logger.warning("use_gpu=false; MLP will use CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg["environment"]["require_cuda"]:
        logger.error("CUDA not available.")
        sys.exit(1)
    logger.info(f"Training device: {device}")

    feat_file = feat_dir / "features_all.npz"
    if not feat_file.exists():
        logger.error("features_all.npz not found.")
        sys.exit(1)

    data = np.load(feat_file)
    X_all = data["X"].astype(np.float32)
    meta = pd.read_csv(feat_dir / "feature_metadata.csv")
    logger.info(f"Features: {X_all.shape}")

    train_cfg = cfg["training"]
    seeds = train_cfg.get("seeds", cfg["splits"]["seeds"])
    lambda_list = train_cfg.get("lambda_rank_list", [0.5])
    # For efficiency: only random split, all seeds and lambdas
    split_types = ["random"]

    for split_type in split_types:
        for seed in seeds:
            for lam in lambda_list:
                tag = f"{split_type}_seed{seed}_lambda{lam:.1f}"
                with timer(logger, f"MLP {tag}"):
                    train_one_seed(X_all, meta, split_type, seed, lam, cfg, device, results_dir, ckpt_dir, logger)

    logger.info("MLP ranker training complete.")


if __name__ == "__main__":
    main()
