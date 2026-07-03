"""Extract ESM2 embeddings for WT and mutant sequences.

Saves one compressed numpy archive per GFP type instead of per-sample .pt files,
which is much faster for large datasets (avoids 141k small file writes).
"""
import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .utils import load_config, setup_logger, get_output_dir, timer, parse_args_config


def load_esm2(weight_path: str, device, logger):
    logger.info(f"Loading ESM2 from {weight_path}")
    import esm as esm_lib

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_data = torch.load(weight_path, map_location="cpu")
            model_name = Path(weight_path).stem
            model, alphabet = esm_lib.pretrained.load_model_and_alphabet_core(
                model_name, model_data, None
            )
        logger.info("ESM2 loaded via load_model_and_alphabet_core")
    except Exception as e1:
        logger.warning(f"Core load failed ({e1}), trying load_model_and_alphabet_local")
        try:
            model, alphabet = esm_lib.pretrained.load_model_and_alphabet_local(weight_path)
        except Exception as e2:
            logger.error(f"ESM2 load failed: {e1} | {e2}")
            sys.exit(1)

    model = model.eval()
    try:
        model = model.half()
        logger.info("Using fp16")
    except Exception:
        logger.warning("fp16 failed, using fp32")
    model = model.to(device)
    logger.info(f"ESM2 on {device}")
    return model, alphabet


def extract_batch(seqs_batch: list, model, alphabet, repr_layer: int, device):
    """Returns list of float32 CPU tensors [L, D]."""
    batch_converter = alphabet.get_batch_converter()
    data = [(f"s{i}", seq) for i, seq in enumerate(seqs_batch)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    token_repr = results["representations"][repr_layer].float().cpu()  # [B, L+2, D]
    out = []
    for i, seq in enumerate(seqs_batch):
        L = len(seq)
        emb = token_repr[i, 1:L + 1]
        if emb.shape[0] != L:
            raise RuntimeError(f"Embedding length {emb.shape[0]} != seq length {L}")
        out.append(emb)
    return out


def safe_extract(seqs_batch, model, alphabet, repr_layer, device, logger):
    while True:
        try:
            return extract_batch(seqs_batch, model, alphabet, repr_layer, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            if len(seqs_batch) == 1:
                raise
            half = max(1, len(seqs_batch) // 2)
            logger.warning(f"OOM: halving batch {len(seqs_batch)} → {half}")
            return (safe_extract(seqs_batch[:half], model, alphabet, repr_layer, device, logger) +
                    safe_extract(seqs_batch[half:], model, alphabet, repr_layer, device, logger))


def extract_wt_embeddings(wt_seqs: dict, model, alphabet, repr_layer: int,
                          cache_dir: Path, device, overwrite: bool, logger):
    wt_token_embs = {}
    for gfp_type, seq in wt_seqs.items():
        cache_file = cache_dir / f"wt_{gfp_type}.pt"
        if cache_file.exists() and not overwrite:
            d = torch.load(cache_file, map_location="cpu")
            wt_token_embs[gfp_type] = d["token_embeddings"]
            logger.info(f"Cached WT: {gfp_type}")
            continue
        embs = extract_batch([seq], model, alphabet, repr_layer, device)
        token_emb = embs[0]
        torch.save({"sequence": seq, "token_embeddings": token_emb,
                    "mean_embedding": token_emb.mean(0)}, cache_file)
        wt_token_embs[gfp_type] = token_emb
        logger.info(f"Extracted WT: {gfp_type}  shape={token_emb.shape}")
    return wt_token_embs


def extract_mutant_embeddings_bulk(df: pd.DataFrame, wt_token_embs: dict,
                                    model, alphabet, repr_layer: int,
                                    cache_dir: Path, batch_size: int,
                                    device, overwrite: bool, logger):
    """Extract embeddings per GFP type and save as bulk .npz files."""
    for gfp_type in df["GFP_type"].unique():
        out_file = cache_dir / f"mutants_{gfp_type}.npz"
        if out_file.exists() and not overwrite:
            logger.info(f"Cached bulk embeddings: {gfp_type}")
            continue

        gdf = df[df["GFP_type"] == gfp_type].reset_index(drop=True)
        N = len(gdf)
        logger.info(f"Extracting {N} samples for {gfp_type}...")

        wt_emb = wt_token_embs.get(gfp_type)
        wt_mean = wt_emb.float().mean(0).numpy() if wt_emb is not None else None

        sample_ids = []
        mean_embs = []
        mutsite_delta_means = []
        mutsite_mutant_means = []

        i = 0
        while i < N:
            batch = gdf.iloc[i:i + batch_size]
            seqs = batch["mutant_sequence"].tolist()

            try:
                embs = safe_extract(seqs, model, alphabet, repr_layer, device, logger)
            except Exception as e:
                logger.warning(f"Batch {i} failed for {gfp_type}: {e}")
                for _, row in batch.iterrows():
                    D = 1280
                    sample_ids.append(str(row["sample_id"]))
                    mean_embs.append(np.zeros(D, dtype=np.float16))
                    mutsite_delta_means.append(np.zeros(D, dtype=np.float16))
                    mutsite_mutant_means.append(np.zeros(D, dtype=np.float16))
                i += batch_size
                continue

            for j, (_, row) in enumerate(batch.iterrows()):
                emb = embs[j]  # [L, D]
                mean_emb = emb.mean(0).numpy().astype(np.float16)

                try:
                    mut_positions = json.loads(row["mut_positions"])
                except Exception:
                    mut_positions = []

                if mut_positions and wt_emb is not None:
                    pos_0 = [p for p in mut_positions
                             if 0 <= p < emb.shape[0] and p < wt_emb.shape[0]]
                    if pos_0:
                        idx = torch.tensor(pos_0, dtype=torch.long)
                        ms_mut = emb[idx].float().mean(0).numpy().astype(np.float16)
                        ms_delta = (emb[idx].float() - wt_emb[idx].float()).mean(0).numpy().astype(np.float16)
                    else:
                        D = emb.shape[1]
                        ms_mut = np.zeros(D, dtype=np.float16)
                        ms_delta = np.zeros(D, dtype=np.float16)
                else:
                    D = emb.shape[1]
                    ms_mut = np.zeros(D, dtype=np.float16)
                    ms_delta = np.zeros(D, dtype=np.float16)

                sample_ids.append(str(row["sample_id"]))
                mean_embs.append(mean_emb)
                mutsite_delta_means.append(ms_delta)
                mutsite_mutant_means.append(ms_mut)

            i += batch_size
            if i % 10000 < batch_size:
                logger.info(f"  {gfp_type}: {i}/{N}")

        # Save bulk
        np.savez_compressed(
            out_file,
            sample_ids=np.array(sample_ids),
            mean_embeddings=np.array(mean_embs),
            mutsite_delta_means=np.array(mutsite_delta_means),
            mutsite_mutant_means=np.array(mutsite_mutant_means),
        )
        logger.info(f"  Saved {N} embeddings → {out_file.name}  "
                    f"({np.array(mean_embs).shape})")


def main():
    args = parse_args_config("Extract ESM2 embeddings")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("embeddings_esm2", log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg["environment"]["require_cuda"]:
        logger.error("CUDA not available.")
        sys.exit(1)
    logger.info(f"Device: {device}  GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")
    logger.info(f"GPU free: {torch.cuda.mem_get_info()[0]//1024**2}MB" if device.type=="cuda" else "")

    cache_dir = get_output_dir(cfg, "outputs", "embeddings", "esm2")
    repr_layer = cfg["esm2"]["repr_layer"]
    batch_size = cfg["esm2"]["batch_size"]

    processed_csv = Path(cfg["_project_dir"]) / "outputs" / "processed" / "gfp_processed.csv"
    if not processed_csv.exists():
        logger.error("gfp_processed.csv not found. Run src.mutation first.")
        sys.exit(1)

    df = pd.read_csv(processed_csv)
    logger.info(f"Loaded {len(df)} samples")

    wt_seqs = {}
    for _, row in df.iterrows():
        if row["GFP_type"] not in wt_seqs:
            wt_seqs[row["GFP_type"]] = row["wt_sequence"]

    with timer(logger, "ESM2 loading"):
        model, alphabet = load_esm2(cfg["paths"]["esm2_weight"], device, logger)

    with timer(logger, "WT embedding extraction"):
        wt_token_embs = extract_wt_embeddings(
            wt_seqs, model, alphabet, repr_layer, cache_dir, device, args.overwrite, logger
        )

    # Also save WT mean embeddings for feature building
    wt_means = {gt: wt_token_embs[gt].float().mean(0).numpy()
                for gt in wt_token_embs}
    np.savez_compressed(cache_dir / "wt_means.npz",
                        **{gt: wt_means[gt].astype(np.float16) for gt in wt_means})

    with timer(logger, "Mutant embedding extraction"):
        extract_mutant_embeddings_bulk(
            df, wt_token_embs, model, alphabet, repr_layer,
            cache_dir, batch_size, device, args.overwrite, logger
        )

    logger.info("ESM2 embedding extraction complete.")


if __name__ == "__main__":
    main()
