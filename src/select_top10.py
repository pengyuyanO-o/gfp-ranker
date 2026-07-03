"""Generate and score candidate GFP mutants, output top-10 high-brightness candidates."""
import gc
import itertools
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import (build_single_prior, physchem_delta, single_prior_features,
                       load_esm2_features, PHYSCHEM)
from .models import MLPRanker
from .utils import load_config, setup_logger, get_output_dir, timer, parse_args_config


# ── Candidate generation ───────────────────────────────────────────────────────

def generate_candidates(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    """Generate novel combined mutants via random sampling (not exhaustive enumeration)."""
    top_q = cfg["top10"]["beneficial_single_top_quantile"]
    min_order = cfg["top10"]["min_combination_order"]
    max_order = cfg["top10"]["max_combination_order"]
    max_cands = cfg["top10"]["max_candidates_per_gfp_type"]

    existing = set(zip(df["GFP_type"], df["aaMutations"]))

    candidates = []
    rng = random.Random(42)

    for gfp_type in df["GFP_type"].unique():
        gdf = df[df["GFP_type"] == gfp_type]
        wt_seq = gdf.iloc[0]["wt_sequence"]
        singles = gdf[gdf["mutation_count"] == 1].copy()
        if len(singles) == 0:
            logger.warning(f"No single mutations for {gfp_type}")
            continue

        threshold = singles["delta_log_brightness"].quantile(1 - top_q)
        beneficial = singles[singles["delta_log_brightness"] >= threshold]
        ben_muts = beneficial["aaMutations"].tolist()
        logger.info(f"  {gfp_type}: {len(beneficial)} beneficial single mutations for combination")

        if len(ben_muts) < min_order:
            logger.warning(f"  {gfp_type}: not enough beneficial mutations for combinations")
            continue

        # Pre-parse all beneficial mutations
        parsed_muts = {}
        for mut_str in ben_muts:
            m = re.fullmatch(r"([A-Z\*])(\d+)([A-Z\*])", mut_str)
            if m:
                parsed_muts[mut_str] = (m.group(1), int(m.group(2)), m.group(3))

        valid_muts = [m for m in ben_muts if m in parsed_muts]
        # Group by position to ensure no overlap
        pos_to_mut = {}
        for mut_str in valid_muts:
            pos = parsed_muts[mut_str][1]
            if pos not in pos_to_mut:
                pos_to_mut[pos] = []
            pos_to_mut[pos].append(mut_str)

        unique_positions = list(pos_to_mut.keys())
        count = 0
        max_attempts = max_cands * 20  # avoid infinite loop

        attempts = 0
        while count < max_cands and attempts < max_attempts:
            attempts += 1
            order = rng.randint(min_order, min(max_order, len(unique_positions)))
            chosen_positions = rng.sample(unique_positions, order)
            # Pick one mutation per position
            combo = [rng.choice(pos_to_mut[p]) for p in chosen_positions]
            combo.sort()  # canonical ordering

            muts_parsed = [parsed_muts[m] for m in combo]
            aa_muts = ":".join(combo)

            if (gfp_type, aa_muts) in existing:
                continue

            # Validate against WT (using mature-protein 0-based indexing)
            seq = list(wt_seq)
            valid = True
            for wt_aa, pos, mut_aa in muts_parsed:
                idx = pos  # mature-protein 0-based index
                if idx < 0 or idx >= len(seq) or seq[idx] != wt_aa:
                    valid = False
                    break
                seq[idx] = mut_aa
            if not valid:
                continue

            mutant_seq = "".join(seq)
            existing.add((gfp_type, aa_muts))  # prevent duplicates within generation
            candidates.append({
                "GFP_type": gfp_type,
                "aaMutations": aa_muts,
                "mutation_count": order,
                "mut_positions": json.dumps([p for _, p, _ in muts_parsed]),
                "mutations_json": json.dumps([{"wt": w, "pos": p, "mut": m} for w, p, m in muts_parsed]),
                "wt_sequence": wt_seq,
                "mutant_sequence": mutant_seq,
                "is_WT": False,
                "valid_mutation": True,
                "is_candidate": True,
            })
            count += 1

        logger.info(f"  {gfp_type}: generated {count} candidates ({attempts} attempts)")

    return pd.DataFrame(candidates) if candidates else pd.DataFrame()


# ── ESM2 embedding for candidates ─────────────────────────────────────────────

def extract_candidate_embeddings(cands_df: pd.DataFrame, cfg: dict, logger):
    """Extract ESM2 embeddings for candidate sequences."""
    if len(cands_df) == 0:
        return

    try:
        import esm as esm_lib
    except ImportError:
        logger.warning("fair-esm not available, cannot extract candidate embeddings")
        return

    cache_dir = Path(cfg["_project_dir"]) / "outputs" / "embeddings" / "esm2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Extracting ESM2 embeddings for {len(cands_df)} candidates on {device}")

    try:
        import warnings
        weight_path = cfg["paths"]["esm2_weight"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_data = torch.load(weight_path, map_location="cpu")
            model_name = Path(weight_path).stem
            model, alphabet = esm_lib.pretrained.load_model_and_alphabet_core(
                model_name, model_data, None
            )
        model = model.eval().half().to(device)
    except Exception as e:
        logger.warning(f"ESM2 load failed for candidate scoring: {e}")
        return

    # Load WT token embeddings
    wt_token_embs = {}
    for gfp_type in cands_df["GFP_type"].unique():
        wt_file = cache_dir / f"wt_{gfp_type}.pt"
        if wt_file.exists():
            d = torch.load(wt_file, map_location="cpu")
            wt_token_embs[gfp_type] = d["token_embeddings"].float()

    batch_size = cfg["esm2"]["batch_size"]
    repr_layer = cfg["esm2"]["repr_layer"]

    from .embeddings_esm2 import safe_extract

    rows = list(cands_df.iterrows())
    total = len(rows)
    for start in range(0, total, batch_size):
        batch_rows = rows[start:start + batch_size]
        seqs = [r["mutant_sequence"] for _, r in batch_rows]

        try:
            embs = safe_extract(seqs, model, alphabet, repr_layer, device, logger)
        except Exception as e:
            logger.warning(f"Batch extraction failed: {e}")
            for _, row in batch_rows:
                sid = f"cand_{row['GFP_type']}_{row['aaMutations'][:30]}"
                cache_dir.joinpath(f"sample_{sid}.pt")  # skip
            continue

        for (_, row), emb in zip(batch_rows, embs):
            sid = f"cand_{row['GFP_type']}_{row['aaMutations']}"
            mean_emb = emb.mean(0)
            mut_positions = json.loads(row["mut_positions"])
            gfp_type = row["GFP_type"]

            if mut_positions and gfp_type in wt_token_embs:
                wt_emb = wt_token_embs[gfp_type]
                pos_0 = [p for p in mut_positions if 0 <= p < emb.shape[0] and p < wt_emb.shape[0]]
                if pos_0:
                    mutsite_mutant_mean = emb[pos_0].mean(0)
                    mutsite_delta_mean = (emb[pos_0] - wt_emb[pos_0]).mean(0)
                else:
                    mutsite_mutant_mean = torch.zeros_like(mean_emb)
                    mutsite_delta_mean = torch.zeros_like(mean_emb)
            else:
                mutsite_mutant_mean = torch.zeros_like(mean_emb)
                mutsite_delta_mean = torch.zeros_like(mean_emb)

            # Hash the sid to avoid too-long filenames
            import hashlib
            sid_hash = hashlib.md5(sid.encode()).hexdigest()[:16]
            torch.save({
                "sequence": row["mutant_sequence"],
                "mean_embedding": mean_emb.cpu(),
                "mutsite_mutant_mean": mutsite_mutant_mean.cpu(),
                "mutsite_delta_mean": mutsite_delta_mean.cpu(),
            }, cache_dir / f"sample_cand_{sid_hash}.pt")
            cands_df.loc[cands_df["aaMutations"] == row["aaMutations"], "_cache_hash"] = sid_hash

        if (start + batch_size) % 1000 < batch_size:
            logger.info(f"  Extracted {min(start+batch_size, total)}/{total} candidate embeddings")

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_candidates_with_ensemble(cands_df: pd.DataFrame, df_train: pd.DataFrame,
                                    cfg: dict, logger) -> pd.DataFrame:
    """Score candidates using ensemble of trained MLP models."""
    ckpt_dir = Path(cfg["_project_dir"]) / "outputs" / "checkpoints"
    cache_dir = Path(cfg["_project_dir"]) / "outputs" / "embeddings" / "esm2"
    feat_dir = Path(cfg["_project_dir"]) / "outputs" / "features"

    ckpts = list(ckpt_dir.glob("mlp_ranker_random_seed*.pt"))
    if not ckpts:
        logger.warning("No MLP checkpoints found for ensemble scoring")
        return cands_df

    # Load feature shape from training data
    train_feat = np.load(feat_dir / "features_all.npz")
    input_dim = train_feat["X"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build prior from training data
    prior = build_single_prior(df_train)

    # Build candidate feature rows using cached embeddings
    # We need to map each candidate to its cache file
    wt_mean_cache = {}
    all_preds = []

    for ckpt_path in ckpts:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model = MLPRanker(input_dim).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
        except Exception as e:
            logger.warning(f"Cannot load {ckpt_path}: {e}")
            continue

        preds_this = []
        for _, row in cands_df.iterrows():
            cache_hash = row.get("_cache_hash", None)
            if cache_hash is None:
                # Try to find by sequence hash
                import hashlib
                sid = f"cand_{row['GFP_type']}_{row['aaMutations']}"
                cache_hash = hashlib.md5(sid.encode()).hexdigest()[:16]

            cache_file = cache_dir / f"sample_cand_{cache_hash}.pt"
            if not cache_file.exists():
                preds_this.append(np.nan)
                continue

            data = torch.load(cache_file, map_location="cpu")
            mut_mean = data["mean_embedding"].float().numpy()
            mutsite_mutant = data["mutsite_mutant_mean"].float().numpy()
            mutsite_delta = data["mutsite_delta_mean"].float().numpy()

            gfp_type = row["GFP_type"]
            if gfp_type not in wt_mean_cache:
                wt_file = cache_dir / f"wt_{gfp_type}.pt"
                if wt_file.exists():
                    wd = torch.load(wt_file, map_location="cpu")
                    wt_mean_cache[gfp_type] = wd["mean_embedding"].float().numpy()
                else:
                    wt_mean_cache[gfp_type] = np.zeros_like(mut_mean)

            wt_mean = wt_mean_cache[gfp_type]
            global_delta = mut_mean - wt_mean
            esm2_feat = np.concatenate([mut_mean, wt_mean, global_delta, mutsite_delta, mutsite_mutant])
            phys_feat = physchem_delta(row["mutations_json"])
            prior_feat = single_prior_features(row["mutations_json"], gfp_type, prior)
            feat_vec = np.concatenate([esm2_feat, phys_feat, prior_feat]).astype(np.float32)

            # Pad/trim to input_dim
            if len(feat_vec) < input_dim:
                feat_vec = np.concatenate([feat_vec, np.zeros(input_dim - len(feat_vec))])
            elif len(feat_vec) > input_dim:
                feat_vec = feat_vec[:input_dim]

            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                x = torch.tensor(feat_vec[None], dtype=torch.float32).to(device)
                p = model(x).item()
            preds_this.append(p)

        all_preds.append(preds_this)

    if not all_preds:
        return cands_df

    preds_array = np.array(all_preds)  # [n_models, n_cands]
    cands_df = cands_df.copy()
    cands_df["pred_delta_log_brightness_mean"] = np.nanmean(preds_array, axis=0)
    cands_df["pred_delta_log_brightness_std"] = np.nanstd(preds_array, axis=0)

    beta = cfg["top10"]["uncertainty_beta"]
    cands_df["final_score"] = cands_df["pred_delta_log_brightness_mean"] - beta * cands_df["pred_delta_log_brightness_std"]

    # Get WT log_brightness for each GFP type
    wt_lb = {}
    for gfp_type in df_train["GFP_type"].unique():
        wt_rows = df_train[(df_train["GFP_type"] == gfp_type) & df_train["is_WT"]]
        if len(wt_rows):
            wt_lb[gfp_type] = wt_rows["log_brightness"].iloc[0]

    cands_df["pred_log_brightness"] = cands_df.apply(
        lambda r: r["pred_delta_log_brightness_mean"] + wt_lb.get(r["GFP_type"], 0.0), axis=1
    )
    return cands_df


# ── Stability proxy + diversity selection ─────────────────────────────────────

def stability_risk(row) -> str:
    risks = []
    try:
        muts = json.loads(row["mutations_json"])
    except Exception:
        muts = []

    n_to_pro = sum(1 for m in muts if m.get("mut") == "P" and m.get("wt") != "P")
    n_charge_flip = sum(1 for m in muts
                        if PHYSCHEM.get(m.get("wt", "A"), {}).get("charge", 0) *
                           PHYSCHEM.get(m.get("mut", "A"), {}).get("charge", 0) < 0)

    if row.get("mutation_count", 0) > 5:
        risks.append("high_mutation_count")
    if n_to_pro >= 2:
        risks.append("multi_proline")
    if n_charge_flip >= 2:
        risks.append("charge_reversal")
    if row.get("pred_delta_log_brightness_std", 0) > 0.5:
        risks.append("high_uncertainty")

    return ";".join(risks) if risks else "low"


def select_diverse_top10(cands_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Select top-k with diversity constraint."""
    if len(cands_df) == 0:
        return cands_df

    cands_df = cands_df.dropna(subset=["final_score"]).copy()
    cands_df = cands_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    selected = []
    used_position_sets = []

    for _, row in cands_df.iterrows():
        if len(selected) >= k:
            break
        try:
            pos_set = frozenset(json.loads(row["mut_positions"]))
        except Exception:
            pos_set = frozenset()

        # Diversity: skip if position set is identical to an already selected candidate
        too_similar = any(pos_set == used for used in used_position_sets)
        if too_similar and len(selected) < k - 1:
            continue

        selected.append(row)
        used_position_sets.append(pos_set)

    # Fill up if needed
    if len(selected) < k:
        already = {r.name for r in selected}
        for _, row in cands_df.iterrows():
            if len(selected) >= k:
                break
            if row.name not in already:
                selected.append(row)

    out = pd.DataFrame(selected).head(k).copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def main():
    args = parse_args_config("Select top-10 high-brightness candidates")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("select_top10", log_dir)
    top10_dir = get_output_dir(cfg, "outputs", "top10")
    proc_dir = Path(cfg["_project_dir"]) / "outputs" / "processed"

    df = pd.read_csv(proc_dir / "gfp_processed.csv")
    logger.info(f"Loaded {len(df)} processed samples")

    # Check for user-provided candidate CSV
    candidate_csv = getattr(args, "candidate_csv", None)
    if candidate_csv and Path(candidate_csv).exists():
        logger.info(f"Loading candidates from {candidate_csv}")
        cands_df = pd.read_csv(candidate_csv)
    elif cfg["top10"]["generate_candidates_if_missing"]:
        with timer(logger, "candidate generation"):
            cands_df = generate_candidates(df, cfg, logger)
        logger.info(f"Generated {len(cands_df)} candidates")
        if len(cands_df) == 0:
            logger.warning("No candidates generated.")
            # Fallback: use top-k from known data
            logger.info("Using top samples from known data as fallback.")
            cands_df = df[~df["is_WT"]].nlargest(200, "delta_log_brightness").copy()
            cands_df["is_candidate"] = False
    else:
        logger.error("No candidate CSV and generate_candidates_if_missing=false.")
        sys.exit(1)

    if len(cands_df) > 0 and cands_df.get("is_candidate", pd.Series([False])).any():
        with timer(logger, "candidate embedding extraction"):
            extract_candidate_embeddings(cands_df, cfg, logger)

    with timer(logger, "ensemble scoring"):
        cands_df = score_candidates_with_ensemble(cands_df, df, cfg, logger)

    if "final_score" not in cands_df.columns:
        logger.warning("No ensemble scores; falling back to delta_log_brightness")
        if "delta_log_brightness" in cands_df.columns:
            cands_df["pred_delta_log_brightness_mean"] = cands_df["delta_log_brightness"]
            cands_df["pred_delta_log_brightness_std"] = 0.0
            cands_df["final_score"] = cands_df["delta_log_brightness"]
            cands_df["pred_log_brightness"] = cands_df["log_brightness"]
        else:
            logger.error("Cannot score candidates.")
            sys.exit(1)

    # Single prior features for report
    prior = build_single_prior(df)
    cands_df["single_prior_sum"] = cands_df.apply(
        lambda r: single_prior_features(r.get("mutations_json", "[]"), r["GFP_type"], prior)[0], axis=1
    )
    cands_df["single_prior_available_ratio"] = cands_df.apply(
        lambda r: single_prior_features(r.get("mutations_json", "[]"), r["GFP_type"], prior)[4], axis=1
    )
    cands_df["stability_risk"] = cands_df.apply(stability_risk, axis=1)
    cands_df["notes"] = cands_df["stability_risk"].apply(
        lambda x: "caution: " + x if x != "low" else "stable proxy"
    )

    output_cols = [
        "rank", "GFP_type", "aaMutations", "mutant_sequence",
        "mutation_count", "pred_delta_log_brightness_mean", "pred_delta_log_brightness_std",
        "pred_log_brightness", "final_score", "single_prior_sum",
        "single_prior_available_ratio", "stability_risk", "notes",
    ]

    k = cfg["top10"]["final_topk"]

    # Save all scored candidates for analysis
    save_cols = [c for c in output_cols[1:] if c in cands_df.columns]  # skip "rank"
    cands_df[save_cols].to_csv(top10_dir / "all_candidates_scored.csv", index=False)
    logger.info(f"All scored candidates saved → {top10_dir}/all_candidates_scored.csv")

    # ── Per-GFP-type top-k ────────────────────────────────────────────────────
    with timer(logger, "per-GFP top-10 selection"):
        per_gfp_frames = []
        for gfp_type, grp in cands_df.groupby("GFP_type"):
            top_g = select_diverse_top10(grp.copy(), k=k)
            per_gfp_frames.append(top_g)

    per_gfp_all = pd.concat(per_gfp_frames, ignore_index=True)
    per_gfp_out = per_gfp_all[[c for c in output_cols if c in per_gfp_all.columns]]
    per_gfp_out.to_csv(top10_dir / "top10_per_gfp_type.csv", index=False)

    # ── Overall top-k across all GFP types ───────────────────────────────────
    with timer(logger, "overall top-10 selection"):
        top10 = select_diverse_top10(cands_df, k=k)

    top10_out = top10[[c for c in output_cols if c in top10.columns]]
    top10_out.to_csv(top10_dir / "top10_candidates.csv", index=False)

    # ── Logging ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("TOP 10 PER GFP TYPE:")
    logger.info(f"{'='*60}")
    for gfp_type, grp in per_gfp_out.groupby("GFP_type", sort=False):
        logger.info(f"\n  [{gfp_type}]")
        for _, row in grp.iterrows():
            logger.info(
                f"    #{int(row['rank'])} {row['aaMutations'][:40]:40s}  "
                f"score={row.get('final_score', 0):.4f}  "
                f"n_mut={int(row.get('mutation_count',0))}  risk={row.get('stability_risk','?')}"
            )

    logger.info(f"\n{'='*60}")
    logger.info("OVERALL TOP 10 (across all GFP types):")
    logger.info(f"{'='*60}")
    for _, row in top10_out.iterrows():
        logger.info(
            f"  #{int(row['rank'])} {row['GFP_type']:10s} {row['aaMutations'][:40]:40s}  "
            f"score={row.get('final_score', 0):.4f}  risk={row.get('stability_risk','?')}"
        )
    logger.info(f"\nSaved → {top10_dir}/top10_per_gfp_type.csv")
    logger.info(f"Saved → {top10_dir}/top10_candidates.csv")


if __name__ == "__main__":
    main()
