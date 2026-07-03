"""Build feature matrix from ESM2 embeddings + physicochemical + single-prior features."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .utils import load_config, setup_logger, get_output_dir, timer, parse_args_config


# ── Physicochemical properties ─────────────────────────────────────────────────

PHYSCHEM = {
    #                  hydro  charge  polar  mw     vol  arom  pro   gly
    "A": dict(hydro=1.8,   charge=0,  polar=0, mw=89.1,  vol=88.6,  arom=0, pro=0, gly=0),
    "R": dict(hydro=-4.5,  charge=1,  polar=1, mw=174.2, vol=173.4, arom=0, pro=0, gly=0),
    "N": dict(hydro=-3.5,  charge=0,  polar=1, mw=132.1, vol=114.1, arom=0, pro=0, gly=0),
    "D": dict(hydro=-3.5,  charge=-1, polar=1, mw=133.1, vol=111.1, arom=0, pro=0, gly=0),
    "C": dict(hydro=2.5,   charge=0,  polar=0, mw=121.2, vol=108.5, arom=0, pro=0, gly=0),
    "E": dict(hydro=-3.5,  charge=-1, polar=1, mw=147.1, vol=138.4, arom=0, pro=0, gly=0),
    "Q": dict(hydro=-3.5,  charge=0,  polar=1, mw=146.2, vol=143.8, arom=0, pro=0, gly=0),
    "G": dict(hydro=-0.4,  charge=0,  polar=0, mw=75.0,  vol=60.1,  arom=0, pro=0, gly=1),
    "H": dict(hydro=-3.2,  charge=0.1,polar=1, mw=155.2, vol=153.2, arom=1, pro=0, gly=0),
    "I": dict(hydro=4.5,   charge=0,  polar=0, mw=131.2, vol=166.7, arom=0, pro=0, gly=0),
    "L": dict(hydro=3.8,   charge=0,  polar=0, mw=131.2, vol=166.7, arom=0, pro=0, gly=0),
    "K": dict(hydro=-3.9,  charge=1,  polar=1, mw=146.2, vol=168.6, arom=0, pro=0, gly=0),
    "M": dict(hydro=1.9,   charge=0,  polar=0, mw=149.2, vol=162.9, arom=0, pro=0, gly=0),
    "F": dict(hydro=2.8,   charge=0,  polar=0, mw=165.2, vol=189.9, arom=1, pro=0, gly=0),
    "P": dict(hydro=-1.6,  charge=0,  polar=0, mw=115.1, vol=112.7, arom=0, pro=1, gly=0),
    "S": dict(hydro=-0.8,  charge=0,  polar=1, mw=105.1, vol=89.0,  arom=0, pro=0, gly=0),
    "T": dict(hydro=-0.7,  charge=0,  polar=1, mw=119.1, vol=116.1, arom=0, pro=0, gly=0),
    "W": dict(hydro=-0.9,  charge=0,  polar=0, mw=204.2, vol=227.8, arom=1, pro=0, gly=0),
    "Y": dict(hydro=-1.3,  charge=0,  polar=1, mw=181.2, vol=193.6, arom=1, pro=0, gly=0),
    "V": dict(hydro=4.2,   charge=0,  polar=0, mw=117.1, vol=140.0, arom=0, pro=0, gly=0),
    "*": dict(hydro=0.0,   charge=0,  polar=0, mw=0.0,   vol=0.0,   arom=0, pro=0, gly=0),
}


def physchem_delta(mutations_json_str: str) -> np.ndarray:
    """Compute physicochemical delta features from mutation list."""
    try:
        muts = json.loads(mutations_json_str)
    except Exception:
        muts = []

    n_mut = len(muts)
    sum_dh = sum_dc = sum_dm = 0.0
    max_abs_dh = 0.0
    n_to_pro = n_from_pro = n_to_gly = n_from_gly = n_arom_change = 0

    for m in muts:
        wt_aa = m.get("wt", "A")
        mut_aa = m.get("mut", "A")
        wp = PHYSCHEM.get(wt_aa, PHYSCHEM["A"])
        mp = PHYSCHEM.get(mut_aa, PHYSCHEM["A"])

        dh = mp["hydro"] - wp["hydro"]
        dc = mp["charge"] - wp["charge"]
        dm = mp["mw"] - wp["mw"]

        sum_dh += dh
        sum_dc += dc
        sum_dm += dm
        max_abs_dh = max(max_abs_dh, abs(dh))

        if mp["pro"] and not wp["pro"]:
            n_to_pro += 1
        if wp["pro"] and not mp["pro"]:
            n_from_pro += 1
        if mp["gly"] and not wp["gly"]:
            n_to_gly += 1
        if wp["gly"] and not mp["gly"]:
            n_from_gly += 1
        if mp["arom"] != wp["arom"]:
            n_arom_change += 1

    mean_dh = sum_dh / max(n_mut, 1)
    mean_dc = sum_dc / max(n_mut, 1)
    mean_dm = sum_dm / max(n_mut, 1)

    return np.array([
        n_mut, sum_dh, mean_dh, max_abs_dh,
        sum_dc, mean_dc,
        sum_dm, mean_dm,
        n_to_pro, n_from_pro,
        n_to_gly, n_from_gly,
        n_arom_change,
    ], dtype=np.float32)


PHYSCHEM_FEAT_NAMES = [
    "mutation_count", "sum_delta_hydrophobicity", "mean_delta_hydrophobicity", "max_abs_delta_hydrophobicity",
    "sum_delta_charge", "mean_delta_charge",
    "sum_delta_weight", "mean_delta_weight",
    "num_to_proline", "num_from_proline",
    "num_to_glycine", "num_from_glycine",
    "num_aromatic_change",
]


# ── Single-mutation prior ──────────────────────────────────────────────────────

def build_single_prior(df_train: pd.DataFrame) -> dict:
    """Build {gfp_type: {mutation_str: delta_log_brightness}} from single-mut train rows."""
    prior = {}
    singles = df_train[df_train["mutation_count"] == 1].copy()
    for _, row in singles.iterrows():
        gt = row["GFP_type"]
        if gt not in prior:
            prior[gt] = {}
        prior[gt][row["aaMutations"]] = row["delta_log_brightness"]
    return prior


def single_prior_features(mutations_json_str: str, gfp_type: str, prior: dict) -> np.ndarray:
    try:
        muts = json.loads(mutations_json_str)
    except Exception:
        muts = []
    if not muts:
        return np.zeros(5, dtype=np.float32)

    gfp_prior = prior.get(gfp_type, {})
    effects = []
    for m in muts:
        key = f"{m['wt']}{m['pos']}{m['mut']}"
        if key in gfp_prior:
            effects.append(gfp_prior[key])

    ratio = len(effects) / max(len(muts), 1)
    if not effects:
        return np.array([0.0, 0.0, 0.0, 0.0, ratio], dtype=np.float32)

    arr = np.array(effects)
    return np.array([arr.sum(), arr.mean(), arr.max(), arr.min(), ratio], dtype=np.float32)


SINGLE_PRIOR_FEAT_NAMES = [
    "single_prior_sum", "single_prior_mean", "single_prior_max",
    "single_prior_min", "single_prior_available_ratio",
]


# ── ESM2 bulk data loader ─────────────────────────────────────────────────────

class ESM2BulkLoader:
    """Load all ESM2 embeddings from per-GFP-type .npz files into memory."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._emb_by_sid: dict = {}  # sample_id_str → (mean, delta, mutant)
        self._wt_means: dict = {}    # gfp_type → np.ndarray [D]
        self._loaded_types: set = set()

    def _load_type(self, gfp_type: str):
        if gfp_type in self._loaded_types:
            return
        npz_file = self.cache_dir / f"mutants_{gfp_type}.npz"
        if not npz_file.exists():
            self._loaded_types.add(gfp_type)
            return
        data = np.load(npz_file)
        ids = data["sample_ids"]
        means = data["mean_embeddings"]
        deltas = data["mutsite_delta_means"]
        mutants = data["mutsite_mutant_means"]
        for i, sid in enumerate(ids):
            self._emb_by_sid[str(sid)] = (means[i], deltas[i], mutants[i])
        self._loaded_types.add(gfp_type)

    def load_wt_means(self):
        wt_file = self.cache_dir / "wt_means.npz"
        if wt_file.exists():
            data = np.load(wt_file)
            for key in data.files:
                self._wt_means[key] = data[key].astype(np.float32)
        else:
            # Fall back to individual .pt files
            for pt_file in self.cache_dir.glob("wt_*.pt"):
                import torch
                gfp_type = pt_file.stem[3:]  # remove "wt_"
                d = torch.load(pt_file, map_location="cpu")
                self._wt_means[gfp_type] = d["mean_embedding"].float().numpy()

    def get_features(self, sample_id: str, gfp_type: str) -> np.ndarray | None:
        self._load_type(gfp_type)
        entry = self._emb_by_sid.get(str(sample_id))
        if entry is None:
            return None
        mut_mean, ms_delta, ms_mutant = [x.astype(np.float32) for x in entry]
        wt_mean = self._wt_means.get(gfp_type, np.zeros_like(mut_mean))
        global_delta = mut_mean - wt_mean
        return np.concatenate([mut_mean, wt_mean, global_delta, ms_delta, ms_mutant])


def load_esm2_features(row, cache_dir: Path, wt_mean_cache: dict, bulk_loader=None):
    """Load ESM2 features for a sample."""
    if bulk_loader is not None:
        return bulk_loader.get_features(str(row["sample_id"]), row["GFP_type"])
    # Legacy: individual .pt files
    import torch
    sid = str(row["sample_id"])
    gfp_type = row["GFP_type"]
    cache_file = cache_dir / f"sample_{sid}.pt"
    if not cache_file.exists():
        return None
    data = torch.load(cache_file, map_location="cpu")
    mut_mean = data["mean_embedding"].float().numpy()
    mutsite_mutant = data["mutsite_mutant_mean"].float().numpy()
    mutsite_delta = data["mutsite_delta_mean"].float().numpy()
    if gfp_type not in wt_mean_cache:
        wt_cache_file = cache_dir / f"wt_{gfp_type}.pt"
        if wt_cache_file.exists():
            wd = torch.load(wt_cache_file, map_location="cpu")
            wt_mean_cache[gfp_type] = wd["mean_embedding"].float().numpy()
        else:
            wt_mean_cache[gfp_type] = np.zeros_like(mut_mean)
    wt_mean = wt_mean_cache[gfp_type]
    global_delta = mut_mean - wt_mean
    return np.concatenate([mut_mean, wt_mean, global_delta, mutsite_delta, mutsite_mutant])


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, cfg: dict, prior: dict, cache_dir: Path, logger):
    use_esm2 = cfg["features"]["use_esm2"]
    use_physchem = cfg["features"]["use_physchem"]
    use_prior = cfg["features"]["use_single_prior"]

    # Use bulk loader for efficiency
    bulk_loader = None
    if use_esm2:
        bulk_loader = ESM2BulkLoader(cache_dir)
        bulk_loader.load_wt_means()

    rows_feat = []
    valid_indices = []
    wt_mean_cache = {}
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows()):
        parts = []

        if use_esm2:
            esm2_feat = load_esm2_features(row, cache_dir, wt_mean_cache, bulk_loader)
            if esm2_feat is None:
                skipped += 1
                continue
            parts.append(esm2_feat)

        if use_physchem:
            parts.append(physchem_delta(row["mutations_json"]))

        if use_prior:
            parts.append(single_prior_features(row["mutations_json"], row["GFP_type"], prior))

        if parts:
            rows_feat.append(np.concatenate(parts))
            valid_indices.append(i)
        else:
            rows_feat.append(np.zeros(1, dtype=np.float32))
            valid_indices.append(i)

    if skipped:
        logger.warning(f"Skipped {skipped} samples (missing ESM2 cache)")

    return (np.array(rows_feat, dtype=np.float32) if rows_feat else np.zeros((0, 1), dtype=np.float32),
            valid_indices)


def main():
    args = parse_args_config("Build feature matrix")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("features", log_dir)
    feat_dir = get_output_dir(cfg, "outputs", "features")
    proc_dir = Path(cfg["_project_dir"]) / "outputs" / "processed"
    cache_dir = Path(cfg["_project_dir"]) / "outputs" / "embeddings" / "esm2"

    processed_csv = proc_dir / "gfp_processed.csv"
    if not processed_csv.exists():
        logger.error("gfp_processed.csv not found.")
        sys.exit(1)

    df = pd.read_csv(processed_csv)
    logger.info(f"Loaded {len(df)} samples")

    with timer(logger, "feature building"):
        prior = build_single_prior(df)
        X, valid_indices = build_features(df, cfg, prior, cache_dir, logger)

    if X.shape[0] == 0:
        logger.error("No features built. Check ESM2 embeddings.")
        sys.exit(1)

    logger.info(f"Feature matrix shape: {X.shape}")
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    np.savez_compressed(
        feat_dir / "features_all.npz",
        X=X.astype(np.float16),
    )

    meta_cols = ["sample_id", "GFP_type", "aaMutations", "Brightness", "log_brightness",
                 "delta_log_brightness", "mutation_count", "mut_positions", "mutant_sequence"]
    meta = df_valid[[c for c in meta_cols if c in df_valid.columns]].copy()
    meta.to_csv(feat_dir / "feature_metadata.csv", index=False)

    logger.info(f"Features saved → {feat_dir}/features_all.npz  ({X.shape})")
    logger.info(f"Metadata saved → {feat_dir}/feature_metadata.csv")


if __name__ == "__main__":
    main()
