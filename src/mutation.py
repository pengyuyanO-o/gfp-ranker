"""Parse WT sequences, apply mutations, validate, output processed dataset."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import load_config, setup_logger, get_output_dir, timer, parse_args_config


# ── WT sequence parsing ────────────────────────────────────────────────────────

def parse_wt_sequences(filepath: str, logger) -> dict[str, str]:
    """Parse FASTA or plain-text WT sequence file. Returns {name: seq}."""
    text = Path(filepath).read_text()
    lines = text.splitlines()

    # Show first 10 for debugging
    logger.info(f"First 10 lines of WT file:\n" + "\n".join(lines[:10]))

    wt_seqs = {}
    current_name = None
    current_seq = []
    aa_chars = set("ACDEFGHIKLMNPQRSTVWY")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if current_name and current_seq:
                seq = "".join(current_seq).upper()
                wt_seqs[current_name] = seq
                current_name = None
                current_seq = []
            continue
        if line.startswith(">"):
            if current_name and current_seq:
                wt_seqs[current_name] = "".join(current_seq).upper()
            current_name = line[1:].strip().split()[0]
            current_seq = []
        else:
            # might be sequence continuation
            clean = re.sub(r"[^A-Za-z]", "", line).upper()
            if current_name and all(c in aa_chars for c in clean):
                current_seq.append(clean)

    if current_name and current_seq:
        wt_seqs[current_name] = "".join(current_seq).upper()

    if not wt_seqs:
        logger.error("Could not parse any WT sequences! File content:")
        logger.error(text[:2000])
        sys.exit(1)

    for name, seq in wt_seqs.items():
        logger.info(f"  {name}: {len(seq)} aa  ({seq[:20]}...)")
    return wt_seqs


# ── Mutation parsing and application ──────────────────────────────────────────

def parse_single_mutation(mut_str: str):
    """Parse 'A109D' → (wt_aa='A', pos=109, mut_aa='D')."""
    m = re.fullmatch(r"([A-Z\*])(\d+)([A-Z\*])", mut_str.strip())
    if not m:
        raise ValueError(f"Cannot parse mutation: {mut_str!r}")
    return m.group(1), int(m.group(2)), m.group(3)


def parse_mutations(aa_muts_str: str):
    """Parse 'A109D:K157R' → list of (wt_aa, pos, mut_aa)."""
    if aa_muts_str == "WT":
        return []
    parts = aa_muts_str.split(":")
    return [parse_single_mutation(p) for p in parts if p]


def apply_mutations(wt_seq: str, mutations: list) -> str:
    seq = list(wt_seq)
    for wt_aa, pos, mut_aa in mutations:
        # Position numbering in the data is "mature protein" numbering:
        # Met is excluded so position 1 = index 1 (0-based) in the full seq.
        # Equivalently, treat pos as a 0-based index directly.
        idx = pos  # mature-protein numbering → 0-based index of full sequence
        seq[idx] = mut_aa
    return "".join(seq)


def validate_mutation(wt_seq: str, mutations: list) -> tuple[bool, list]:
    """Check WT residues match. Return (all_ok, list_of_mismatch_info)."""
    mismatches = []
    for wt_aa, pos, mut_aa in mutations:
        idx = pos  # mature-protein numbering (0-based in full seq)
        if idx < 0 or idx >= len(wt_seq):
            mismatches.append({"pos": pos, "expected": wt_aa, "found": "OOB", "seq_len": len(wt_seq)})
            continue
        actual = wt_seq[idx]
        if actual != wt_aa:
            # Try immediate neighbors
            off = None
            if 0 < idx < len(wt_seq) - 1:
                if wt_seq[idx - 1] == wt_aa:
                    off = pos - 1
                elif wt_seq[idx + 1] == wt_aa:
                    off = pos + 1
            mismatches.append({"pos": pos, "expected": wt_aa, "found": actual, "off_by_one_match": off})
    return len(mismatches) == 0, mismatches


# ── Main processing ───────────────────────────────────────────────────────────

def build_processed_data(df: pd.DataFrame, wt_sequences: dict, cfg: dict, logger) -> pd.DataFrame:
    eps = cfg["data"]["eps"]
    max_mismatch_ratio = cfg["data"]["max_allowed_mismatch_ratio"]

    records = []
    mismatch_records = []

    for idx, row in df.iterrows():
        gfp_type = row["GFP type"]
        aa_muts = row["aaMutations"]
        brightness = row["Brightness_mean"]
        brightness_std = row.get("Brightness_std", 0.0)
        replicate_count = row.get("replicate_count", 1)

        if gfp_type not in wt_sequences:
            logger.warning(f"GFP type {gfp_type!r} not in WT sequences, skipping sample {idx}")
            continue

        wt_seq = wt_sequences[gfp_type]
        is_wt = aa_muts == "WT"

        try:
            mutations = parse_mutations(aa_muts)
        except ValueError as e:
            logger.warning(f"Sample {idx} parse error: {e}")
            continue

        valid = True
        mut_positions = []
        mutations_json = "[]"

        if not is_wt:
            ok, mismatches = validate_mutation(wt_seq, mutations)
            if not ok:
                for mm in mismatches:
                    mm["sample_id"] = idx
                    mm["GFP_type"] = gfp_type
                    mm["aaMutations"] = aa_muts
                    mismatch_records.append(mm)
                valid = False
            else:
                mut_positions = [p for _, p, _ in mutations]
                mutations_json = json.dumps([{"wt": w, "pos": p, "mut": m} for w, p, m in mutations])

        if valid:
            if is_wt:
                mutant_seq = wt_seq
                mut_positions = []
            else:
                mutant_seq = apply_mutations(wt_seq, mutations)

            records.append({
                "sample_id": idx,
                "GFP_type": gfp_type,
                "aaMutations": aa_muts,
                "mutation_count": len(mutations),
                "mut_positions": json.dumps(mut_positions),
                "mutations_json": mutations_json,
                "wt_sequence": wt_seq,
                "mutant_sequence": mutant_seq,
                "Brightness": brightness,
                "Brightness_std": brightness_std,
                "replicate_count": replicate_count,
                "is_WT": is_wt,
                "valid_mutation": True,
            })

    total = len(df)
    n_mismatch = len(set(r["sample_id"] for r in mismatch_records))
    mismatch_ratio = n_mismatch / max(total, 1)
    logger.info(f"Total={total}, valid={len(records)}, mismatch={n_mismatch} ({mismatch_ratio:.1%})")

    if mismatch_ratio > max_mismatch_ratio:
        logger.error(
            f"Mismatch ratio {mismatch_ratio:.1%} > {max_mismatch_ratio:.1%}. "
            "Large amounts of mutations don't match WT sequences. "
            "Check position indexing (1-based?), GFP type mapping, or WT file."
        )
        sys.exit(1)

    out_df = pd.DataFrame(records)

    # Compute log_brightness and delta
    out_df["log_brightness"] = np.log(out_df["Brightness"] + cfg["data"]["eps"])

    # WT baseline per GFP type
    wt_log_bright = {}
    for gfp_type in out_df["GFP_type"].unique():
        wt_rows = out_df[(out_df["GFP_type"] == gfp_type) & out_df["is_WT"]]
        if len(wt_rows) > 0:
            wt_log_bright[gfp_type] = wt_rows["log_brightness"].iloc[0]
        else:
            median_lb = out_df[out_df["GFP_type"] == gfp_type]["log_brightness"].median()
            wt_log_bright[gfp_type] = median_lb
            logger.warning(f"No WT row for {gfp_type}, using median log_brightness={median_lb:.4f}")

    out_df["delta_log_brightness"] = out_df.apply(
        lambda r: r["log_brightness"] - wt_log_bright[r["GFP_type"]], axis=1
    )

    return out_df, pd.DataFrame(mismatch_records) if mismatch_records else pd.DataFrame()


def main():
    args = parse_args_config("Parse mutations and build processed dataset")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("mutation", log_dir)
    out_dir = get_output_dir(cfg, "outputs", "processed")

    with timer(logger, "mutation processing"):
        wt_seqs = parse_wt_sequences(cfg["paths"]["wt_seq_file"], logger)

        raw_dedup = out_dir / "gfp_raw_dedup.csv"
        if not raw_dedup.exists():
            logger.error("gfp_raw_dedup.csv not found. Run src.data first.")
            sys.exit(1)
        df_raw = pd.read_csv(raw_dedup)

        processed_df, mismatch_df = build_processed_data(df_raw, wt_seqs, cfg, logger)

        processed_df.to_csv(out_dir / "gfp_processed.csv", index=False)
        logger.info(f"Saved {len(processed_df)} valid samples → gfp_processed.csv")

        if len(mismatch_df) > 0:
            mismatch_df.to_csv(out_dir / "mutation_mismatch_report.csv", index=False)
            logger.info(f"Mismatch report: {len(mismatch_df)} rows → mutation_mismatch_report.csv")


if __name__ == "__main__":
    main()
