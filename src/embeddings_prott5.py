"""Optional ProtT5 embeddings (ablation only). Skipped gracefully if unavailable."""
import sys
from pathlib import Path

from .utils import load_config, setup_logger, get_output_dir, parse_args_config


def main():
    args = parse_args_config("Extract ProtT5 embeddings (optional)")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("embeddings_prott5", log_dir)

    if not cfg["features"]["use_prott5"]:
        logger.info("use_prott5=false in config, skipping ProtT5 extraction.")
        return

    prott5_path = Path(cfg["paths"]["prott5_weight"])
    if not prott5_path.exists():
        logger.warning(f"ProtT5 weight not found at {prott5_path}. Skipping.")
        return

    try:
        from transformers import T5Tokenizer, T5EncoderModel
        import torch
    except ImportError as e:
        logger.warning(f"transformers not available ({e}). Skipping ProtT5.")
        return

    try:
        tokenizer_dir = prott5_path.parent
        tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_dir), do_lower_case=False)
        logger.info("ProtT5 tokenizer loaded.")
    except Exception as e:
        logger.warning(f"ProtT5 tokenizer load failed ({e}). Skipping.")
        return

    try:
        model = T5EncoderModel.from_pretrained(str(tokenizer_dir))
        model = model.eval()
        logger.info("ProtT5 model loaded (not extracting embeddings in this stub).")
    except Exception as e:
        logger.warning(f"ProtT5 model load failed ({e}). Skipping.")
        return

    logger.info("ProtT5 stub: full extraction not implemented. Set use_prott5=false to suppress this.")


if __name__ == "__main__":
    main()
