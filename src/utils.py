import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve project_dir: "." means the directory containing the config file
    raw_project_dir = cfg["paths"].get("project_dir", ".")
    if raw_project_dir == ".":
        project_dir = config_path.parent.parent  # configs/ is one level below root
    else:
        project_dir = Path(raw_project_dir)
        if not project_dir.is_absolute():
            project_dir = config_path.parent.parent / project_dir
    cfg["_project_dir"] = str(project_dir.resolve())

    # Resolve data paths relative to project_dir if not absolute
    for key in ("data_xlsx", "wt_seq_file", "esm2_weight", "prott5_weight"):
        val = cfg["paths"].get(key, "")
        if val and not Path(val).is_absolute():
            cfg["paths"][key] = str((project_dir / val).resolve())

    return cfg


def get_output_dir(cfg: dict, *subpath) -> Path:
    base = Path(cfg["_project_dir"])
    p = base.joinpath(*subpath)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(name: str, log_dir: str | Path, level=logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@contextmanager
def timer(logger, msg: str):
    logger.info(f"[START] {msg}")
    t0 = time.time()
    yield
    logger.info(f"[END]   {msg}  ({time.time()-t0:.1f}s)")


def parse_args_config(description=""):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()
