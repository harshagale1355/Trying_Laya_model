"""Repo paths shared by every module, plus a tiny .env loader (no python-dotenv needed)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ERP_DIR = ROOT / "data" / "erp"              # erp.db, schema.sql, docs/ (generate_erp_data.py)
ROUTER_DATA_DIR = ROOT / "data" / "router"   # Laya train/val/test + handwritten eval (build_dataset.py)
CHECKPOINTS_DIR = ROOT / "checkpoints"
BASE_MODEL = "convaiinnovations/laya"


def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines from .env into os.environ. Variables already set in the environment win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)
