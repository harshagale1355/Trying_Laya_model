#!/usr/bin/env python3
"""
Bundle what notebooks/train_laya_colab.ipynb needs into colab_upload.zip (repo root).
Rerun it whenever you regenerate data/router/ or edit router_question.py / eval_router.py,
or Colab will train on stale files.

Usage:  python -m erp_assistant.make_colab_zip
"""
import zipfile

from .config import ROOT, ROUTER_DATA_DIR

FILES = [ROOT / "erp_assistant" / name for name in ("__init__.py", "config.py", "router_question.py", "eval_router.py")]
FILES += [ROUTER_DATA_DIR / name for name in ("train.jsonl", "val.jsonl", "test.jsonl", "handwritten_eval.jsonl")]


def main():
    out = ROOT / "colab_upload.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in FILES:
            z.write(path, path.relative_to(ROOT).as_posix())
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB, {len(FILES)} files)")


if __name__ == "__main__":
    main()
