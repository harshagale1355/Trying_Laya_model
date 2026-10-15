#!/usr/bin/env python3
"""
Score one or more Laya checkpoints on the router eval sets.

Usage:
  python -m erp_assistant.eval_router --models convaiinnovations/laya checkpoints/laya-erp-router-head
  python -m erp_assistant.eval_router --models checkpoints/laya-erp-router --data data/router/handwritten_eval.jsonl --show-errors 30

Accepts rows in laya-train format ({state, expected}) or the short {text, label} format.
"""
import argparse
import json
import time
from collections import Counter, defaultdict

import laya

from .config import BASE_MODEL, ROUTER_DATA_DIR
from .router_question import LABELS, QUESTIONS, make_state


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "state" in r:
                rows.append({"text": r["state"]["text"], "label": r["expected"]["route"], "meta": r.get("meta", {})})
            else:
                rows.append({"text": r["text"], "label": r["label"], "meta": r.get("meta", {})})
    return rows


def route_of(result):
    answer = (result.get("answers") or result)["route"]
    return answer["choice"], answer.get("confidence")


def evaluate(agent, rows, batch_size):
    preds = []
    start = time.perf_counter()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        results = agent.predict_batch([make_state(r["text"]) for r in chunk], QUESTIONS)
        preds.extend(route_of(res) for res in results)
    ms = (time.perf_counter() - start) * 1000 / max(1, len(rows))
    return preds, ms


def report(name, rows, preds, ms, show_errors):
    correct = sum(p == r["label"] for (p, _), r in zip(preds, rows))
    print("\n== %s  (%d rows)" % (name, len(rows)))
    print("accuracy  %.1f%%  (%d/%d)   %.0f ms/query" % (100 * correct / len(rows), correct, len(rows), ms))

    conf = Counter((r["label"], p) for (p, _), r in zip(preds, rows))
    print("confusion (rows = gold, cols = predicted)")
    print("  %-12s" % "" + "".join("%13s" % l for l in LABELS))
    for g in LABELS:
        print("  %-12s" % g + "".join("%13d" % conf[(g, p)] for p in LABELS))

    for key in ("template", "concept"):
        groups = defaultdict(lambda: [0, 0])
        for (p, _), r in zip(preds, rows):
            if key in r["meta"]:
                groups[r["meta"][key]][0] += p == r["label"]
                groups[r["meta"][key]][1] += 1
        weak = sorted((c / n, k, n) for k, (c, n) in groups.items() if c < n)[:8]
        if weak:
            print("weakest %ss: " % key + ", ".join("%s %.0f%% (n=%d)" % (k, 100 * a, n) for a, k, n in weak))

    errors = [(c or 0, r, p) for (p, c), r in zip(preds, rows) if p != r["label"]]
    errors.sort(key=lambda e: -e[0])
    for c, r, p in errors[:show_errors]:
        print("  x %-11s -> %-11s conf=%.2f  %s" % (r["label"], p, c, r["text"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[BASE_MODEL])
    ap.add_argument("--data", nargs="+", default=[str(ROUTER_DATA_DIR / "test.jsonl"),
                                                str(ROUTER_DATA_DIR / "handwritten_eval.jsonl")])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--show-errors", type=int, default=10)
    a = ap.parse_args()

    datasets = {path: load_rows(path) for path in a.data}
    for model in a.models:
        print("\n#### model: %s" % model)
        agent = laya.load(model)
        for path, rows in datasets.items():
            preds, ms = evaluate(agent, rows, a.batch_size)
            report(path, rows, preds, ms, a.show_errors)


if __name__ == "__main__":
    main()
