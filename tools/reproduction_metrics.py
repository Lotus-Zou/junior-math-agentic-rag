#!/usr/bin/env python3
"""Rerun five RAGAS cases without overwriting the user's latest summary."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.evaluation import load_ragas_dataset, run_e2e
from tools.reproduce_experiments import PREFIX, REPORTS, write


def main():
    report, metrics = run_e2e(load_ragas_dataset(), 0, 5, PREFIX+"-ragas", False)
    report.to_csv(REPORTS / f"{PREFIX}-ragas.csv",index=False,encoding="utf-8-sig")
    write("ragas", {"summary":metrics,"scope":"Five synthetic knowledge questions only; judge and generator share Qwen2.5-7B weights, with separate requests. Not full 1000-case RAGAS, not historical 0.50-to-0.78 reproduction."})


if __name__ == "__main__":
    main()
