"""Audit the provenance and consistency of the synthetic labelled benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

from evaluation.generate_dataset import build_rows


ROOT = Path(__file__).resolve().parents[1]


def audit_dataset(dataset: Path = ROOT / "evaluation/math_benchmark_1000.csv", *, root: Path = ROOT) -> dict:
    with dataset.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000:
        raise ValueError("The benchmark must contain 1000 cases")
    ids = [row["question_id"] for row in rows]
    stems = ["".join(unicodedata.normalize("NFKC", row["problem_text"]).split()) for row in rows]
    if len(set(ids)) != 1000 or len(set(stems)) != 1000 or not all(stems):
        raise ValueError("Duplicate case IDs or normalized problem stems")
    expected = {row["question_id"]: row for row in build_rows()}
    for row in rows:
        if row != expected.get(row["question_id"]):
            raise ValueError(f"Labels or fields differ from the versioned generator: {row['question_id']}")
    sources = []
    for name in sorted({row["relevant_source"] for row in rows}):
        source = (root / name).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise ValueError(f"Missing or external knowledge source: {name}")
        sources.append({"path": name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    return {"schema_version": 1, "rows": len(rows), "unique_normalized_stems": len(set(stems)),
            "annotation_method": "deterministic_programmatic",
            "human_reviewed_cases": 0,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "generator_sha256": hashlib.sha256((ROOT / "evaluation/generate_dataset.py").read_bytes()).hexdigest(),
            "case_types": dict(sorted(Counter(row["case_type"] for row in rows).items())),
            "knowledge_points": dict(sorted(Counter(row["knowledge_point"] for row in rows).items())),
            "source_files": sources,
            "checks": {"generator_consistency": True, "normalized_deduplication": True,
                       "knowledge_sources_exist": True},
            "limitations": ["Programmatic labels are not human annotations.",
                            "Source files support knowledge points; they are not the origins of mined competition questions.",
                            "Generator consistency does not prove external answer accuracy or student adoption."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/math_benchmark_1000.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_dataset(args.dataset)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
