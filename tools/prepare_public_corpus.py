#!/usr/bin/env python3
"""Audit public sources; build a deduplicated train-only TAL case corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import unicodedata
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TAL_REVISION = "8a66d6e14c65ba002beda39747e6b7c43bd2d472"
HF_SOURCES = ("nvidia/OpenMathReasoning", "nvidia/OpenMathInstruct-2",
              "AI-MO/NuminaMath-CoT", "Hothan/OlympiadBench", "XiaoMi/CMATH")
MAX_BYTES = 16 * 1024 * 1024


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def fetch(url: str, destination: Path, *, reserve_bytes: int, accept: str = "application/json") -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < reserve_bytes + MAX_BYTES:
        raise OSError("Insufficient disk reserve; no large dataset download was attempted")
    request = urllib.request.Request(url, headers={"User-Agent": "junior-math-source-audit/1.0", "Accept": accept})
    with urllib.request.urlopen(request, timeout=8) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("Resource exceeds the 16 MiB metadata/source budget")
    destination.write_bytes(body)
    return {"url": url, "path": display_path(destination), "bytes": len(body), "sha256": digest(body)}


def canonical_problem(row: dict) -> str:
    choices = sorted(str(item["content"])
                     for group in row["answer_option_list"] for item in group)
    text = str(row["problem"]) + "\n" + json.dumps(choices, ensure_ascii=False)
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def stem_digest(row: dict) -> str:
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(row["problem"])))
    return digest(text.encode())


def build_corpus(directory: Path, database: Path) -> dict:
    files = {split: directory / "ch_single_choice_constructed_5K" / f"ch_single_choice_{split}_{size}K.jsonl"
             for split, size in (("train", 3), ("test", 2))}
    counts, rows, hashes = {}, {}, {}
    for split, path in files.items():
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        counts[split] = dict(Counter(str(row["dataset_name"]).split("_math_")[0] for row in records))
        rows[split] = [row for row in records if str(row["dataset_name"]).startswith("mid_math_")]
        hashes[split] = digest(path.read_bytes())
    heldout = {digest(canonical_problem(row).encode()) for row in rows["test"]}
    heldout_stems = {stem_digest(row) for row in rows["test"]}
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS questions(
                problem_sha256 TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                question TEXT NOT NULL, options_json TEXT NOT NULL,
                reference_answer TEXT NOT NULL, analysis_json TEXT NOT NULL,
                revision TEXT NOT NULL, split TEXT NOT NULL CHECK(split='train'));
            CREATE TABLE IF NOT EXISTS heldout(problem_sha256 TEXT PRIMARY KEY);
            CREATE VIRTUAL TABLE IF NOT EXISTS question_search USING fts5(problem_sha256 UNINDEXED, terms);
        """)
        # Only this tool's named tables are rebuilt, not the app's vector store.
        connection.execute("DELETE FROM questions")
        connection.execute("DELETE FROM heldout")
        connection.execute("DELETE FROM question_search")
        connection.executemany("INSERT INTO heldout VALUES(?)", ((value,) for value in sorted(heldout)))
        leaked = duplicates = 0
        for row in rows["train"]:
            key = digest(canonical_problem(row).encode())
            if key in heldout or stem_digest(row) in heldout_stems:
                leaked += 1
                continue
            cursor = connection.execute("INSERT OR IGNORE INTO questions VALUES(?,?,?,?,?,?,?,?)",
                (key, row["queId"], row["problem"], json.dumps(row["answer_option_list"], ensure_ascii=False),
                 row["answer_value"], json.dumps(row["answer_analysis"], ensure_ascii=False), TAL_REVISION, "train"))
            duplicates += cursor.rowcount == 0
            if cursor.rowcount:
                from agentic_rag.math_taxonomy import tokenize_math
                tokens = tokenize_math(row["problem"])
                connection.execute("INSERT INTO question_search VALUES(?,?)", (key, " ".join(tokens)))
        actual = connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    return {"source": "math-eval/TAL-SCQ5K", "revision": TAL_REVISION,
            "license": "MIT per publisher README; upstream problem rights not independently audited",
            "scope_evidence": "Publisher dataset_name starts with mid_math_; not inferred from LLM difficulty",
            "raw_level_counts": counts, "source_sha256": hashes,
            "junior_train_rows": len(rows["train"]), "junior_test_rows": len(rows["test"]),
            "unique_train_questions": actual, "unique_test_questions": len(heldout),
            "train_test_overlaps_excluded": leaked, "duplicate_train_rows_excluded": duplicates,
            "dedup_scope": "NFKC/whitespace normalized question + unordered choice contents; train/test also isolated by stem; not semantic/translation dedup",
            "retrieval_database_built": True, "serving_enabled": False, "database": display_path(database),
            "target_unique_public_junior_competition_questions": 10000000,
            "qualified_count_including_heldout": actual + len(heldout),
            "target_met": actual + len(heldout) >= 10000000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / ".runtime/public-corpus")
    parser.add_argument("--reserve-gib", type=float, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/reports/target-20261020-sources.json")
    parser.add_argument("--offline", action="store_true", help="Verify downloaded snapshots against the previous audit and rebuild locally")
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.offline:
        report = json.loads(args.output.read_text())
        for resource in report['downloads'] + [r for candidate in report['candidates'] for r in candidate.get('resources', [])]:
            path = Path(resource['path'])
            path = path if path.is_absolute() else ROOT/path
            if digest(path.read_bytes()) != resource['sha256']:
                raise ValueError('Downloaded snapshot hash changed: '+str(path))
        report['summary'] = build_corpus(args.directory, args.directory/'train-only.sqlite3')
        report['generated_at'] = datetime.now(timezone.utc).isoformat()
        report['refresh_mode'] = 'Verified previous source snapshots locally; no new source claims'
        state = ROOT/'.runtime/stack.json'
        if state.is_file():
            pid = json.loads(state.read_text()).get('processes', {}).get('grpc', {}).get('pid')
            process = Path(f'/proc/{pid}/environ')
            if process.exists():
                running = dict(item.split('=', 1) for item in process.read_text().split('\0') if '=' in item)
                report['summary']['serving_enabled'] = running.get('PUBLIC_CASE_CORPUS_PATH') == str(args.directory/'train-only.sqlite3')
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
        return
    records = []
    reserve = int(args.reserve_gib * 1024**3)
    for path in ("README.md", "ch_single_choice_constructed_5K/ch_single_choice_train_3K.jsonl",
                 "ch_single_choice_constructed_5K/ch_single_choice_test_2K.jsonl"):
        url = f"https://api.github.com/repos/math-eval/TAL-SCQ5K/contents/{path}?ref={TAL_REVISION}"
        records.append(fetch(url, args.directory / path, reserve_bytes=reserve, accept="application/vnd.github.raw+json"))
    corpus = build_corpus(args.directory, args.directory / "train-only.sqlite3")
    candidates = []
    for repository in HF_SOURCES:
        attempts, metadata = [], None
        for endpoint in ("https://huggingface.co", "https://hf-mirror.com"):
            try:
                target = args.directory / (repository.replace("/", "-") + "-metadata.json")
                resource = fetch(endpoint + "/api/datasets/" + repository, target, reserve_bytes=reserve)
                metadata = json.loads(target.read_text())
                card = metadata.get("cardData", {})
                info = card.get("dataset_info", {})
                if isinstance(info, list):
                    info = {"configs": info}
                card_resource = fetch(endpoint + f"/datasets/{repository}/resolve/{metadata['sha']}/README.md",
                    target.with_suffix(".README.md"), reserve_bytes=reserve, accept="text/plain")
                candidates.append({"repository": repository, "revision": metadata["sha"],
                    "license": card.get("license"), "dataset_info": info,
                    "description": metadata.get("description", ""),
                    "resources": [resource, card_resource], "download_attempts": attempts,
                    "qualified_junior_count": 0,
                    "eligibility": "not counted: no validated junior-only subset and independent unique-problem count",
                    "bulk_downloaded": False})
                break
            except (OSError, ValueError) as exc:
                attempts.append({"endpoint": endpoint, "error": type(exc).__name__ + ": " + str(exc)})
        else:
            candidates.append({"repository": repository, "qualified_junior_count": 0, "download_attempts": attempts})
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "summary": corpus,
              "downloads": records, "candidates": candidates,
              "policy": "No paywall/captcha bypass, unlicensed scraping, synthetic padding, or counting solutions as unique questions"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "summary": corpus}, ensure_ascii=False))


if __name__ == "__main__":
    main()
