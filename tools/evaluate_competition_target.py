#!/usr/bin/env python3
"""Frozen, unique TAL holdout evaluation through the real authenticated gateway."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_public_corpus import canonical_problem, digest, TAL_REVISION
from tools.reproduce_experiments import post_question
from evaluation.public_benchmarks import extract_prediction


def wilson(correct: int, total: int) -> list[float]:
    if not total:
        return [0.0, 1.0]
    z, p = 1.959963984540054, correct / total
    center = (p + z*z/(2*total)) / (1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total)) / (1+z*z/total)
    return [max(0.0, center-half), min(1.0, center+half)]


def select_cases(source: Path, development_report: Path, seed: int, limit: int) -> tuple[list[dict], dict]:
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    records = [row for row in records if row['dataset_name'].startswith('mid_math_')]
    dev = json.loads(development_report.read_text()) if development_report.is_file() else {}
    dev_ids = {row['case_id'] for row in dev.get('results', [])}
    development_hashes = {digest(canonical_problem(row).encode()) for row in records if row['queId'] in dev_ids}
    unique = {}
    conflicts = set()
    for row in records:
        key = digest(canonical_problem(row).encode())
        options = {choice['aoVal']: choice['content'] for group in row['answer_option_list'] for choice in group}
        answer_content = options[str(row['answer_value']).strip()]
        if key in unique and unique[key]['answer_content'] != answer_content:
            conflicts.add(key)
        unique.setdefault(key, {'case_id': row['queId'], 'problem_sha256': key, 'record': row, 'answer_content': answer_content})
    selected = [row for key, row in sorted(unique.items()) if key not in development_hashes]
    # Keep conflicting publisher labels in the denominator, flagged for human audit.
    for row in selected:
        row['publisher_label_conflict'] = row['problem_sha256'] in conflicts
    random.Random(seed).shuffle(selected)
    full_size = len(selected)
    if limit:
        selected = selected[:limit]
    info = {'raw_junior_test_rows': len(records), 'unique_junior_test_questions': len(unique),
            'excluded_previously_inspected_development_questions': len(development_hashes),
            'available_non_development_questions': full_size, 'selected': len(selected),
            'label_conflict_groups': len(conflicts), 'source_sha256': digest(source.read_bytes()),
            'development_report_sha256': digest(development_report.read_bytes()) if development_report.is_file() else None,
            'seed': seed, 'revision': TAL_REVISION, 'pretraining_contamination_unknown': True}
    return selected, info


def summarize(rows: list[dict], total: int, minimum_cases: int) -> dict:
    correct = sum(row['exact_match'] for row in rows)
    completed = len(rows) == total
    interval = wilson(correct, len(rows))
    cache_hits = sum(bool(row.get('response', {}).get('cached')) for row in rows)
    best_possible = (correct + total - len(rows))/total if total else 0.0
    return {'evaluated': len(rows), 'planned': total, 'complete': completed, 'correct': correct,
            'accuracy': correct/len(rows) if rows else 0.0, 'wilson_95': interval,
            'best_possible_final_accuracy': best_possible,
            'point_target_unreachable': best_possible < .9,
            'no_answer_or_unparseable': sum(not row['prediction'] for row in rows),
            'http_failures': sum(row.get('status') != 200 for row in rows),
            'cache_hits': cache_hits,
            'point_target': .90, 'minimum_cases': minimum_cases,
            'point_target_met': completed and not cache_hits and len(rows) >= minimum_cases and correct/max(len(rows), 1) >= .9,
            'statistical_target_met': completed and not cache_hits and len(rows) >= minimum_cases and interval[0] >= .9}


def evaluate(case: dict, index: int) -> dict:
    raw = case['record']
    options = [choice for group in raw['answer_option_list'] for choice in group]
    query = raw['problem']+'\n'+'\n'.join(choice['aoVal']+'. '+choice['content'] for choice in options)
    query += '\n请解释解题思路，最后一行写答案字母，格式为：答案：A。'
    # References/analyses are only used after the service has returned.
    result = post_question(query, index=900000+index)
    prediction = extract_prediction('ceval_middle_school_mathematics', result.get('response', {}).get('answer', ''))
    return {**result, 'case_id': case['case_id'], 'problem_sha256': case['problem_sha256'],
            'publisher_label_conflict': case['publisher_label_conflict'], 'prediction': prediction,
            'expected': str(raw['answer_value']).strip(),
            'exact_match': result['status'] == 200 and bool(result.get('verified')) and bool(prediction) and prediction == str(raw['answer_value']).strip()}


def runtime_identity() -> dict:
    if os.environ.get('FORCE_LLM_EVERY_TURN', '').lower() != 'true':
        raise ValueError('Run the independent evaluation with answer cache disabled')
    files = ['agentic_rag/skill_handlers.py', 'agentic_rag/skill_runtime/pipeline.py',
             'agentic_rag/public_case_retriever.py', 'agentic_rag/pipelines/correction.yaml',
             'tools/evaluate_competition_target.py']
    state = json.loads((ROOT/'.runtime/stack.json').read_text())
    pid = state['processes']['grpc']['pid']
    running_env = dict(item.split('=', 1) for item in Path(f'/proc/{pid}/environ').read_text().split('\0') if '=' in item)
    if running_env.get('FORCE_LLM_EVERY_TURN', '').lower() != 'true':
        raise ValueError('Live inference process has not disabled its answer cache')
    keys = ['LLM_MODEL_NAME', 'CRITIC_MODEL_NAME', 'EMBEDDING_MODEL_NAME',
            'ANSWER_CACHE_VERSION', 'FORCE_LLM_EVERY_TURN', 'PUBLIC_CASE_CORPUS_PATH',
            'RUN_TIMEOUT_SECONDS', 'LLM_CALL_TIMEOUT_SECONDS', 'AI_MAX_INFLIGHT']
    identity = {'configuration': {key: running_env.get(key) for key in keys},
                'code_sha256': {name: digest((ROOT/name).read_bytes()) for name in files}}
    corpus = Path(running_env.get('PUBLIC_CASE_CORPUS_PATH', ''))
    identity['corpus_sha256'] = digest(corpus.read_bytes()) if corpus.is_file() else None
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'.runtime/public-corpus/ch_single_choice_constructed_5K/ch_single_choice_test_2K.jsonl')
    parser.add_argument('--development-report', type=Path, default=ROOT/'evaluation/reports/reconstructed-20261020-public-tal.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--minimum-cases', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--stop-when-target-impossible', action='store_true', help='Stop only for mathematical impossibility, never declare a partial run passed')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.limit < 0 or args.minimum_cases < 1:
        parser.error('workers must be 1..4; limit >= 0; minimum-cases >= 1')
    cases, info = select_cases(args.source, args.development_report, args.seed, args.limit)
    if not cases:
        parser.error('No eligible test questions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selection_path = args.output.with_suffix('.selection.json')
    identity = runtime_identity()
    selection = {'dataset': info, 'runtime': identity, 'case_hashes': [row['problem_sha256'] for row in cases]}
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Frozen selection differs; use a new output path')
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    checkpoint = args.output.with_suffix('.checkpoint.jsonl')
    previous = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    prior = {row['problem_sha256']: row for row in previous}
    if len(prior) != len(previous):
        raise ValueError('Duplicate checkpoint attempt; do not cherry-pick retries')
    if set(prior) - {case['problem_sha256'] for case in cases}:
        raise ValueError('Checkpoint contains questions outside the frozen selection')
    results = [prior[case['problem_sha256']] for case in cases if case['problem_sha256'] in prior]

    def save_report(stop_reason=None):
        report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'summary': summarize(results, len(cases), args.minimum_cases),
                  'dataset': info, 'runtime': identity, 'workers': args.workers, 'selection_sha256': digest(selection_path.read_bytes()),
                  'stop_reason': stop_reason,
                  'policy': 'No cache reuse, answer injection, best-of retries, figure exclusions or test-label retrieval; all timeouts/refusals count wrong',
                  'results': sorted(results, key=lambda row: row['index'])}
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        temp.replace(args.output)
        return report

    remaining = iter((case, index) for index, case in enumerate(cases) if case['problem_sha256'] not in prior)
    stop = args.stop_when_target_impossible and summarize(results, len(cases), args.minimum_cases)['point_target_unreachable']
    with ThreadPoolExecutor(max_workers=args.workers) as pool, checkpoint.open('a') as log:
        futures = set()
        def submit_next():
            if not stop:
                item = next(remaining, None)
                if item is not None:
                    case, index = item
                    futures.add(pool.submit(evaluate, case, index))
        for _ in range(args.workers):
            submit_next()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                futures.remove(future)
                result = future.result()
                results.append(result)
                log.write(json.dumps(result, ensure_ascii=False)+'\n')
                log.flush()
                status = summarize(results, len(cases), args.minimum_cases)
                stop = args.stop_when_target_impossible and status['point_target_unreachable']
                report = save_report('point_target_mathematically_unreachable' if stop else None)
                if len(results) % 20 == 0 or len(results) == len(cases):
                    print(json.dumps(report['summary']), flush=True)
                submit_next()
    save_report('point_target_mathematically_unreachable' if stop else None)
    raise SystemExit(0 if summarize(results, len(cases), args.minimum_cases)['statistical_target_met'] else 1)


if __name__ == '__main__':
    main()
