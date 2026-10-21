#!/usr/bin/env python3
"""Check evidence completeness without inventing provenance or authenticating people."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


KINDS = {'annotations', 'school_authorization', 'school_operations', 'historical_before', 'historical_after'}


def timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise ValueError('Evidence dates require a timezone and cannot be in the future')
    return parsed


def audit(directory: Path, manifest: Path) -> dict:
    directory = directory.resolve()
    result = {'artifact_count': 0, 'artifacts': [], 'errors': [],
              'claims': {}, 'authenticity_verified': False,
              'policy': 'Hashes prove file integrity, not human identity, school adoption or historical truth; independent review is required.'}
    entries = json.loads(manifest.read_text()).get('artifacts', []) if manifest.is_file() else []
    groups = defaultdict(list)
    for entry in entries:
        try:
            kind = entry['kind']
            if kind not in KINDS:
                raise ValueError('Unknown evidence kind')
            relative = Path(entry['path'])
            path = (directory/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(directory):
                raise ValueError('Evidence path must remain inside its private intake directory')
            body = path.read_bytes()
            actual = hashlib.sha256(body).hexdigest()
            if actual != entry['sha256']:
                raise ValueError('SHA256 mismatch')
            if not entry.get('origin'):
                raise ValueError('Missing origin, cannot distinguish third-party from project evidence')
            timestamp(entry['recorded_at'])
            metadata = {key: entry[key] for key in ('kind', 'path', 'sha256', 'origin', 'recorded_at')}
            metadata['bytes'] = len(body)
            result['artifacts'].append(metadata)
            groups[kind].append((entry, path))
        except (KeyError, OSError, ValueError, TypeError) as exc:
            result['errors'].append({'path': entry.get('path'), 'error': str(exc)})
    result['artifact_count'] = len(result['artifacts'])

    labelled = defaultdict(set)
    annotation_errors = []
    for entry, path in groups['annotations']:
        if entry['origin'] != 'project_primary':
            continue
        try:
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key, person = row['problem_sha256'], row['annotator_id']
                if len(key) != 64 or not person or not row['answer_label'] or not row['error_type']:
                    raise ValueError('Annotation lacks a problem hash, pseudonymous human ID or labels')
                if row.get('method') != 'human':
                    raise ValueError('Model-generated labels are not human annotation evidence')
                timestamp(row['annotated_at'])
                labelled[key].add(person)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            annotation_errors.append(str(exc))
    double_labelled = sum(len(people) >= 2 for people in labelled.values())
    result['claims']['human_annotations'] = {
        'required_unique_questions': 1000, 'unique_questions': len(labelled),
        'independently_double_labelled': double_labelled,
        'structurally_complete': double_labelled >= 1000 and not annotation_errors,
        'errors': annotation_errors, 'verified': False,
        'remaining_review': 'Verify annotator identities, consent, sampling quality and disagreement adjudications.'}

    schools = defaultdict(set)
    for kind in ('school_authorization', 'school_operations'):
        for entry, path in groups[kind]:
            if entry['origin'] == 'project_primary' and entry.get('school_id'):
                schools[entry['school_id']].add(kind)
    matched = sum({'school_authorization', 'school_operations'} <= kinds for kinds in schools.values())
    result['claims']['multi_school_operations'] = {
        'required_schools': 2, 'schools_with_authorization_and_logs': matched,
        'structurally_complete': matched >= 2, 'verified': False,
        'remaining_review': 'Verify school signatories, lawful student-data handling and authentic timestamped production logs; third-party school webpages do not qualify.'}

    history = {}
    history_errors = []
    for kind in ('historical_before', 'historical_after'):
        files = [(entry, path) for entry, path in groups[kind] if entry['origin'] == 'project_primary']
        if len(files) != 1:
            history_errors.append(kind+': exactly one primary raw result artifact is required')
            continue
        try:
            data = json.loads(files[0][1].read_text())
            timestamp(data['run_at'])
            cases = data['cases']
            if not cases or len({row['case_sha256'] for row in cases}) != len(cases):
                raise ValueError('Historical cases must be nonempty and unique')
            scores = [float(row['answer_relevance']) for row in cases]
            if any(not 0 <= score <= 1 for score in scores):
                raise ValueError('Answer relevance scores must be finite and in [0, 1]')
            for key in ('dataset_sha256', 'evaluator_sha256', 'judge_model', 'model', 'prompt_sha256'):
                if not data[key]:
                    raise ValueError('Missing evaluation identity: '+key)
            history[kind] = {**{key: data[key] for key in ('run_at', 'dataset_sha256', 'evaluator_sha256', 'judge_model')},
                             'case_ids': sorted(row['case_sha256'] for row in cases), 'mean': sum(scores)/len(scores)}
        except (OSError, KeyError, ValueError, TypeError) as exc:
            history_errors.append(kind+': '+str(exc))
    paired = len(history) == 2
    if paired:
        before, after = history['historical_before'], history['historical_after']
        paired = all(before[key] == after[key] for key in ('dataset_sha256', 'evaluator_sha256', 'judge_model', 'case_ids'))
        paired = paired and timestamp(before['run_at']) < timestamp(after['run_at'])
        paired = paired and round(before['mean'], 2) == .50 and round(after['mean'], 2) == .78
    result['claims']['historical_050_to_078'] = {
        'structurally_complete': paired and not history_errors, 'verified': False,
        'reported_means': {kind: data['mean'] for kind, data in history.items()}, 'errors': history_errors,
        'remaining_review': 'Confirm original archival timestamps and raw responses; a new prospective A/B result cannot establish a 2025 historical claim.'}
    result['structurally_complete'] = not result['errors'] and all(claim['structurally_complete'] for claim in result['claims'].values())
    result['generated_at'] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('.runtime/project-evidence'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory, args.manifest or args.directory/'manifest.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['structurally_complete'] else 1)


if __name__ == '__main__':
    main()
