import sys
from types import SimpleNamespace

import pytest

from tools import reproduce_experiments as experiments
from tools import verify_reproduction_traces as traces


@pytest.mark.parametrize("passed,exit_code", [(False, 1), (True, 0)])
def test_smoke_command_propagates_acceptance_failure(monkeypatch, tmp_path, passed, exit_code):
    monkeypatch.setattr(sys, "argv", ["reproduce_experiments.py", "smoke"])
    monkeypatch.setattr(experiments, "REPORTS", tmp_path)
    monkeypatch.setattr(experiments, "smoke", lambda: passed)
    assert experiments.main() == exit_code


def test_smoke_identity_preserves_follow_up_scope_and_isolates_runs():
    first = experiments.experiment_identity(70001, "first-run")
    assert experiments.experiment_identity(70001, "first-run") == first
    assert experiments.experiment_identity(70002, "first-run")[1] != first[1]
    second = experiments.experiment_identity(70001, "second-run")
    assert first[0] != second[0] and first[1] != second[1]


def test_trace_acceptance_rejects_empty_spans_and_never_replays_previous_ask(monkeypatch, tmp_path):
    (tmp_path / ".runtime/traces").mkdir(parents=True)
    (tmp_path / ".runtime/gateway.log").write_text("")
    monkeypatch.setattr(traces, "ROOT", tmp_path)
    monkeypatch.setenv("AUTH_SHARED_SECRET", "acceptance-test-secret")
    requests, reports = [], []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, path, **kwargs):
            if path == "/ask":
                requests.append(kwargs["json"])
            return SimpleNamespace(status_code=200, json=lambda: {"task_id": "test-task"})

        def get(self, *args, **kwargs):
            return SimpleNamespace(json=lambda: {"status": "done"})

    monkeypatch.setattr(traces.httpx, "Client", Client)
    monkeypatch.setattr(traces, "write", lambda name, data: reports.append(data))
    assert traces.main() == traces.main() == 1
    assert all(not report["summary"]["passed"] for report in reports)
    assert requests[0]["request_id"] != requests[1]["request_id"]
    assert requests[0]["user_id"] != requests[1]["user_id"]
    assert requests[0]["session_id"] != requests[1]["session_id"]
