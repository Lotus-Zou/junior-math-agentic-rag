import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tools.evening_capacity_test import acceptance
from tools.gpu_cluster_preflight import inspect_nodes
from tools.realistic_load_test import read_sse, summarize
from tools.review_gpu_images import Registry, checked, resolve


ROOT = Path(__file__).resolve().parents[1]


def node(runtime="containerd://2.0.0", gpu="1", ready="True"):
    return {"metadata": {"name": "gpu-node"}, "status": {
        "nodeInfo": {"containerRuntimeVersion": runtime},
        "allocatable": {"nvidia.com/gpu": gpu},
        "conditions": [{"type": "Ready", "status": ready}],
    }}


def test_simulator_cannot_pass_even_with_fake_gpu_capacity():
    result = inspect_nodes([node("kwok-v0.8.0")])
    assert not result["passed"]
    assert any("KWOK" in error for error in result["errors"])


@pytest.mark.parametrize("nodes", [[], [node(gpu="0")], [node(ready="False")], [node(runtime="")]])
def test_gpu_preflight_rejects_missing_capacity_or_runtime(nodes):
    assert not inspect_nodes(nodes)["passed"]


def test_device_plugin_install_can_start_before_gpu_is_registered():
    assert inspect_nodes([node(gpu="0")], require_gpu=False)["passed"]
    assert inspect_nodes([node()])["passed"]


def test_oci_artifact_hash_mismatch_is_rejected():
    with pytest.raises(ValueError, match="integrity mismatch"):
        checked(b"unexpected artifact", "sha256:" + "0" * 64)


def test_image_resolution_uses_linux_amd64_and_validates_config():
    config = json.dumps({"os": "linux", "architecture": "amd64", "config": {}}).encode()
    config_hash = checked(config, None)
    manifest = json.dumps({"config": {"digest": config_hash}, "layers": [{"size": 123}]}).encode()
    manifest_hash = checked(manifest, None)
    index = json.dumps({"manifests": [
        {"platform": {"architecture": "unknown", "os": "unknown"}, "digest": "attestation"},
        {"platform": {"architecture": "amd64", "os": "linux"}, "digest": manifest_hash},
    ]}).encode()
    with patch.object(Registry, "get", side_effect=[(index, checked(index, None)), (manifest, manifest_hash), (config, config_hash)]):
        result = resolve("example.com/model:v1")
    assert result["platform"] == "linux/amd64"
    assert result["compressed_layer_bytes"] == 123
    assert result["vulnerability_scan"] == "pending"
    assert result["runtime_gpu_validation"] == "pending"


def test_sse_done_after_error_is_not_a_success():
    stream = io.BytesIO(b'event: error\ndata: {"code":"Unavailable"}\n\nevent: done\ndata: {"ok":false}\n\n')
    result = read_sse(stream, 0)
    assert result["events"] == ["error", "done"]
    assert not result["business_success"]


def test_sse_measures_deltas_before_final_answer():
    stream = io.BytesIO(b'event: answer\ndata: {"delta":"x"}\n\nevent: answer\ndata: {"delta":"=4"}\n\nevent: result\ndata: {"answer":"x=4","response_type":"verified_answer","validation_passed":true}\n\nevent: done\ndata: {}\n\n')
    with patch("tools.realistic_load_test.time.perf_counter", side_effect=range(1, 20)):
        result = read_sse(stream, 0)
    assert result["business_success"]
    assert result["delta_events"] == 2
    assert result["ttft_ms"] < 10000
    assert result["first_event_ms"] <= result["ttft_ms"]


def test_sse_eof_without_result_or_done_is_not_a_success():
    assert not read_sse(io.BytesIO(b'event: answer\ndata: {"delta":"x"}\n\n'), 0)["business_success"]


def test_status_200_does_not_override_business_failure_budget():
    rows = [{"status": 200, "latency_ms": 1, "stream": True, "business_success": False, "delta_events": 0}]
    stage = {"summary": summarize(rows), "completed_answers_per_second": 0, "sse_requests": 1}
    assert stage["summary"]["success_rate"] == 1
    assert not acceptance(stage, 1, 1000, .1)["passed"]


def model_downloader():
    spec = importlib.util.spec_from_file_location("model_download", ROOT / "deploy/k8s/model-artifacts/download.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_download_checks_hash_before_publishing(tmp_path):
    module = model_downloader()
    expected = b'{"model_type":"qwen2"}'
    manifest = {"repository": "Qwen/model", "revision": "fixed", "files": [
        {"path": "config.json", "bytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest()}
    ]}
    with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(expected)):
        module.download(manifest, tmp_path, reserve_bytes=0)
    assert (tmp_path / "config.json").read_bytes() == expected
    # A matching existing snapshot must not require network access.
    with patch.object(module.urllib.request, "urlopen", side_effect=AssertionError("network")):
        module.download(manifest, tmp_path, reserve_bytes=0)


def test_corrupt_weight_download_never_replaces_existing_file(tmp_path):
    module = model_downloader()
    (tmp_path / "config.json").write_bytes(b"existing")
    manifest = {"repository": "Qwen/model", "revision": "fixed", "files": [
        {"path": "config.json", "bytes": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
    ]}
    with patch.object(module.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(b"bad")), patch.object(module.time, "sleep"):
        with pytest.raises(ValueError, match="mismatch"):
            module.download(manifest, tmp_path, reserve_bytes=0)
    assert (tmp_path / "config.json").read_bytes() == b"existing"


@pytest.mark.parametrize("name", ["../model.safetensors", "model.py"])
def test_model_artifact_rejects_path_escape_and_executable_code(tmp_path, name):
    with pytest.raises(ValueError):
        model_downloader().download({"files": [{"path": name}]}, tmp_path)


@pytest.mark.skipif(not shutil.which("kubectl"), reason="kubectl is needed for Kustomize rendering")
@pytest.mark.parametrize("profile", ["deploy/k8s", "deploy/overlays/single-gpu"])
def test_rendered_gpu_stack_has_locked_images_and_valid_single_card_limits(profile):
    rendered = subprocess.check_output(["kubectl", "kustomize", str(ROOT / profile)], text=True)
    resources = list(yaml.safe_load_all(rendered))
    deployments = {r["metadata"]["name"]: r for r in resources if r["kind"] == "Deployment"}
    locks = {data["pinned"] for p in (ROOT / "deploy/k8s/artifacts").glob("*.json") if "pinned" in (data := json.loads(p.read_text()))}
    local_refs = {"math-agent-ai:local", "math-agent-gateway:local"}
    for resource in resources:
        if resource["kind"] not in {"Deployment", "DaemonSet"}:
            continue
        pod = resource["spec"]["template"]["spec"]
        for container in pod.get("containers", []) + pod.get("initContainers", []):
            assert container["image"] in locks | local_refs
    model = deployments["math-agent-model"]["spec"]
    pod = model["template"]["spec"]
    assert pod["runtimeClassName"] == "nvidia"
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod["containers"][0]["startupProbe"]["failureThreshold"] >= 120
    metrics = deployments["metrics-server"]["spec"]["template"]["spec"]["containers"][0]["args"]
    hpa = next(r for r in resources if r["kind"] == "HorizontalPodAutoscaler" and r["metadata"]["name"] == "math-agent-model")
    assert hpa["spec"]["metrics"][0]["pods"]["metric"]["name"] == "vllm_num_requests_waiting"
    if "single-gpu" in profile:
        assert model["replicas"] == 1
        assert model["strategy"]["rollingUpdate"]["maxSurge"] == 0
        assert hpa["spec"]["minReplicas"] == hpa["spec"]["maxReplicas"] == 1
        assert "--kubelet-insecure-tls" in metrics
    else:
        assert model["replicas"] == 2
        assert "--kubelet-insecure-tls" not in metrics
