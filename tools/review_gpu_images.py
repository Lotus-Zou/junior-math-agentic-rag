#!/usr/bin/env python3
"""Resolve immutable OCI artifacts and record reproducible metadata evidence.

This does not execute the image or replace a vulnerability/signature scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


class Registry:
    def __init__(self, host: str, repository: str, timeout: float = 20):
        self.host, self.repository, self.timeout = host, repository, timeout
        self.token = None

    def get(self, kind: str, reference: str) -> tuple[bytes, str | None]:
        url = f"https://{self.host}/v2/{self.repository}/{kind}/{reference}"
        headers = {"Accept": ACCEPT, "User-Agent": "math-agent-artifact-review/1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=self.timeout) as response:
                return response.read(), response.headers.get("Docker-Content-Digest")
        except urllib.error.HTTPError as exc:
            if exc.code != 401 or self.token:
                raise
            challenge = exc.headers.get("WWW-Authenticate", "")
            if not challenge.lower().startswith("bearer "):
                raise ValueError("Registry did not offer bearer authentication") from exc
            fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
            realm = fields.pop("realm")
            if urllib.parse.urlparse(realm).scheme != "https":
                raise ValueError("Refusing non-TLS registry authentication")
            fields.setdefault("scope", f"repository:{self.repository}:pull")
            with urllib.request.urlopen(realm + "?" + urllib.parse.urlencode(fields), timeout=self.timeout) as response:
                auth = json.load(response)
            self.token = auth.get("token") or auth["access_token"]
            return self.get(kind, reference)


def checked(data: bytes, expected: str | None) -> str:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"Artifact integrity mismatch: {digest} != {expected}")
    return digest


def resolve(reference: str, source: str | None = None) -> dict:
    host, path = (source or reference).split("/", 1)
    repository, tag = path.rsplit(":", 1)
    registry = Registry(host, repository)
    raw, header_digest = registry.get("manifests", tag)
    root_digest = checked(raw, header_digest)
    root = json.loads(raw)
    manifest, manifest_digest = root, root_digest
    if "manifests" in root:
        descriptor = next(item for item in root["manifests"] if item.get("platform", {}).get("architecture") == "amd64" and item.get("platform", {}).get("os") == "linux")
        raw, _ = registry.get("manifests", descriptor["digest"])
        manifest_digest = checked(raw, descriptor["digest"])
        manifest = json.loads(raw)
    config_raw, _ = registry.get("blobs", manifest["config"]["digest"])
    checked(config_raw, manifest["config"]["digest"])
    config = json.loads(config_raw)
    if (config.get("os"), config.get("architecture")) != ("linux", "amd64"):
        raise ValueError("Expected linux/amd64 image")
    original_repo = reference.rsplit(":", 1)[0]
    env = config.get("config", {}).get("Env", [])
    return {
        "upstream": reference,
        "fetched_from": source or reference,
        "pinned": f"{original_repo}@{root_digest}",
        "mirror_pinned": f"{host}/{repository}@{root_digest}",
        "manifest_digest": manifest_digest,
        "config_digest": manifest["config"]["digest"],
        "platform": "linux/amd64",
        "created": config.get("created"),
        "compressed_layer_bytes": sum(item["size"] for item in manifest["layers"]),
        "entrypoint": config.get("config", {}).get("Entrypoint"),
        "cuda_environment": [item for item in env if item.startswith(("CUDA_VERSION=", "NVIDIA_REQUIRE_CUDA="))],
        "integrity": "manifest and config SHA-256 verified",
        "source_status": "upstream registry" if source is None else "mirror; upstream signature verification pending",
        "vulnerability_scan": "pending",
        "runtime_gpu_validation": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Official registry/repository:tag")
    parser.add_argument("--source", help="Optional explicit mirror registry/repository:tag")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = resolve(args.image, args.source)
    result["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["pinned"], flush=True)


if __name__ == "__main__":
    main()
