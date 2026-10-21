#!/usr/bin/env python3
"""Fetch pinned public resources and preserve their download provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import subprocess
import os
import time
import urllib.request


MYSQL_VERSION = "8.4.6"
MYSQL_ARCHIVE = f"mysql-{MYSQL_VERSION}-linux-glibc2.28-x86_64-minimal.tar.xz"
MYSQL_URL = f"https://cdn.mysql.com/archives/mysql-8.4/{MYSQL_ARCHIVE}"
REVISIONS = {
    "ShichaoSun/math_seq2tree": "d2310d08818ef073d75a3e736f2c0ee0c0aac02a",
    "XiaoMi/cmath": "9fa13ef0b2d03f5a18c1ace7001248d3981d65ea",
    "math-eval/TAL-SCQ5K": "8a66d6e14c65ba002beda39747e6b7c43bd2d472",
}
MODELS = {
    "qwen2.5:7b-instruct": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
    "bge-m3:567m": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab",
}
PACKAGE_HASHES = {
    "libaio1_0.3.112-13build1_amd64.deb": "2dcfe0b49d7cccfbf1bd6a3f627cf44bea9f51fb32f7e0e552a0df75698e0d27",
    "redis-server_7.0.15-1ubuntu0.24.04.5_amd64.deb": "5f41607fb77674078c013b2a6196a748c3a5f6ec4bd3201cd08478dce3e26d05",
    "redis-tools_7.0.15-1ubuntu0.24.04.5_amd64.deb": "6e5c009ce7761d0a956dafcfcf054ddeecc7a51efcc38215f26bb5794a4e93e1",
}


def download(url: str, destination: Path, *, accept: str = "application/json") -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "math-agent-reproduction/1.0", "Accept": accept})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".runtime/resources"))
    parser.add_argument("--mysql", action="store_true")
    parser.add_argument("--native-deps", action="store_true", help="download user-local libaio and Redis Ubuntu packages")
    parser.add_argument("--models", action="store_true", help="pull pinned Ollama models into the isolated model server")
    parser.add_argument("--github", action="append", default=[], metavar="OWNER/REPO:PATH")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.native_deps:
        packages = {
            "libaio": ["main/liba/libaio/libaio1_0.3.112-13build1_amd64.deb"],
            "redis": [f"universe/r/redis/{package}_7.0.15-1ubuntu0.24.04.5_amd64.deb" for package in ("redis-server", "redis-tools")],
        }
        for resource, paths in packages.items():
            records = []
            for path in paths:
                url = "https://archive.ubuntu.com/ubuntu/pool/"+path
                destination = args.directory / Path(path).name
                body = destination.read_bytes() if destination.exists() else download(url, destination)
                if hashlib.sha256(body).hexdigest() != PACKAGE_HASHES[destination.name]:
                    raise ValueError(f"Package checksum mismatch: {destination.name}")
                subprocess.run(["dpkg-deb", "-x", str(destination), str(args.directory / resource)], check=True)
                records.append({"url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
            manifest[resource] = {"packages": records, "directory": str((args.directory / resource).resolve())}
            print(json.dumps({"resource": resource, **manifest[resource]}), flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    if args.models:
        base = "http://127.0.0.1:21434"
        with urllib.request.urlopen(base+"/api/tags", timeout=10) as response:
            tags = {item["name"]: item for item in json.load(response)["models"]}
        for model, digest in MODELS.items():
            if model not in tags:
                subprocess.run(["ollama", "pull", model], env=dict(os.environ, OLLAMA_HOST=base), check=True)
        with urllib.request.urlopen(base+"/api/tags", timeout=10) as response:
            tags = {item["name"]: item for item in json.load(response)["models"]}
        for model, digest in MODELS.items():
            if tags[model]["digest"] != digest:
                raise ValueError(f"Model tag changed; expected pinned digest for {model}")
            manifest["model:"+model] = {**tags[model], "registry": "registry.ollama.ai/library/"+model,
                                        "verification": "Ollama content-addressed blob verification and pinned manifest digest"}
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    if args.mysql:
        destination = args.directory / MYSQL_ARCHIVE
        body = destination.read_bytes() if destination.exists() else download(MYSQL_URL, destination)
        checksum = download(MYSQL_URL + ".md5", destination.with_suffix(destination.suffix + ".md5"))
        expected = checksum.decode().strip().split()[0]
        if hashlib.md5(body).hexdigest() != expected:
            raise ValueError("Official MySQL archive checksum mismatch")
        if hashlib.sha256(body).hexdigest() != "f284b17b9e038adbe77f0dd5fb11ed30262286b23a390b8b4e367abc3574c42e":
            raise ValueError("Pinned MySQL SHA256 mismatch")
        extracted = args.directory / MYSQL_ARCHIVE.removesuffix(".tar.xz")
        if not extracted.exists():
            with tarfile.open(destination) as archive:
                archive.extractall(args.directory, filter="data")
        manifest["mysql"] = {
            "version": MYSQL_VERSION, "url": MYSQL_URL, "official_md5": expected,
            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
            "directory": str(extracted.resolve()),
        }
        print(json.dumps({"resource": "mysql", **manifest["mysql"]}), flush=True)
    for reference in args.github:
        repository, path = reference.split(":", 1)
        api = f"https://api.github.com/repos/{repository}/commits/HEAD"
        revision = REVISIONS.get(repository) or manifest.get(reference, {}).get("revision")
        if not revision:
            revision = json.loads(download(api, args.directory / f"{repository.replace('/', '-')}-revision.json"))["sha"]
        url = f"https://api.github.com/repos/{repository}/contents/{path}?ref={revision}"
        destination = args.directory / repository.replace("/", "-") / path
        body = download(url, destination, accept="application/vnd.github.raw+json")
        manifest[reference] = {"repository": repository, "revision": revision, "path": path,
                               "url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                               "local_path": str(destination.resolve())}
        print(json.dumps({"resource": reference, **manifest[reference]}), flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
