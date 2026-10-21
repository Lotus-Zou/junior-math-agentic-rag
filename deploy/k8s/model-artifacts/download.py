#!/usr/bin/env python3
"""Materialize a hash-locked safetensors snapshot from ModelScope over TLS."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(manifest: dict, destination: Path, reserve_bytes: int = 5 * 1024**3) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for entry in manifest["files"]:
            name = entry["path"]
            if Path(name).name != name or name.startswith("."):
                raise ValueError("Only top-level model files are allowed")
            if not name.endswith((".safetensors", ".json", ".txt")) and name != "LICENSE":
                raise ValueError("Executable model artifacts are forbidden")
            target = destination / name
            if target.exists() and target.stat().st_size == entry["bytes"] and digest(target) == entry["sha256"]:
                print(f"Verified cached {name}", flush=True)
                continue
            if shutil.disk_usage(destination).free < entry["bytes"] + reserve_bytes:
                raise RuntimeError(f"Insufficient free disk space for {name} plus reserve")
            query = urllib.parse.urlencode({"Revision": manifest["revision"], "FilePath": name})
            url = f"https://modelscope.cn/api/v1/models/{manifest['repository']}/repo?{query}"
            temporary = target.with_suffix(target.suffix + ".part")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                        shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
                    if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
                        raise ValueError(f"Size or SHA-256 mismatch for {name}")
                    os.replace(temporary, target)
                    print(f"Downloaded and verified {name}", flush=True)
                    break
                except Exception:
                    temporary.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    download(json.loads(args.manifest.read_text(encoding="utf-8")), args.destination)


if __name__ == "__main__":
    main()
