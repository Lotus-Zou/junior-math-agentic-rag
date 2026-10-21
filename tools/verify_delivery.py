#!/usr/bin/env python3
"""Build or verify an engineering delivery manifest, not product acceptance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "evaluation/reports/release-manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode()


def indexed_paths() -> list[Path]:
    return [ROOT / name for name in git("ls-files", "-z").split("\0") if name]


def check_paths(paths: list[Path]) -> None:
    forbidden = {".runtime", ".git", "node_modules", "chroma_db"}
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Unexpected external path: {relative}")
        if forbidden.intersection(relative.parts) or relative.name == ".env":
            raise ValueError(f"Private/runtime artifact in delivery: {relative}")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError(f"Artifact exceeds 20 MiB: {relative}")


def check_links() -> None:
    indexed = {path.resolve() for path in indexed_paths()}
    documents = sorted(path for path in indexed if path.suffix.lower() == ".md")
    for path in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            local = (path.parent / unquote(parsed.path)).resolve()
            if local not in indexed:
                raise ValueError(f"Broken delivery link in {path.relative_to(ROOT)}: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Refresh the manifest after validation")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Manifest must be inside the repository")
    paths = [p for p in indexed_paths() if p != output]
    check_paths(paths)
    check_links()
    artifacts = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                  "sha256": digest(p)} for p in sorted(paths)]
    if args.check:
        manifest = json.loads(output.read_text())
        if artifacts != manifest["artifacts"]:
            raise ValueError("Delivery files changed; regenerate the manifest after validation")
        if manifest["all_product_targets_passed"] is not False:
            raise ValueError("Delivery manifest must not claim unproven product acceptance")
        print(json.dumps({"manifest_integrity": True, "files": len(artifacts),
                          "all_product_targets_passed": False}))
        return
    if output.exists() and not args.refresh:
        raise FileExistsError(f"Preserve the existing manifest: {output.relative_to(ROOT)}")
    manifest = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                "scope": "Indexed engineering artifacts and public reproduction evidence",
                "all_product_targets_passed": False,
                "limitations": ["Hashes establish integrity, not authenticity or accuracy.",
                                "Deployment and model-quality acceptance are evaluated separately.",
                                "Private runtime, downloaded weights and databases are excluded."],
                "artifacts": artifacts}
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(output.relative_to(ROOT)), "files": len(artifacts),
                      "all_product_targets_passed": False}))


if __name__ == "__main__":
    main()
