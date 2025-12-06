"""
Quickly verify that all Hugging Face model repos referenced by this package are reachable.

It scans `src/nodetool/package_metadata/nodetool-huggingface.json` for `repo_id` fields,
then issues lightweight HEAD requests to `https://huggingface.co/api/models/<repo_id>`.

Usage:
    python scripts/check_hf_models.py [--token YOUR_HF_TOKEN] [--workers 16] [--timeout 5]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_METADATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "nodetool"
    / "package_metadata"
    / "nodetool-huggingface.json"
)


def iter_repo_ids(metadata_path: Path) -> set[str]:
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    repo_ids: set[str] = set()

    def visit(obj: object) -> None:
        if isinstance(obj, dict):
            repo_id = obj.get("repo_id")
            if isinstance(repo_id, str):
                repo_ids.add(repo_id)
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(data)
    return repo_ids


def check_repo(
    repo_id: str,
    token: str | None,
    timeout: float,
    retries: int,
    backoff: float,
) -> tuple[str, bool, str]:
    url = f"https://huggingface.co/api/models/{repo_id}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries + 1):
        try:
            resp = requests.head(
                url, headers=headers, timeout=timeout, allow_redirects=True
            )
            if resp.status_code == 200:
                return repo_id, True, "ok"
            if resp.status_code == 429 and attempt < retries:
                delay = backoff * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue
            return repo_id, False, f"status {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            if attempt < retries:
                delay = backoff * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue
            return repo_id, False, str(exc)


def main(
    metadata_path: Path,
    token: str | None,
    workers: int,
    timeout: float,
    retries: int,
    backoff: float,
    log_all: bool,
) -> int:
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}", file=sys.stderr)
        return 1

    repo_ids = iter_repo_ids(metadata_path)
    print(f"Found {len(repo_ids)} unique repo_ids in {metadata_path}")

    unreachable: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for repo_id, ok, detail in pool.map(
            lambda rid: check_repo(rid, token, timeout, retries, backoff),
            sorted(repo_ids),
        ):
            if ok:
                print(f"[OK]   {repo_id}")
                continue
            unreachable.append((repo_id, detail))
            print(f"[FAIL] {repo_id}: {detail}")

    total = len(repo_ids)
    failed = len(unreachable)
    succeeded = total - failed
    print(f"Checked {total} repos: {succeeded} ok, {failed} failed")

    if not unreachable:
        print("All repos reachable ✅")
        return 0

    print(f"{len(unreachable)} repos unreachable:")
    for repo_id, detail in unreachable:
        print(f"  - {repo_id}: {detail}")
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check HF model reachability.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to metadata JSON with repo_id fields.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HF token for gated models (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent requests (lower to avoid 429).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per repo on errors/429.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Base backoff (seconds) for retries; doubles each attempt.",
    )
    parser.add_argument(
        "--log-all",
        action="store_true",
        help="Log every repo status instead of just failures.",
    )
    args = parser.parse_args()
    sys.exit(
        main(
            metadata_path=args.metadata,
            token=args.token,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            backoff=args.backoff,
            log_all=args.log_all,
        )
    )

