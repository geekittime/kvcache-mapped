#!/usr/bin/env python3
"""Parallel, resumable range downloader for a pinned immutable artifact."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-bytes", type=int, default=8 << 20)
    parser.add_argument("--retries", type=int, default=20)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    args = parse_args()
    if len(args.sha256) != 64:
        raise ValueError("--sha256 must be a hexadecimal SHA256 digest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".parts")
    partial.mkdir(parents=True, exist_ok=True)
    ranges = [
        (index, start, min(start + args.chunk_bytes, args.size) - 1)
        for index, start in enumerate(range(0, args.size, args.chunk_bytes))
    ]

    def fetch(item: tuple[int, int, int]) -> tuple[int, int]:
        index, start, end = item
        destination = partial / f"{index:06d}.part"
        expected = end - start + 1
        if destination.is_file() and destination.stat().st_size == expected:
            return index, expected
        for attempt in range(args.retries):
            temporary = destination.with_suffix(".tmp")
            try:
                request = urllib.request.Request(
                    args.url,
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                )
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    if response.status != 206:
                        raise RuntimeError(f"range {index} returned HTTP {response.status}")
                    while block := response.read(1 << 20):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if temporary.stat().st_size != expected:
                    raise RuntimeError(
                        f"range {index} expected {expected} bytes, got {temporary.stat().st_size}"
                    )
                os.replace(temporary, destination)
                return index, expected
            except BaseException:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                if attempt + 1 == args.retries:
                    raise
                time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    completed = 0
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, item) for item in ranges]
        for future in concurrent.futures.as_completed(futures):
            _, count = future.result()
            completed += count
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"downloaded_or_verified={completed}/{args.size} "
                f"rate={completed / elapsed / (1 << 20):.2f} MiB/s",
                flush=True,
            )

    temporary_output = args.output.with_suffix(args.output.suffix + ".assembling")
    with temporary_output.open("wb") as output:
        for index, _, _ in ranges:
            with (partial / f"{index:06d}.part").open("rb") as source:
                while block := source.read(8 << 20):
                    output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if temporary_output.stat().st_size != args.size:
        raise RuntimeError("assembled file has the wrong size")
    observed = digest(temporary_output)
    if observed != args.sha256:
        raise RuntimeError(f"SHA256 mismatch: expected {args.sha256}, observed {observed}")
    os.replace(temporary_output, args.output)
    print(f"complete path={args.output} sha256={observed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

