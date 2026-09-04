"""Create self-contained, append-only experiment run directories."""

from __future__ import annotations

import datetime as dt
import os
import platform
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .io import atomic_write_json


@dataclass(frozen=True)
class RunDirectory:
    path: Path
    run_id: str


def make_run_directory(project_root: str | Path, stage: str, name: str) -> RunDirectory:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = lambda value: "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    run_id = f"{timestamp}_{safe(stage)}_{safe(name)}"
    path = Path(project_root) / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return RunDirectory(path=path, run_id=run_id)


def record_run_start(run: RunDirectory, argv: list[str], config: dict) -> None:
    (run.path / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(argv) + "\n",
        encoding="utf-8",
    )
    atomic_write_json(run.path / "config.json", config)
    atomic_write_json(
        run.path / "environment.json",
        {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
    )


def append_reproduction_log(
    project_root: str | Path,
    *,
    heading: str,
    bullets: list[str],
) -> None:
    """Append a concise stage summary to the living Markdown ledger."""
    path = Path(project_root) / "REPRODUCTION.md"
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    block = [f"\n### {heading}\n", f"- UTC：{timestamp}\n"]
    block.extend(f"- {bullet}\n" for bullet in bullets)
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(block)
        handle.flush()
        os.fsync(handle.fileno())

