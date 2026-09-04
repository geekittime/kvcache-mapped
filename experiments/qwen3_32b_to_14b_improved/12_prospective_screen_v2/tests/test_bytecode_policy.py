import os
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / ".formal_empty_pycache"
SCRIPTS = tuple(
    ROOT / "scripts" / name
    for name in (
        "check_readiness.py",
        "prepare.py",
        "freeze_run.py",
        "run_gpu.py",
        "seal_run.py",
        "validate_run.py",
    )
)


def _env_without_policy() -> dict[str, str]:
    value = os.environ.copy()
    value.pop("PYTHONDONTWRITEBYTECODE", None)
    value.pop("PYTHONPYCACHEPREFIX", None)
    return value


def _env_with_policy(prefix: Path = PREFIX) -> dict[str, str]:
    value = os.environ.copy()
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    value["PYTHONPYCACHEPREFIX"] = str(prefix)
    return value


def _run_help(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_formal_entrypoints_check_policy_before_local_src_import():
    constants = set()
    for script in SCRIPTS:
        text = script.read_text(encoding="utf-8")
        marker = '_FORMAL_PYCACHE_PREFIX = "'
        start = text.index(marker) + len(marker)
        constants.add(text[start : text.index('"', start)])
        guard = text.index("\n_require_formal_bytecode_policy()\n")
        local_import = text.index("from src.")
        assert guard < local_import, script
    assert constants == {str(PREFIX)}


def test_all_formal_entrypoints_fail_closed_without_policy():
    for script in SCRIPTS:
        result = _run_help(script, _env_without_policy())
        assert result.returncode != 0, (script, result.stdout, result.stderr)
        assert "bytecode policy" in (result.stdout + result.stderr)


def test_entrypoint_rejects_wrong_prefix_even_when_no_write_is_enabled():
    env = _env_without_policy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / ".wrong-formal-prefix")
    result = _run_help(SCRIPTS[0], env)
    assert result.returncode != 0
    assert "bytecode policy" in (result.stdout + result.stderr)


def test_all_formal_entrypoints_run_help_with_exact_absent_prefix():
    assert not os.path.lexists(PREFIX)
    for script in SCRIPTS:
        result = _run_help(script, _env_with_policy())
        assert result.returncode == 0, (script, result.stdout, result.stderr)
    assert not os.path.lexists(PREFIX)


def test_entrypoint_rejects_existing_or_dangling_exact_prefix():
    PREFIX.mkdir()
    try:
        result = _run_help(SCRIPTS[0], _env_with_policy())
        assert result.returncode != 0
        assert "must be absent" in (result.stdout + result.stderr)
    finally:
        PREFIX.rmdir()

    PREFIX.symlink_to(ROOT / ".prefix-target-does-not-exist")
    try:
        assert os.path.lexists(PREFIX)
        result = _run_help(SCRIPTS[0], _env_with_policy())
        assert result.returncode != 0
        assert "symlink" in (result.stdout + result.stderr)
    finally:
        PREFIX.unlink()


def _write_timestamp_pyc_pair(root: Path) -> Path:
    source = root / "transitive_probe.py"
    cache = root / "__pycache__"
    cache.mkdir()
    good = b"VALUE = 'source-import'\n"
    bad = b"VALUE = 'pyc-import!!!'\n"
    assert len(good) == len(bad)
    source.write_bytes(bad)
    timestamp = 1_700_000_321
    os.utime(source, (timestamp, timestamp))
    pyc = cache / f"transitive_probe.cpython-{sys.version_info[0]}{sys.version_info[1]}.pyc"
    py_compile.compile(str(source), cfile=str(pyc), doraise=True)
    source.write_bytes(good)
    os.utime(source, (timestamp, timestamp))
    return source


def test_fresh_prefix_blocks_transitive_timestamp_pyc_reads(tmp_path):
    _write_timestamp_pyc_pair(tmp_path)
    fresh_prefix = tmp_path / "fresh-prefix-that-must-stay-absent"
    env = _env_with_policy(fresh_prefix)
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", "import transitive_probe; print(transitive_probe.VALUE)"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "source-import"
    assert not os.path.lexists(fresh_prefix)
