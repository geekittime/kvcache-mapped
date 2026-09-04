import json

from kvcache_transfer.io import atomic_write_json, sha256_file


def test_atomic_json_and_hash(tmp_path) -> None:
    path = tmp_path / "nested" / "value.json"
    atomic_write_json(path, {"z": 1, "text": "测试"})
    assert json.loads(path.read_text()) == {"z": 1, "text": "测试"}
    assert len(sha256_file(path)) == 64

