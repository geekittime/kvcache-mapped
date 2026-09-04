"""CPU-only tests for v2's authorization and durable-result contracts."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cache_oracle import canonical_sha256, intervention_catalog # noqa: E402
from gpu_oracle import GPUOracleError, complete_g0_status, contract_preflight, load_v2, validate_atomic_result # noqa: E402

def _write(path:Path,value:object)->None: path.write_text(json.dumps(value,sort_keys=True),encoding="utf-8")

def test_v2_cpu_preflight_is_blocked_without_complete_g0_and_never_uses_cuda() -> None:
    result, rows=contract_preflight(load_v2(),profile="smoke")
    assert result["status"]=="blocked_missing_complete_g0"
    assert result["gpu_used"] is False and result["cuda_api_called"] is False
    assert len(rows)==4

def test_complete_gate_checks_identity_and_explicit_complete_scope(tmp_path:Path) -> None:
    receipt=tmp_path/"g0.json"; value={"status":"pass","scope":{"closes_complete_g0":True}}; _write(receipt,value)
    cfg={"complete_g0_gate":{"path":str(receipt),"nbytes":receipt.stat().st_size,"sha256":hashlib.sha256(receipt.read_bytes()).hexdigest(),"required_status":"pass","required_closes_complete_g0":True}}
    assert complete_g0_status(cfg)["passed"] is True
    _write(receipt,{"status":"pass","scope":{"closes_complete_g0":False}})
    assert complete_g0_status(cfg)["passed"] is False

def test_atomic_result_rejects_token_trace_tampering(tmp_path:Path) -> None:
    row={"dataset":"hotpotqa_train","sample_id":"s","prompt_token_ids_sha256":"a"*64,"cache_context":{"x":1}}
    ids=[1,2]; expected=[x.id for x in intervention_catalog()]
    traces=[{"intervention_id":name,"generated_token_ids":ids,"generated_token_ids_sha256":canonical_sha256(ids),"generated_token_count":2,"stop_reason":"eos","cap":256} for name in expected]
    value={"status":"complete","request_fingerprint":"f","dataset":"hotpotqa_train","sample_id":"s","prompt_token_ids_sha256":"a"*64,"cache_context":{"x":1},"paired":[{"intervention_id":name} for name in expected],"intervention_generation":traces}
    path=tmp_path/"result.json";_write(path,value); assert validate_atomic_result(path,"f",row)["status"]=="complete"
    value["intervention_generation"][0]["generated_token_ids"]=[9];_write(path,value)
    with pytest.raises(GPUOracleError,match="commitment"): validate_atomic_result(path,"f",row)
