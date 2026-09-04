"""Concrete, fail-closed GPU runner for the A1 cache oracle.

This module deliberately has no CUDA import or model load at import time.  It
is a genuine runtime rather than a mock: after the gate it loads 32B on five
cards and 14B on three, builds both prefix caches from the same ``C[:-1]``,
replays ``C[-1]``, and records every one of the eleven frozen interventions.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl, hashlib, json, os, sys, tempfile, time, traceback
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from cache_oracle import (CacheContext, CacheProvenance, OracleContractError,
    OracleObservation, attention_readout_fp32, canonical_sha256,
    compose_legacy_cache, intervention_catalog, score_paired_observations,
    wrap_cache)
from oracle_diagnostics import (PreparationError, artifact_ref, read_json, read_jsonl,
    sha256_file, validate_mapper_manifest, validate_prepared, verify_pinned_file)
from runtime_harness import SamplePlan

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
BASELINE = REPO / "experiments/qwen3_32b_to_14b"
CONFIG = ROOT / "config_v2.json"

class GPUOracleError(RuntimeError): pass

def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()+b"\n"

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_canonical(value)); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally: Path(tmp).unlink(missing_ok=True)

def _ref(spec: Mapping[str, Any], label: str) -> Path:
    try: return verify_pinned_file({k: spec[k] for k in ("path","nbytes","sha256")}, label=label)
    except (PreparationError, KeyError) as e: raise GPUOracleError(str(e)) from e

def _inside(path: Path, root: Path) -> Path:
    resolved=path.resolve()
    try: resolved.relative_to(root.resolve())
    except ValueError as e: raise GPUOracleError(f"output escapes A1 runs: {resolved}") from e
    return resolved

def load_v2(path: str|Path=CONFIG) -> dict[str,Any]:
    value=read_json(path)
    if value.get("schema_version") != 2 or value.get("tier") != "method-dev": raise GPUOracleError("unsupported v2 config")
    return value

def complete_g0_status(cfg: Mapping[str,Any]) -> dict[str,Any]:
    """Return an explanatory non-authorized status rather than weakening G0."""
    gate=cfg["complete_g0_gate"]; p=Path(str(gate["path"]))
    if not p.is_file(): return {"passed":False,"reason":"complete G0 receipt is absent","path":str(p)}
    if p.stat().st_size != gate["nbytes"] or sha256_file(p) != gate["sha256"]:
        return {"passed":False,"reason":"complete G0 receipt identity mismatch","path":str(p)}
    try: r=read_json(p)
    except Exception as e: return {"passed":False,"reason":f"unreadable G0 receipt: {e}","path":str(p)}
    ok=r.get("status")==gate["required_status"] and r.get("scope",{}).get("closes_complete_g0") is gate["required_closes_complete_g0"]
    return {"passed":ok,"reason":"pass" if ok else "receipt is not a complete G0 closure","path":str(p),"sha256":sha256_file(p)}

def _select(rows: Sequence[Mapping[str,Any]], profile: str, cfg: Mapping[str,Any]) -> list[dict[str,Any]]:
    limits=cfg["profiles"].get(profile)
    if not isinstance(limits, Mapping): raise GPUOracleError("unknown profile")
    result=[]
    for dataset, need in limits.items():
        picked=[dict(r) for r in rows if r.get("dataset")==dataset][:int(need)]
        if len(picked)!=int(need): raise GPUOracleError(f"not enough {dataset} rows for {profile}")
        result.extend(picked)
    return result

def contract_preflight(cfg: Mapping[str,Any], *, profile: str) -> tuple[dict[str,Any], list[dict[str,Any]]]:
    legacy=read_json(cfg["legacy_v1_config"]["path"])
    if legacy.get("execution_policy",{}).get("gpu_execution_authorized_by_this_config") is not False: raise GPUOracleError("v1 GPU authorization drift")
    manifest=_ref(cfg["prepared_registry"],"prepared registry")
    records_path=_ref(cfg["prepared_records"],"prepared records")
    _ref(cfg["interventions"],"interventions")
    prepared=validate_prepared(manifest.parent)
    if prepared["manifest_sha256"] != cfg["prepared_registry"]["sha256"]: raise GPUOracleError("prepared manifest mismatch")
    mapper_cfg=legacy
    mapper=validate_mapper_manifest(mapper_cfg, deep_layer_hashes=False)
    if mapper["manifest"]["sha256"] != cfg["mapper"]["manifest_sha256"]: raise GPUOracleError("k20 mapper binding drift")
    rows=_select(read_jsonl(records_path),profile,cfg)
    for row in rows:
        SamplePlan.from_prepared_row(row) # validates token IDs/C[:-1] boundary
        if row.get("split") != "train_screen" or row.get("tier") != "method-dev": raise GPUOracleError("non-method-dev row")
    gate=complete_g0_status(cfg)
    result={"schema_version":2,"status":"passed_cpu_contract_preflight" if gate["passed"] else "blocked_missing_complete_g0",
       "gpu_used":False,"cuda_api_called":False,"profile":profile,"samples":len(rows),"interventions":11,
       "expected_atomic_results":len(rows),"gate":gate,"prepared":prepared,"mapper":mapper,
       "sample_bindings_sha256":canonical_sha256([r["cache_context"] for r in rows])}
    return result,rows

def _model_identity(root: Path, *, deep: bool) -> dict[str,Any]:
    index=root/"model.safetensors.index.json"
    if not index.is_file(): raise GPUOracleError(f"missing model index {index}")
    files={name:artifact_ref(root/name).to_dict() for name in ("config.json","tokenizer.json","tokenizer_config.json","generation_config.json","model.safetensors.index.json")}
    raw=read_json(index); shards=sorted(set(raw.get("weight_map",{}).values()))
    if not shards: raise GPUOracleError("model index contains no shards")
    entries=[]
    for name in shards:
        p=root/name
        if not p.is_file(): raise GPUOracleError(f"missing model shard {p}")
        entry={"path":name,"nbytes":p.stat().st_size,"sha256":sha256_file(p) if deep else None}
        entries.append(entry)
    return {"path":str(root),"metadata":files,"weight_shards":entries,"weights_deep_hashed":deep,
            "weight_inventory_sha256":canonical_sha256(entries)}

def _device(model: Any) -> torch.device: return next(model.get_input_embeddings().parameters()).device

def _sync() -> None:
    for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)

def _peaks() -> dict[str,Any]: return {str(i):{"allocated":int(torch.cuda.max_memory_allocated(i)),"reserved":int(torch.cuda.max_memory_reserved(i))} for i in range(torch.cuda.device_count())}

def _full_inputs(ids: Sequence[int], device: torch.device) -> dict[str,Any]:
    x=torch.tensor([list(ids)],dtype=torch.long,device=device); pos=torch.arange(x.shape[1],device=device)
    return {"input_ids":x,"attention_mask":torch.ones_like(x),"position_ids":pos[None],"cache_position":pos,"use_cache":True,"return_dict":True}

@contextmanager
def _capture_attn_inputs(model: Any) -> Iterator[list[torch.Tensor]]:
    base=getattr(model,"model",model); layers=base.layers
    captured:[torch.Tensor|None]=[None]*40; handles=[]
    for i,layer in enumerate(layers):
        def hook(_m,args,kwargs,slot=i):
            h=kwargs.get("hidden_states") if kwargs else args[0]
            if not isinstance(h,torch.Tensor) or captured[slot] is not None: raise GPUOracleError(f"bad native boundary capture layer {slot}")
            captured[slot]=h.detach()
        handles.append(layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True))
    try:
        yield captured # filled after forward
        if any(x is None for x in captured): raise GPUOracleError("native boundary Q/K/V capture incomplete")
    finally:
        for h in handles:h.remove()

def _native_boundary_qkv(model: Any, captures: Sequence[torch.Tensor|None], position: int) -> tuple[dict[int,torch.Tensor],dict[int,torch.Tensor],dict[int,torch.Tensor]]:
    from kvcache_transfer.rope import apply_rope,qwen3_rope_cos_sin_from_config
    base=getattr(model,"model",model); qs={};ks={};vs={}
    for i,h in enumerate(captures):
        if h is None: raise GPUOracleError("missing captured state")
        a=base.layers[i].self_attn; b,t,_=h.shape
        q=a.q_norm(a.q_proj(h).view(b,t,40,128)).transpose(1,2)
        k=a.k_norm(a.k_proj(h).view(b,t,8,128)).transpose(1,2)
        v=a.v_proj(h).view(b,t,8,128).transpose(1,2)
        p=torch.full((b,t),position,dtype=torch.long,device=h.device)
        cos,sin=qwen3_rope_cos_sin_from_config(p,model.config,device=h.device,dtype=torch.float32)
        qs[i]=apply_rope(q,cos,sin,output_dtype=q.dtype).detach()
        ks[i]=apply_rope(k,cos,sin,output_dtype=k.dtype).detach(); vs[i]=v.detach()
    return qs,ks,vs

def _assert_boundary_identity(full: Any, replay: Any, prefix_len: int) -> None:
    if not torch.allclose(full.logits[:,-1,:].float(),replay.logits[:,-1,:].float(),rtol=2e-3,atol=2e-3): raise GPUOracleError("C[:-1]+boundary logits do not match full C")
    from kvcache_transfer.cache_adapter import cache_to_legacy
    a,b=cache_to_legacy(full.past_key_values),cache_to_legacy(replay.past_key_values)
    for i,((ak,av),(bk,bv)) in enumerate(zip(a,b,strict=True)):
        if not (torch.allclose(ak[..., :prefix_len+1,:].float(),bk.float(),rtol=2e-3,atol=2e-3) and torch.allclose(av[..., :prefix_len+1,:].float(),bv.float(),rtol=2e-3,atol=2e-3)): raise GPUOracleError(f"full/replay cache mismatch layer {i}")

def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    result=set(); x=getattr(tokenizer,"eos_token_id",None)
    if x is not None: result.add(int(x)); y=getattr(getattr(model,"generation_config",None),"eos_token_id",None)
    if isinstance(y,int):result.add(y)
    elif y is not None:result.update(map(int,y))
    return result

def _decode_from_boundary(model: Any, tokenizer: Any, boundary_output: Any, cap: int) -> tuple[str,tuple[int,...],str]:
    from kvcache_transfer.cache_adapter import cache_sequence_length,full_attention_mask,replay_cache_position,replay_position_ids
    out=boundary_output; ids=[]; eos=_eos_ids(model,tokenizer); device=_device(model)
    for step in range(cap):
        token=int(out.logits[:,-1,:].argmax(-1).item())
        if token in eos: return tokenizer.decode(ids,skip_special_tokens=True),tuple(ids),"eos"
        ids.append(token)
        if step+1==cap: break
        n=cache_sequence_length(out.past_key_values); x=torch.tensor([[token]],device=device)
        mask=full_attention_mask(n,1,batch_size=1,device=device)
        out=model(input_ids=x,past_key_values=out.past_key_values,attention_mask=mask,position_ids=replay_position_ids(mask,1),cache_position=replay_cache_position(n,1,device=device),use_cache=True,return_dict=True)
    return tokenizer.decode(ids,skip_special_tokens=True),tuple(ids),"cap"

def _attention_for(prefix_layers: Any, qs: Mapping[int,torch.Tensor], bks: Mapping[int,torch.Tensor], bvs: Mapping[int,torch.Tensor], length:int) -> dict[int,Any]:
    result={}
    for i,(k,v) in enumerate(prefix_layers):
        key=torch.cat((k,bks[i]),dim=-2); val=torch.cat((v,bvs[i]),dim=-2)
        result[i]=attention_readout_fp32(qs[i],key,val,query_absolute_positions=[length])
    return result

def _observation(model: Any,tokenizer: Any,context: CacheContext,composed: Any,qs: Mapping[int,torch.Tensor],bks:Mapping[int,torch.Tensor],bvs:Mapping[int,torch.Tensor]) -> tuple[OracleObservation,dict[str,Any]]:
    from kvcache_transfer.cache_adapter import boundary_replay_inputs,build_dynamic_cache
    device=_device(model); boundary=torch.tensor([[context.boundary_token_id]],dtype=torch.long,device=device)
    cache=build_dynamic_cache(composed,model=model,clone=True,detach=True)
    inputs=boundary_replay_inputs(boundary,cache,clone_cache=False)
    out=model(**inputs,return_dict=True); text,tokens,reason=_decode_from_boundary(model,tokenizer,out,256)
    trace={"generated_token_ids":[int(x) for x in tokens],"generated_token_ids_sha256":canonical_sha256([int(x) for x in tokens]),"generated_token_count":len(tokens),"stop_reason":reason,"cap":256,"decoded_text_sha256":hashlib.sha256(text.encode()).hexdigest()}
    return OracleObservation(first_token_logits=out.logits[:,-1,:].detach(),generated_text=text,generated_token_count=len(tokens),stop_reason=reason,attention=_attention_for(composed,qs,bks,bvs,context.prefix_length)),trace

def _wait_idle(cfg:Mapping[str,Any])->Mapping[str,Any]:
    sys.path.insert(0,str(BASELINE))
    from reverse_kv.reverse_latency import wait_for_idle_gpus
    return wait_for_idle_gpus({"runtime":{"require_cuda_visible_devices":cfg["runtime"]["required_cuda_visible_devices"],"idle_memory_limit_mib":cfg["runtime"]["idle_memory_limit_mib"],"idle_consecutive_samples":cfg["runtime"]["idle_consecutive_samples"],"idle_sample_interval_seconds":cfg["runtime"]["idle_sample_interval_seconds"],"idle_timeout_seconds":cfg["runtime"]["idle_timeout_seconds"]}})

def _load_components(cfg:Mapping[str,Any])->tuple[Any,Any,Any,Any]:
    for p in (REPO,REPO/"src",BASELINE):
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    from transformers import AutoTokenizer
    from kvcache_transfer.modeling import load_qwen3
    from reverse_kv.runtime import ReverseMappedPrefixCacheBuilder
    m=cfg["models"]; tok=AutoTokenizer.from_pretrained(m["target_14b"],local_files_only=True,use_fast=True)
    target=load_qwen3(m["target_14b"],devices=m["target_devices"],layer_counts=m["target_layer_counts"],role="source",attention_implementation=m["attention_implementation"])
    source=load_qwen3(m["source_32b"],devices=m["source_devices"],layer_counts=m["source_layer_counts"],role="target",attention_implementation=m["attention_implementation"])
    builder=ReverseMappedPrefixCacheBuilder(source_model=source,target_model=target,mapper_manifest=cfg["mapper"]["manifest"],token_chunk=m["mapper_token_chunk"])
    return tok,source,target,builder

def _result_path(out:Path,row:Mapping[str,Any])->Path:
    h=hashlib.sha256((row["dataset"]+"\0"+row["sample_id"]).encode()).hexdigest()[:24]
    return out/"results"/row["dataset"]/f"{h}.json"

def validate_atomic_result(path:Path,fingerprint:str,row:Mapping[str,Any])->dict[str,Any]:
    if not path.is_file(): raise GPUOracleError("checkpoint does not exist")
    try:v=read_json(path)
    except Exception as e: raise GPUOracleError(f"checkpoint JSON is malformed: {e}") from e
    if not (v.get("status")=="complete" and v.get("request_fingerprint")==fingerprint and v.get("sample_id")==row["sample_id"] and v.get("dataset")==row["dataset"] and v.get("prompt_token_ids_sha256")==row["prompt_token_ids_sha256"]): raise GPUOracleError("checkpoint identity changed")
    context=row.get("cache_context")
    if v.get("cache_context") != context: raise GPUOracleError("checkpoint cache context changed")
    expected=[x.id for x in intervention_catalog()]
    if [x.get("intervention_id") for x in v.get("paired",[])] != expected: raise GPUOracleError("checkpoint intervention order changed")
    traces=v.get("intervention_generation")
    if not isinstance(traces,list) or [x.get("intervention_id") for x in traces] != expected: raise GPUOracleError("checkpoint generation trace/order changed")
    for trace in traces:
        ids=trace.get("generated_token_ids")
        if not isinstance(ids,list) or any(type(x) is not int or x<0 for x in ids): raise GPUOracleError("checkpoint token IDs are malformed")
        if trace.get("generated_token_ids_sha256") != canonical_sha256(ids): raise GPUOracleError("checkpoint token-ID commitment changed")
        if trace.get("generated_token_count") != len(ids) or trace.get("cap") != 256 or trace.get("stop_reason") not in {"eos","cap"}: raise GPUOracleError("checkpoint generation cap/stop changed")
        if trace["stop_reason"]=="cap" and len(ids)!=256: raise GPUOracleError("checkpoint cap reason disagrees with length")
    return v

def _validate_checkpoint(path:Path,fingerprint:str,row:Mapping[str,Any])->bool:
    try: validate_atomic_result(path,fingerprint,row); return True
    except GPUOracleError: return False

def execute(config_path:str|Path,*,profile:str,output_dir:str|Path)->int:
    cfg=load_v2(config_path); preflight,rows=contract_preflight(cfg,profile=profile)
    if not preflight["gate"]["passed"]: raise GPUOracleError("GPU execution refused: "+preflight["gate"]["reason"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["runtime"]["required_cuda_visible_devices"]: raise GPUOracleError("CUDA_VISIBLE_DEVICES must expose exactly 0..7")
    out=_inside(Path(output_dir),Path(cfg["outputs_root"])); out.mkdir(parents=True,exist_ok=True)
    definition={"schema_version":2,"config":artifact_ref(config_path).to_dict(),"preflight":preflight,"profile":profile,"rows":[{"dataset":r["dataset"],"sample_id":r["sample_id"],"prompt_token_ids_sha256":r["prompt_token_ids_sha256"]} for r in rows]}
    definition["request_fingerprint"]=canonical_sha256(definition); fingerprint=definition["request_fingerprint"]
    lock_path=Path(cfg["runtime"]["global_gpu_lock"]); lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("a+b") as lock:
      fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
      _atomic_json(out/"run_definition.json",definition); _atomic_json(out/"status.json",{"status":"waiting_for_idle_gpus","request_fingerprint":fingerprint})
      idle=_wait_idle(cfg); _atomic_json(out/"gpu_idle_gate.json",idle)
      if not torch.cuda.is_available() or torch.cuda.device_count()!=8: raise GPUOracleError("exactly 8 CUDA devices required")
      _atomic_json(out/"model_identity.json",{"source":_model_identity(Path(cfg["models"]["source_32b"]),deep=True),"target":_model_identity(Path(cfg["models"]["target_14b"]),deep=True)})
      tok,source,target,builder=_load_components(cfg); _atomic_json(out/"runtime_provenance.json",{"mapper":builder.provenance,"source_sharding":getattr(source,"_kvcache_transfer_layer_sharding_evidence"),"target_sharding":getattr(target,"_kvcache_transfer_layer_sharding_evidence")})
      _atomic_json(out/"status.json",{"status":"running","request_fingerprint":fingerprint})
      for n,row in enumerate(rows,1):
        dest=_result_path(out,row)
        if _validate_checkpoint(dest,fingerprint,row):continue
        try:
          plan=SamplePlan.from_prepared_row(row); ctx=plan.context; prefix=list(plan.prompt_token_ids[:-1]); base=getattr(target,"model",target); dev=_device(target)
          full=base(**_full_inputs(plan.prompt_token_ids,dev)); native_prefix=base(**_full_inputs(prefix,dev))
          from kvcache_transfer.cache_adapter import cache_to_legacy,boundary_replay_inputs
          native_env=wrap_cache(cache_to_legacy(native_prefix.past_key_values),context=ctx,provenance=CacheProvenance("native","Qwen3-14B exact C[:-1]",sha256_file(Path(cfg["models"]["target_14b"])/"config.json")))
          # Capture native Q/K/V on its actual boundary replay, then prove it equals full C.
          b=torch.tensor([[ctx.boundary_token_id]],device=dev)
          with _capture_attn_inputs(target) as caps: native_replay=target(**boundary_replay_inputs(b,native_prefix.past_key_values),return_dict=True)
          _assert_boundary_identity(full,native_replay,ctx.prefix_length); qs,bks,bvs=_native_boundary_qkv(target,caps,ctx.prefix_length)
          for i,(k,v) in enumerate(cache_to_legacy(native_replay.past_key_values)):
            if not (torch.allclose(k[...,-1:,:].float(),bks[i].float(),rtol=2e-3,atol=2e-3) and torch.allclose(v[...,-1:,:].float(),bvs[i].float(),rtol=2e-3,atol=2e-3)): raise GPUOracleError(f"native post-RoPE K/V identity failed layer {i}")
          mapped_cache=builder(prefix); mapped_env=wrap_cache(cache_to_legacy(mapped_cache),context=ctx,provenance=CacheProvenance("mapped","Qwen3-32B C[:-1] then frozen k20",sha256_file(Path(cfg["models"]["source_32b"])/"config.json"),cfg["mapper"]["manifest_sha256"]))
          reference=None; paired=[]; compositions=[]; traces=[]
          for spec in intervention_catalog():
            composed,receipt=compose_legacy_cache(native_env,mapped_env,spec,clone=True); obs,trace=_observation(target,tok,ctx,composed,qs,bks,bvs); compositions.append(receipt.to_dict()); traces.append({"intervention_id":spec.id,**trace})
            if spec.id=="native_k_native_v":reference=obs
            if reference is None:raise GPUOracleError("native reference not first")
            paired.append(score_paired_observations(context=ctx,intervention=spec,references=plan.references,gold_first_token_ids=plan.gold_first_token_ids,native=reference,candidate=obs))
          _sync(); payload={"schema_version":2,"status":"complete","request_fingerprint":fingerprint,"dataset":row["dataset"],"sample_id":row["sample_id"],"prompt_token_ids_sha256":row["prompt_token_ids_sha256"],"prefix_token_ids_sha256":row["prefix_token_ids_sha256"],"cache_context":asdict(ctx),"boundary_identity":"full C == prefill C[:-1] + replay C[-1] logits/cache","native_qkv_identity":"q_norm/k_norm + target RoPE; captured boundary K/V agrees with actual cache","generation":{"cap":256,"stop":"eos_only","exact_token_ids":True},"paired":paired,"intervention_generation":traces,"compositions":compositions,"systems":{"gpu_peak":_peaks()}}
          _atomic_json(dest,payload); validate_atomic_result(dest,fingerprint,row); dest.with_suffix(".failure.json").unlink(missing_ok=True)
        except BaseException as e:
          _atomic_json(dest.with_suffix(".failure.json"),{"status":"failed","request_fingerprint":fingerprint,"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()}); raise
        finally:
          torch.cuda.empty_cache(); _atomic_json(out/"status.json",{"status":"running","request_fingerprint":fingerprint,"completed_prompts":n,"total_prompts":len(rows)})
      inventory=[artifact_ref(_result_path(out,r)).to_dict() for r in rows]; receipt={"schema_version":2,"status":"complete","request_fingerprint":fingerprint,"atomic_prompt_results":len(inventory),"result_inventory":inventory,"result_inventory_sha256":canonical_sha256(inventory),"method_dev_only":True,"dev_or_locked_opened":False}
      _atomic_json(out/"completion_receipt.json",receipt);_atomic_json(out/"status.json",{"status":"complete","request_fingerprint":fingerprint,"completed_prompts":len(rows)}); return 0

def validate_run(config_path: str|Path, *, profile: str, output_dir: str|Path) -> dict[str,Any]:
    """Recheck every atomic prompt, all hashes, and the immutable run receipt.

    This function is CPU-only; it never loads a model or touches a CUDA API.
    It intentionally does not require G0 to remain present so a finished run
    remains auditable after external gate material has been archived.
    """
    cfg=load_v2(config_path); out=_inside(Path(output_dir),Path(cfg["outputs_root"]))
    definition_path=out/"run_definition.json"; receipt_path=out/"completion_receipt.json"
    definition=read_json(definition_path); receipt=read_json(receipt_path)
    cfg_ref=definition.get("config")
    if not isinstance(cfg_ref,Mapping) or _ref(cfg_ref,"run config") != Path(config_path).resolve(): raise GPUOracleError("run definition config binding changed")
    fingerprint=definition.get("request_fingerprint")
    unsigned=dict(definition); unsigned.pop("request_fingerprint",None)
    if not isinstance(fingerprint,str) or canonical_sha256(unsigned)!=fingerprint: raise GPUOracleError("run definition fingerprint changed")
    _preflight,rows=contract_preflight(cfg,profile=profile)
    inventory=[]
    for row in rows:
        path=_result_path(out,row); validate_atomic_result(path,fingerprint,row); inventory.append(artifact_ref(path).to_dict())
    if receipt.get("status")!="complete" or receipt.get("request_fingerprint")!=fingerprint: raise GPUOracleError("completion receipt identity changed")
    if receipt.get("atomic_prompt_results")!=len(rows) or receipt.get("result_inventory")!=inventory or receipt.get("result_inventory_sha256")!=canonical_sha256(inventory): raise GPUOracleError("completion receipt inventory changed")
    if receipt.get("method_dev_only") is not True or receipt.get("dev_or_locked_opened") is not False: raise GPUOracleError("receipt claims invalid data access")
    return {"schema_version":2,"status":"passed_cpu_receipt_validation","gpu_used":False,"profile":profile,"prompts":len(rows),"request_fingerprint":fingerprint,"completion_receipt_sha256":sha256_file(receipt_path)}

__all__=["GPUOracleError","complete_g0_status","contract_preflight","execute","load_v2","validate_atomic_result","validate_run"]
