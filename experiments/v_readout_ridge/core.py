"""KV-Lingo Stage 1: linear (not affine), separate K/V, FP64 moments."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def load_model(path, device='cuda:0'):
    model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':device})
    model.eval().requires_grad_(False)
    return model

@torch.inference_mode()
def attention_mass(q, k, prefix_length, query_start, query_block=64, normalization='causal'):
    """[B,Hq,Tq,D], [B,Hkv,Tk,D] -> [B,prefix_length].

    Queries begin at physical query_start in the full causal teacher sequence.
    Average exactly over all continuation queries AND all query heads (GQA
    heads are repeated only for reading). No per-example mass renormalization.
    'prefix' implements literal prefix-only softmax in Eq 3.4; 'causal'
    retains the teacher's denominator including visible continuation keys.
    """
    if q.ndim!=4 or k.ndim!=4 or q.shape[1]%k.shape[1]: raise ValueError('bad GQA shapes')
    if not 0<prefix_length<=k.shape[2] or q.shape[2]<1: raise ValueError('empty attention domain')
    kr=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1).float()
    mass=torch.zeros((q.shape[0],prefix_length),device=q.device,dtype=torch.float64)
    for start in range(0,q.shape[2],query_block):
        stop=min(start+query_block,q.shape[2])
        end=prefix_length if normalization=='prefix' else min(k.shape[2],query_start+stop)
        scores=(q[:,:,start:stop].float()@kr[:,:,:end].transpose(-1,-2))*(q.shape[-1]**-0.5)
        if normalization=='causal':
            qi=torch.arange(query_start+start,query_start+stop,device=q.device)
            ki=torch.arange(end,device=q.device)
            scores.masked_fill_(ki[None,:]>qi[:,None],float('-inf'))
        elif normalization!='prefix': raise ValueError(normalization)
        p=torch.softmax(scores,dim=-1)[...,:prefix_length]
        mass+=p.double().sum(dim=(1,2))
    return mass/(q.shape[1]*q.shape[2])

@torch.inference_mode()
def solve_moments(gram, cross, rtol=1e-8):
    """Paper Appendix E solver. X-row convention: prediction = X @ W."""
    if gram.dtype!=torch.float64 or cross.dtype!=torch.float64: raise TypeError('FP64 moments required')
    if not torch.isfinite(gram).all() or not torch.isfinite(cross).all(): raise ValueError('nonfinite moments')
    gram=(gram+gram.mT)*0.5
    ev,u=torch.linalg.eigh(gram)
    cutoff=ev.max().clamp_min(0)*rtol
    keep=ev>cutoff
    if not keep.any(): raise ValueError('rank-zero regression')
    uk=u[:,keep]; wk=(uk/ev[keep])@(uk.mT@cross)
    if not torch.isfinite(wk).all(): raise ValueError('nonfinite solve')
    residual=torch.linalg.norm(gram@wk-cross)/torch.linalg.norm(cross).clamp_min(1e-30)
    return wk.float(), {'rank':int(keep.sum()),'max_eigenvalue':float(ev.max()),
        'min_eigenvalue':float(ev.min()),'relative_normal_residual':float(residual)}

class Trace:
    """Projection hooks capture actual BF16 pre-norm keys and values.

    Optional attention callback runs after q_norm/k_norm using saved forward
    tensors: no extra projections, no replacement of the model forward.
    """
    def __init__(self, model, prefix_length, query_start=None, normalization='causal', callback=None):
        self.model=model; self.n=prefix_length; self.qstart=query_start
        self.normalization=normalization; self.callback=callback; self.handles=[]
        self.k={}; self.v={}; self.q={}; self.kn={}; self.mass={}
    def __enter__(self):
        for i,layer in enumerate(self.model.model.layers):
            a=layer.self_attn
            def save(dst, idx, prefix_only=False):
                def hook(_m,_args,out):
                    dst[idx]=out[:,:self.n].detach() if prefix_only else out.detach()
                return hook
            self.handles.append(a.k_proj.register_forward_hook(save(self.k,i,True)))
            self.handles.append(a.v_proj.register_forward_hook(save(self.v,i,True)))
            if self.qstart is not None:
                self.handles.append(a.q_norm.register_forward_hook(save(self.q,i)))
                self.handles.append(a.k_norm.register_forward_hook(save(self.kn,i)))
            def finish(idx):
                def hook(mod,args,kwargs,out):
                    if self.qstart is not None:
                        pos=kwargs.get('position_embeddings') if kwargs else None
                        if pos is None and len(args)>1: pos=args[1]
                        if pos is None: raise RuntimeError('Qwen3 attention did not expose position_embeddings')
                        cos,sin=pos
                        q=self.q.pop(idx).transpose(1,2); k=self.kn.pop(idx).transpose(1,2)
                        q,k=apply_rotary_pos_emb(q,k,cos,sin)
                        self.mass[idx]=attention_mass(q[:,:,self.qstart:],k,self.n,self.qstart,
                            normalization=self.normalization)[0]
                    if self.callback:
                        self.callback(idx,self.k[idx][0],self.v[idx][0],self.mass.pop(idx,None))
                return hook
            self.handles.append(a.register_forward_hook(finish(i),with_kwargs=True))
        return self
    def __exit__(self,*exc):
        for h in self.handles: h.remove()

@torch.inference_mode()
def project_cache(raw_keys, raw_values, target, weights=None, position_ids=None, source_layer_map=None):
    """Map raw source projections then apply TARGET norm+RoPE exactly once."""
    cache=DynamicCache()
    for i,layer in enumerate(target.model.layers):
        src_i=i if source_layer_map is None else int(source_layer_map[i])
        k,v=raw_keys[src_i],raw_values[src_i]
        if weights is not None:
            # Capture hooks run on the source device; translation and target
            # norm must run where the target model and mapper live.
            dev=next(target.parameters()).device
            k=(k.float().to(dev)@weights[f'k.{i}']).to(target.dtype)
            v=(v.float().to(dev)@weights[f'v.{i}']).to(target.dtype)
        else:
            dev=next(target.parameters()).device
            k=k.to(dev);v=v.to(dev)
        B,T,_=k.shape; shape=(B,T,target.config.num_key_value_heads,target.config.head_dim)
        kn=layer.self_attn.k_norm(k.reshape(shape)).transpose(1,2)
        positions=(torch.arange(T,device=k.device)[None,:].expand(B,-1)
                   if position_ids is None else position_ids.to(k.device))
        cos,sin=target.model.rotary_emb(kn,positions)
        _,kr=apply_rotary_pos_emb(kn,kn,cos,sin)
        cache.update(kr,v.reshape(shape).transpose(1,2),i)
    return cache

@torch.inference_mode()
def mapped_cache(source,target,ids,weights):
    with Trace(source,ids.shape[1]) as tr:
        source.model(input_ids=ids,use_cache=False)
    return project_cache(tr.k,tr.v,target,weights)
