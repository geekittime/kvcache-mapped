"""Exact grouped-query readout sufficient statistics; no attention averaging before AX."""
import torch
from core import Trace, apply_rotary_pos_emb


@torch.inference_mode()
def ridge(gram, cross, lam=0.01):
    assert gram.dtype == cross.dtype == torch.float64 and lam > 0
    assert torch.isfinite(gram).all() and torch.isfinite(cross).all()
    g = (gram + gram.mT) * .5
    regularized = g + lam * torch.eye(g.shape[-1], dtype=g.dtype, device=g.device)
    chol = torch.linalg.cholesky(regularized)
    w = torch.cholesky_solve(cross, chol)
    residual = (regularized @ w - cross).norm() / cross.norm().clamp_min(1e-30)
    assert torch.isfinite(w).all() and residual < 1e-7, float(residual)
    return w.float(), dict(lambda_=lam, normal_residual=float(residual),
                          weight_norm=float(w.norm()), finite=True)


@torch.inference_mode()
def readout_moments(q, k, source_v, target_v, query_block=64):
    """q[Hq,m,d], prefix k[Hkv,n,d], X[n,p], Y[n,Hkv*d].

    Softmax is FP32 like the native teacher. All AX/AY, products and sums are
    FP64. Return head-specific Grams and cross moments, plus the paper's
    all-query/head-averaged token mass. Query-head rows are never averaged.
    """
    hq, m, dh = q.shape; hk, n, kd = k.shape; p = source_v.shape[-1]
    assert m > 0 and n > 0 and dh == kd and hq % hk == 0
    assert source_v.shape[0] == n and target_v.shape == (n, hk * dh)
    group = hq // hk
    kr = k.repeat_interleave(group, dim=0).float()
    x = source_v.double(); y = target_v.double().reshape(n, hk, dh).permute(1, 0, 2)
    zs=[]; ts=[]
    mass = torch.zeros(n, device=q.device, dtype=torch.float64)
    max_mass_error = 0.
    for start in range(0, m, query_block):
        a = torch.softmax((q[:, start:start+query_block].float() @ kr.mT) * dh**-.5, dim=-1)
        max_mass_error = max(max_mass_error, float((a.sum(-1)-1).abs().max()))
        a = a.double(); mass += a.sum((0, 1))
        # Flatten group and query dimensions only within the same KV head.
        ag = a.reshape(hk, -1, n)
        zs.append(ag @ x)
        ts.append(ag @ y)
    z=torch.cat(zs, dim=1); t=torch.cat(ts, dim=1)
    assert z.shape == (hk, group*m, p)
    normalizer = group*m
    gram=z.mT @ z / normalizer
    cross=z.mT @ t / normalizer
    yy=t.square().sum((1,2)) / normalizer
    assert max_mass_error < 2e-6 and torch.isfinite(gram).all()
    return gram,cross,yy,mass/(hq*m),dict(queries=m,query_heads=hq,kv_heads=hk,
                prefix=n,max_attention_row_mass_error=max_mass_error)


class ReadoutTrace(Trace):
    """Capture native teacher q/k after norm+RoPE and consume one layer at a time."""
    def __init__(self, model, prefix_length, consume):
        super().__init__(model, prefix_length)
        self.consume=consume

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            a=layer.self_attn
            def save(dst, idx, prefix_only=False):
                def hook(_m,_args,out):
                    dst[idx]=out[:,:self.n].detach() if prefix_only else out.detach()
                return hook
            self.handles.append(a.k_proj.register_forward_hook(save(self.k,i,True)))
            self.handles.append(a.v_proj.register_forward_hook(save(self.v,i,True)))
            self.handles.append(a.q_norm.register_forward_hook(save(self.q,i)))
            self.handles.append(a.k_norm.register_forward_hook(save(self.kn,i)))
            def finish(idx):
                def hook(mod,args,kwargs,out):
                    pos=kwargs.get('position_embeddings')
                    if pos is None and len(args)>1:pos=args[1]
                    assert pos is not None
                    q,k=apply_rotary_pos_emb(self.q.pop(idx).transpose(1,2),
                                             self.kn.pop(idx).transpose(1,2),*pos)
                    self.consume(idx,self.k.pop(idx)[0],self.v.pop(idx)[0],
                                 q[0,:,self.n:],k[0,:,:self.n])
                return hook
            self.handles.append(a.register_forward_hook(finish(i),with_kwargs=True))
        return self
