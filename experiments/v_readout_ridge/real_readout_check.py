"""Exercise actual BF16 long-context hooks, GQA readouts and weighted attention."""
import json,time
from pathlib import Path
import torch
from core import load_model,attention_mass,atomic_json,digest
from readout import ReadoutTrace,readout_moments
from fit import DATA,rows

@torch.inference_mode()
def main():
    h=Path(__file__).resolve().parent
    sample=max(rows(DATA),key=lambda r:len(r['prefix_ids']))
    x,y=sample['prefix_ids'],sample['continuation_ids'];n=len(x)
    model=load_model('/home/tiger/models/Qwen3-1.7B','cuda:0')
    diagnostics=[];start=time.monotonic()
    def consume(i,k,v,q,kn):
        g,c,yy,w,info=readout_moments(q,kn,v,v)
        independent=attention_mass(q[None],kn[None],n,n,normalization='prefix')[0]
        torch.testing.assert_close(w,independent,rtol=1e-10,atol=1e-12)
        for head in range(8):
            target_cols=slice(head*128,(head+1)*128)
            torch.testing.assert_close(c[head],g[head,:,target_cols],atol=1e-9,rtol=1e-8)
            torch.testing.assert_close(yy[head],g[head,target_cols,target_cols].trace(),atol=1e-8,rtol=1e-8)
        diagnostics.append(dict(layer=i,**info,identity_readout_moments_match=True,independent_attention_mass_match=True))
    with ReadoutTrace(model,n,consume):model.model(input_ids=torch.tensor([x+y],device='cuda:0'),use_cache=False)
    assert len(diagnostics)==28
    atomic_json(h/'checks/real_readout_check.json',dict(passed=True,train_row=sample['id'],prefix=n,
        continuation=len(y),layers=diagnostics,seconds=time.monotonic()-start,
        code_sha256={p.name:digest(p) for p in [h/'readout.py',h/'core.py',Path(__file__)]}))
    print('LONG CONTEXT READOUT CHECK PASSED',flush=True)

if __name__=='__main__':main()
