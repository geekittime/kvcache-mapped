"""Isolated repeated-process numerical diagnostic; never enters benchmark scores."""
import argparse,os,json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from core import load_model,atomic_json,digest
from evaluate import dataset,user_prompt,chat_ids,load_weights,generate_batch
from fit import paths

HERE=Path(__file__).resolve().parent

@torch.inference_mode()
def main(a):
    if a.deterministic:
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    sp,tp=paths('17_06')
    tgt=load_model(tp,a.device);src=load_model(sp,a.device)
    tok=AutoTokenizer.from_pretrained(tp,local_files_only=True)
    artifact=HERE/'artifacts/17_06/full_v1'
    w,mapping=load_weights(artifact,'weighted',a.device,sp,tp)
    _,data=dataset('arc')
    sequences=[chat_ids(tok,user_prompt(r,'arc'),False) for r in data[:32]]
    steps=[]
    def record(_m,_args,out):
        scores,ids=out.logits[:,-1].float().topk(2,dim=-1)
        steps.append(dict(top2_ids=ids.cpu().tolist(),top2_logits=scores.cpu().tolist()))
    handle=tgt.register_forward_hook(record)
    generated=generate_batch(src,tgt,sequences,w,a.max_new,tok.pad_token_id,mapping)
    handle.remove()
    atomic_json(a.output,dict(diagnostic_only=True,seed=a.seed,deterministic=a.deterministic,
        max_new=a.max_new,model_run_sha256=digest(artifact/'run.json'),device=a.device,
        generated=generated,steps=steps))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:2');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-new',type=int,default=128);p.add_argument('--deterministic',action='store_true')
    main(p.parse_args())
