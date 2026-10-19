"""Paired held-out weighted/readout errors, and native/mapped continuation KL/CE."""
import argparse,json,math,time
from pathlib import Path
import torch
import torch.nn.functional as F
from core import load_model,Trace,project_cache,atomic_json,digest
from readout import ReadoutTrace,readout_moments
from fit import paths,rows,quadratic
from evaluate import load_weights

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'data/kv_lingo/training_paper_v7/validation.jsonl'

@torch.inference_mode()
def decoder(model,cache,prefix,continuation):
    # Same split boundary for native and mapped routes. Predict y[0] from x[-1].
    seq=prefix[-1:]+continuation[:-1];n=len(prefix);m=len(continuation);dev=next(model.parameters()).device
    result=[]
    for start in range(0,m,64):
        piece=seq[start:start+64];offset=n-1+start
        out=model(input_ids=torch.tensor([piece],device=dev),past_key_values=cache,
            attention_mask=torch.ones((1,offset+len(piece)),device=dev,dtype=torch.long),
            position_ids=torch.arange(offset,offset+len(piece),device=dev)[None],
            cache_position=torch.arange(offset,offset+len(piece),device=dev),use_cache=True)
        cache=out.past_key_values
        result.append(F.log_softmax(out.logits[0].float(),dim=-1).cpu())
    return torch.cat(result)

@torch.inference_mode()
def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);assert not (out/'predictions.jsonl').exists()
    artifact=Path(a.artifact);sp,tp=paths(a.pair);selected=rows(DATA)
    if a.limit:selected=selected[:a.limit]
    protocol=dict(pair=a.pair,source=sp,target=tp,data=str(DATA),data_sha256=digest(DATA),limit=a.limit,
        artifact=str(artifact),artifact_sha256=digest(artifact/'run.json'),
        code_sha256={p.name:digest(p) for p in [Path(__file__),HERE/'core.py',HERE/'readout.py',HERE/'evaluate.py']},
        decoder='native split prefix[:-1] vs mapped prefix[:-1]; teacher forcing from last prefix token; FP32 log probs',
        intrinsic='full fit-prefix native teacher prefix-only attention, head-wise query rows; no ridge penalty on validation')
    atomic_json(out/'protocol.json',protocol)
    src=load_model(sp,a.device);tgt=load_model(tp,a.device)
    weights={mode:load_weights(artifact,mode,a.device,sp,tp)[0] for mode in ['weighted','readout']}
    done=[];start=time.monotonic()
    with (out/'predictions.jsonl').open('x',buffering=1) as f:
        for row in selected:
            x,y=row['prefix_ids'],row['continuation_ids'];n=len(x)
            with Trace(src,n) as st:src.model(input_ids=torch.tensor([x],device=a.device),use_cache=False)
            losses={mode:dict(weighted_v=0.,readout_v=0.) for mode in weights};energy=dict(weighted_v=0.,readout_v=0.)
            def consume(i,k,v,q,kn):
                X=st.v[i][0].double();Y=v.double();g,c,yy,mass,_=readout_moments(q,kn,X,Y)
                for mode,w in weights.items():
                    wv=w[f'v.{i}'].double();err=X@wv-Y
                    losses[mode]['weighted_v']+=float((err.square()*mass[:,None]).sum())
                    rh=wv.reshape(1024,8,128).permute(1,0,2)
                    losses[mode]['readout_v']+=quadratic(rh,g,c,yy.sum())
                energy['weighted_v']+=float((Y.square()*mass[:,None]).sum());energy['readout_v']+=float(yy.sum())
            with ReadoutTrace(tgt,n,consume):tgt.model(input_ids=torch.tensor([x+y],device=a.device),use_cache=False)
            del st
            native=tgt.model(input_ids=torch.tensor([x[:-1]],device=a.device),use_cache=True).past_key_values
            logp=decoder(tgt,native,x,y);del native
            native_nll=-float(logp[torch.arange(len(y)),torch.tensor(y)].sum())
            with Trace(src,n-1) as st:src.model(input_ids=torch.tensor([x[:-1]],device=a.device),use_cache=False)
            values={}
            for mode,w in weights.items():
                cache=project_cache(st.k,st.v,tgt,w);logq=decoder(tgt,cache,x,y);del cache
                kl=float((logp.exp()*(logp-logq)).sum())
                nll=-float(logq[torch.arange(len(y)),torch.tensor(y)].sum())
                assert kl>=-1e-3 and math.isfinite(kl+nll)
                values[mode]=dict(**losses[mode],kl_sum=kl,native_nll_sum=native_nll,mapped_nll_sum=nll)
                del logq
            del logp,st
            record=dict(id=row['id'],uuid=row['uuid'],prefix=n,continuation=len(y),values=values,target_energy=energy)
            f.write(json.dumps(record,allow_nan=False)+chr(10));done.append(record)
            print(json.dumps(dict(done=len(done),total=len(selected),seconds=time.monotonic()-start)),flush=True)
    summary=summarize(done)
    atomic_json(out/'results.json',dict(complete=True,n=len(done),protocol=protocol,summary=summary,
                                      predictions_sha256=digest(out/'predictions.jsonl')))
    audit(out)

def summarize(done):
    nt=sum(x['continuation'] for x in done);summary={}
    for mode in ['weighted','readout']:
        vals=[x['values'][mode] for x in done]
        item={key:sum(v[key] for v in vals)/len(vals) for key in ['weighted_v','readout_v']}
        for key in ['weighted_v','readout_v']:
            item[key+'_relative']=sum(v[key] for v in vals)/sum(x['target_energy'][key] for x in done)
        item['kl_sample_mean']=sum(x['values'][mode]['kl_sum']/x['continuation'] for x in done)/len(done)
        for key,target in [('kl_sum','kl_token_mean'),('native_nll_sum','native_ce'),('mapped_nll_sum','mapped_ce')]:
            item[target]=sum(v[key] for v in vals)/nt
        item['mapped_ppl']=math.exp(item['mapped_ce']);item['native_ppl']=math.exp(item['native_ce']);summary[mode]=item
    return summary

def audit(out):
    out=Path(out);r=json.loads((out/'results.json').read_text());p=r['protocol'];done=rows(out/'predictions.jsonl')
    assert r['complete'] and digest(DATA)==p['data_sha256'] and digest(Path(p['artifact'])/'run.json')==p['artifact_sha256']
    for name,value in p['code_sha256'].items():assert digest(HERE/name)==value
    expected=rows(DATA);expected=expected[:p['limit']] if p['limit'] else expected
    assert len(done)==len(expected)==r['n'] and {x['id'] for x in done}=={x['id'] for x in expected}
    refs={x['id']:x for x in expected}
    for x in done:
        ref=refs[x['id']];assert x['uuid']==ref['uuid'] and x['continuation']==len(ref['continuation_ids']) and x['prefix']==len(ref['prefix_ids'])
        for v in x['values'].values():assert all(math.isfinite(z) and z>=-1e-3 for z in v.values())
    assert summarize(done)==r['summary'] and digest(out/'predictions.jsonl')==r['predictions_sha256']
    atomic_json(out/'audit.json',dict(passed=True,n=len(done),full_validation=not p['limit'],results_sha256=digest(out/'results.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['06_17','17_06'],required=True)
    p.add_argument('--artifact',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--audit-only',action='store_true');a=p.parse_args();audit(a.output) if a.audit_only else main(a)
