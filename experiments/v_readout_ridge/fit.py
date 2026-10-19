"""Paired fits share native activations, attention and exactly the same K mapper."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from core import load_model,Trace,atomic_json,digest
from readout import ReadoutTrace,readout_moments,ridge

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'data/kv_lingo/training_paper_v7/train.jsonl'
PAIRS={'06_17':('0.6B','1.7B'),'17_06':('1.7B','0.6B')}

def paths(pair):return tuple('/home/tiger/models/Qwen3-'+x for x in PAIRS[pair])

def rows(path):return [json.loads(x) for x in Path(path).open() if x.strip()]

def code_hashes():return {p.name:digest(p) for p in [HERE/'fit.py',HERE/'readout.py',HERE/'core.py']}

def metadata(a):
    sp,tp=paths(a.pair)
    return dict(pair=a.pair,source=sp,target=tp,source_layer_map=list(range(28)),
                data=str(DATA),data_sha256=digest(DATA),shards=a.shards,limit=a.limit,
                model_config_sha256={p:digest(Path(p)/'config.json') for p in [sp,tp]},
                code_sha256=code_hashes(),attention_normalization='prefix',bias=False,
                ridge_lambda=.01,architecture='split_KV_head_mixing_identity_layers_pre_norm',
                readout_normalization='sum KV heads, mean continuation queries and GQA queries per KV head',
                weighted_normalization='mean attention over continuation queries and all query heads; sum prefix tokens',
                numerical_precision='BF16 teacher, FP32 attention softmax, FP64 moments and solve, FP32 maps')

def save(path,stats,done,meta,diagnostics):
    tmp=path.with_suffix('.tmp')
    torch.save(dict(stats={k:v.cpu() for k,v in stats.items()},done=done,metadata=meta,diagnostics=diagnostics),tmp)
    os.replace(tmp,path)

@torch.inference_mode()
def fit(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    path=out/f'moments_{a.shard:02d}.pt';assert not path.exists(),'fresh run only'
    meta=metadata(a);meta['shard']=a.shard
    selected=rows(DATA);selected=selected[:a.limit] if a.limit else selected
    selected=[r for i,r in enumerate(selected) if i%a.shards==a.shard]
    sp,tp=paths(a.pair);src=load_model(sp,a.device);tgt=load_model(tp,a.device)
    L=tgt.config.num_hidden_layers;H=tgt.config.num_key_value_heads;d=tgt.config.head_dim;D=H*d
    assert L==28 and H==8 and d==128 and src.config.num_hidden_layers==L
    shapes={'kg':(L,D,D),'kc':(L,D,D),'vg':(L,D,D),'vc':(L,D,D),
            'rg':(L,H,D,D),'rc':(L,H,D,d),'vyy':(L,), 'ryy':(L,H)}
    stats={key:torch.zeros(shape,device=a.device,dtype=torch.float64) for key,shape in shapes.items()}
    start=time.monotonic();done=[];diags=[]
    for row in selected:
        x,y=row['prefix_ids'],row['continuation_ids'];n=len(x);layer_diags=[]
        with Trace(src,n) as st:src.model(input_ids=torch.tensor([x],device=a.device),use_cache=False)
        def consume(i,k,v,q,kn):
            Xv=st.v[i][0].double();Yv=v.double()
            g,c,yy,w,info=readout_moments(q,kn,Xv,Yv)
            stats['rg'][i]+=g;stats['rc'][i]+=c;stats['ryy'][i]+=yy
            for s,X,Y in [('k',st.k[i][0].double(),k.double()),('v',Xv,Yv)]:
                weighted=X*w[:,None]
                stats[s+'g'][i].addmm_(X.mT,weighted)
                stats[s+'c'][i].addmm_(weighted.mT,Y)
            stats['vyy'][i]+=(Yv.square()*w[:,None]).sum()
            layer_diags.append(info)
        with ReadoutTrace(tgt,n,consume):tgt.model(input_ids=torch.tensor([x+y],device=a.device),use_cache=False)
        assert len(layer_diags)==L
        done.append(row['id']);diags.append(dict(id=row['id'],prefix=n,continuation=len(y),
            max_attention_row_mass_error=max(v['max_attention_row_mass_error'] for v in layer_diags)))
        print(json.dumps(dict(pair=a.pair,shard=a.shard,done=len(done),total=len(selected),id=row['id'],
                              prefix=n,continuation=len(y),seconds=time.monotonic()-start)),flush=True)
        del st
        if len(done)%a.save_every==0:save(path,stats,done,meta,diags)
    save(path,stats,done,meta,diags)
    atomic_json(out/f'shard_{a.shard:02d}.json',dict(complete=True,metadata=meta,done=done,
                seconds=time.monotonic()-start,moments_sha256=digest(path),peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30))

def quadratic(w,g,c,yy):
    w=w.double();return float((w*(g@w)).sum()-2*(w*c).sum()+yy)

@torch.inference_mode()
def merge(a):
    out=Path(a.output);assert not (out/'run.json').exists()
    stats=None;done=[];expected=metadata(a);shard_reports=[]
    for shard in range(a.shards):
        p=out/f'moments_{shard:02d}.pt';proof=json.loads((out/f'shard_{shard:02d}.json').read_text())
        assert proof['complete'] and digest(p)==proof['moments_sha256']
        item=torch.load(p,map_location='cpu',weights_only=True);m=item['metadata'].copy();assert m.pop('shard')==shard
        assert m==expected and proof['done']==item['done']
        assert not set(done)&set(item['done']);done+=item['done']
        assert all(v.dtype==torch.float64 and torch.isfinite(v).all() for v in item['stats'].values())
        if stats is None:stats=item['stats']
        else:
            for key in stats:stats[key]+=item['stats'][key]
        shard_reports.append(dict(shard=shard,n=len(item['done']),seconds=proof['seconds']))
    selected=rows(DATA);selected=selected[:a.limit] if a.limit else selected
    assert set(done)=={r['id'] for r in selected}
    for mode in ['weighted','readout']:(out/mode).mkdir(exist_ok=True)
    diagnostics=[]
    for layer in range(28):
        wk,ki=ridge(stats['kg'][layer],stats['kc'][layer])
        wv,vi=ridge(stats['vg'][layer],stats['vc'][layer])
        wr,ri=ridge(stats['rg'][layer],stats['rc'][layer])
        wr=wr.permute(1,0,2).reshape(1024,1024).contiguous()
        metrics={}
        for mode,v in [('weighted',wv),('readout',wr)]:
            np.savez(out/mode/f'layer_{layer:02d}_k.npz',weight=wk.numpy())
            np.savez(out/mode/f'layer_{layer:02d}_v.npz',weight=v.numpy())
            weighted=quadratic(v,stats['vg'][layer],stats['vc'][layer],stats['vyy'][layer])
            vh=v.double().reshape(1024,8,128).permute(1,0,2)
            readout=quadratic(vh,stats['rg'][layer],stats['rc'][layer],stats['ryy'][layer].sum())
            penalty=.01*float(v.double().square().sum())
            metrics[mode]=dict(weighted_train_loss=weighted,readout_train_loss=readout,penalty=penalty)
        def loss(mode,key):return metrics[mode][key]+metrics[mode]['penalty']
        assert loss('weighted','weighted_train_loss')<=loss('readout','weighted_train_loss')+1e-4
        assert loss('readout','readout_train_loss')<=loss('weighted','readout_train_loss')+1e-4
        diagnostics.append(dict(layer=layer,k=ki,v_weighted=vi,v_readout=ri,losses=metrics))
        print('solved and cross-objective checked layer',layer,flush=True)
    artifacts={str(p.relative_to(out)):digest(p) for p in out.glob('*/*.npz')}
    for i in range(28):
        assert np.array_equal(np.load(out/'weighted'/f'layer_{i:02d}_k.npz')['weight'],np.load(out/'readout'/f'layer_{i:02d}_k.npz')['weight'])
    atomic_json(out/'run.json',dict(metadata=expected,train_count=len(done),artifact_sha256=artifacts,
                                  diagnostics=diagnostics,shards=shard_reports))
    atomic_json(out/'audit.json',dict(passed=True,n=len(done),K_bitwise_equal=True,
        objectives_optimal_against_other_solution=True,run_sha256=digest(out/'run.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','merge']);p.add_argument('--pair',choices=PAIRS,required=True)
    p.add_argument('--output',required=True);p.add_argument('--shards',type=int,default=2);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--device',default='cuda:0');p.add_argument('--save-every',type=int,default=25)
    a=p.parse_args();(fit if a.action=='fit' else merge)(a)
