"""Independent checks of serialized FP32 maps against saved corpus normal equations."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from core import digest,atomic_json
from fit import rows,DATA,paths,code_hashes

HERE=Path(__file__).resolve().parent

def verify_artifact(out):
    out=Path(out);run=json.loads((out/'run.json').read_text());m=run['metadata']
    assert m['code_sha256']==code_hashes()
    assert m['data_sha256']==digest(DATA) and m['data']==str(DATA)
    assert m['bias'] is False and m['ridge_lambda']==.01 and m['attention_normalization']=='prefix'
    assert m['source_layer_map']==list(range(28))
    assert (m['source'],m['target'])==paths(m['pair'])
    for path,value in m['model_config_sha256'].items():assert digest(Path(path)/'config.json')==value
    selected=rows(DATA);selected=selected[:m['limit']] if m['limit'] else selected
    assert run['train_count']==len(selected)
    expected={f'{mode}/layer_{i:02d}_{side}.npz' for mode in ['weighted','readout'] for i in range(28) for side in ['k','v']}
    assert set(run['artifact_sha256'])==expected
    assert {str(p.relative_to(out)) for p in out.glob('*/*.npz')}==expected
    for path,value in run['artifact_sha256'].items():assert digest(out/path)==value
    original=json.loads((out/'audit.json').read_text())
    assert original['passed'] and original['n']==len(selected) and original['run_sha256']==digest(out/'run.json')
    return run,selected

def mapped(out,mode,layer,side):
    a=np.load(Path(out)/mode/f'layer_{layer:02d}_{side}.npz')['weight']
    assert a.dtype==np.float32 and a.shape==(1024,1024) and np.isfinite(a).all()
    return torch.from_numpy(a).double()

def residual(g,c,w):
    r=g@w+.01*w-c
    norm=float(r.norm()/c.norm().clamp_min(1e-30))
    assert norm<1e-5, ('Serialized FP32 map fails regularized normal equations',norm)
    return norm

def objective(g,c,yy,w):
    return float((w*(g@w)).sum() - 2*(w*c).sum()+yy+.01*w.square().sum())

@torch.inference_mode()
def audit(out,deep=True):
    out=Path(out);run,selected=verify_artifact(out);receipt=out/'independent_fit_audit.json'
    if not deep:
        p=json.loads(receipt.read_text())
        assert p['passed'] and p['n']==len(selected) and p['run_sha256']==digest(out/'run.json')
        assert p['auditor_sha256']==digest(Path(__file__))
        for name,value in p['moment_sha256'].items():assert digest(out/name)==value
        return p
    total=None;seen=[];moment_hashes={}
    for shard in range(run['metadata']['shards']):
        path=out/f'moments_{shard:02d}.pt';proof=json.loads((out/f'shard_{shard:02d}.json').read_text())
        value=digest(path);assert value==proof['moments_sha256'] and proof['complete'];moment_hashes[path.name]=value
        item=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        meta=item['metadata'].copy();assert meta.pop('shard')==shard and meta==run['metadata']
        expect=[r['id'] for i,r in enumerate(selected) if i%run['metadata']['shards']==shard]
        assert item['done']==expect==proof['done']
        assert [r['id'] for r in item['diagnostics']]==expect
        for d,r in zip(item['diagnostics'],selected[shard::run['metadata']['shards']]):
            assert d['prefix']==len(r['prefix_ids']) and d['continuation']==len(r['continuation_ids'])
            assert d['max_attention_row_mass_error']<2e-6
        seen+=item['done']
        for key,v in item['stats'].items():assert v.dtype==torch.float64 and torch.isfinite(v).all(),key
        if total is None:total={k:v.clone() for k,v in item['stats'].items()}
        else:
            for k,v in item['stats'].items():total[k]+=v
    assert len(seen)==len(set(seen))==len(selected)
    diagnostics=[]
    for i in range(28):
        wk=mapped(out,'weighted',i,'k');kr=mapped(out,'readout',i,'k');assert torch.equal(wk,kr)
        wv=mapped(out,'weighted',i,'v');rv=mapped(out,'readout',i,'v')
        rh=rv.reshape(1024,8,128).permute(1,0,2);wh=wv.reshape(1024,8,128).permute(1,0,2)
        for key in ['kg','vg','rg']:
            g=total[key][i];assert float((g-g.mT).norm()/g.norm().clamp_min(1e-30))<1e-12
        res=dict(k=residual(total['kg'][i],total['kc'][i],wk),weighted_v=residual(total['vg'][i],total['vc'][i],wv),
                 readout_v=residual(total['rg'][i],total['rc'][i],rh))
        weighted=[objective(total['vg'][i],total['vc'][i],total['vyy'][i],w) for w in [wv,rv]]
        readout=[objective(total['rg'][i],total['rc'][i],total['ryy'][i].sum(),w) for w in [wh,rh]]
        assert weighted[0]<=weighted[1]+1e-4 and readout[1]<=readout[0]+1e-4
        diagnostics.append(dict(layer=i,serialized_fp32_residual=res,K_equal=True,weighted_objectives=weighted,readout_objectives=readout))
    p=dict(passed=True,n=len(selected),run_sha256=digest(out/'run.json'),moment_sha256=moment_hashes,
           auditor_sha256=digest(Path(__file__)),all_shapes_and_dtype_checked=True,exact_shard_membership=True,
           K_bitwise_equal=True,layer_diagnostics=diagnostics)
    atomic_json(receipt,p);print('INDEPENDENT FIT AUDIT PASSED',out,len(selected),flush=True);return p

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact');p.add_argument('--receipt-only',action='store_true')
    a=p.parse_args();audit(a.artifact,not a.receipt_only)
