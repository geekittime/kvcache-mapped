"""Only audited full cells enter paired score differences; no partial-run scores."""
import argparse,json
from pathlib import Path
import numpy as np
from core import atomic_json,digest
from evaluate import read_jsonl
from audit_eval import audit
from prepare_eval import cases,case_name
HERE=Path(__file__).resolve().parent

def interval(delta,groups=None):
    """Paired percentile bootstrap, stratified by subject for MMLU-PRO."""
    delta=np.asarray(delta,dtype=np.float64);rng=np.random.default_rng(42);values=[]
    indices=[np.arange(len(delta))] if groups is None else [np.flatnonzero(np.asarray(groups)==g) for g in sorted(set(groups))]
    for _ in range(20):
        draws=[delta[rng.choice(idx,size=(100,len(idx)),replace=True)].mean(1) for idx in indices]
        values.extend(np.stack(draws).mean(0).tolist())
    return [float(x) for x in np.quantile(values,[.025,.975])]

def main(require_complete):
    comparisons=[];accepted=0
    for pair in ['06_17','17_06']:
        for case in cases():
            name=case_name(case);entry=dict(pair=pair,case=name,status='pending');conditions={}
            for mode in ['weighted','readout']:
                base=HERE/'runs'/pair/mode/name
                if not (base/'stage.json').exists():continue
                stage=json.loads((base/'stage.json').read_text());assert stage['passed'] and stage['smoke_full_exact']
                proof=audit(base/'full');assert proof['complete_dataset']
                result=json.loads((base/'full/results.json').read_text())
                assert stage['full_results_sha256']==digest(base/'full/results.json') and result['n']==stage['n']
                conditions[mode]=(result,read_jsonl(base/'full/predictions.jsonl'));accepted+=1
            if len(conditions)==2:
                a,ap=conditions['weighted'];b,bp=conditions['readout']
                assert len(ap)==len(bp) and a['protocol']['data_sha256']==b['protocol']['data_sha256']
                ap=sorted(ap,key=lambda x:x['index']);bp=sorted(bp,key=lambda x:x['index'])
                for x,y in zip(ap,bp):
                    assert (x['index'],x['id'],x['prompt_ids_sha256'],x['gold'])==(y['index'],y['id'],y['prompt_ids_sha256'],y['gold'])
                metrics=['em','f1','raw_em','raw_f1'] if case['task']=='hotpot' else ['recall'] if case['task']=='ruler' else ['macro_accuracy' if case['task']=='mmlu' else 'accuracy']
                values={}
                for key in metrics:
                    itemkey='correct' if key in ('accuracy','macro_accuracy') else key
                    delta=[y[itemkey]-x[itemkey] for x,y in zip(ap,bp)]
                    groups=[x['category'] for x in ap] if case['task']=='mmlu' else None
                    values[key]=dict(weighted=a[key],readout=b[key],delta=b[key]-a[key],paired_bootstrap_95=interval(delta,groups))
                entry.update(status='accepted',n=len(ap),metrics=values,
                    length_stops={m:r[0]['length_stops'] for m,r in conditions.items()},
                    unparsed={m:r[0]['unparsed'] for m,r in conditions.items()})
            comparisons.append(entry)
    complete=accepted==52
    if require_complete:assert complete,f'Only {accepted}/52 audited complete cells'
    result=dict(complete=complete,accepted_cells=accepted,expected_cells=52,comparisons=comparisons,
        interpretation='Paired fixed-dataset comparison; only V loss differs. Bootstrap intervals describe sampling uncertainty, not variation over fit seeds. No claim of exact hidden paper split or guaranteed improvement.',
        protocol_sha256=digest(HERE/'README.md'))
    lines=['# V weighted ridge vs V readout ridge','',f'Audited full cells: {accepted}/52. Complete: {complete}.',
           '', '| Direction | Task/mode | n | Metric | Weighted (%) | Readout (%) | Δ (pp) | Paired 95% CI (pp) |',
           '|---|---|---:|---|---:|---:|---:|---|']
    for c in comparisons:
        if c['status']!='accepted':
            lines.append(f"| {c['pair']} | {c['case']} | — | pending | — | — | — | — |")
            continue
        for metric,v in c['metrics'].items():
            lo,hi=v['paired_bootstrap_95']
            lines.append(f"| {c['pair']} | {c['case']} | {c['n']} | {metric} | {v['weighted']*100:.3f} | {v['readout']*100:.3f} | {v['delta']*100:+.3f} | [{lo*100:+.3f}, {hi*100:+.3f}] |")
    lines+=['','06_17 = Qwen3-0.6B → 1.7B; 17_06 = reverse. on/off indicates reasoning mode.',
            'MMLU-Pro uses macro accuracy; ARC-C accuracy; RULER NIAH recall at each length; HotpotQA answer EM/F1 and raw scores.',
            'Intervals use 2,000 paired bootstrap draws (MMLU stratified by subject). They do not measure training-seed variability. No tuning on these outcomes.',
            'Full protocol and unavailable author details: README.md. Process and stage evidence: PROCESS.md and checks/.', '']
    atomic_json(HERE/'RESULTS.json',result);(HERE/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=complete,accepted=accepted)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');main(p.parse_args().require_complete)
