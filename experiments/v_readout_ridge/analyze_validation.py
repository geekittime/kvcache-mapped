"""Descriptive held-out diagnostics only; never used to select hyperparameters."""
import json
from pathlib import Path
import numpy as np
from core import atomic_json,digest
from fit import rows
from validate import DATA

HERE=Path(__file__).resolve().parent
def main():
    refs={x['id']:x for x in rows(DATA)};report={}
    for pair in ['06_17','17_06']:
        path=HERE/'checks'/f'validation_{pair}_full/predictions.jsonl';records=rows(path)
        assert len(records)==64 and {x['id'] for x in records}==set(refs)
        groups={'all':records}
        for key in ['kind','reasoning']:
            for value in sorted({r[key] for r in refs.values()}):
                groups[f'{key}={value}']=[x for x in records if refs[x['id']][key]==value]
        summaries={}
        for name,rr in groups.items():
            result={}
            for metric in ['kl_sum','mapped_nll_sum','readout_v','weighted_v']:
                values={}
                for mode in ['weighted','readout']:
                    values[mode]=np.array([x['values'][mode][metric]/(x['continuation'] if metric.endswith('_sum') else x['target_energy'][metric]) for x in rr])
                delta=values['readout']-values['weighted']
                result[metric]=dict(weighted_mean=float(values['weighted'].mean()),readout_mean=float(values['readout'].mean()),
                    delta_mean=float(delta.mean()),delta_median=float(np.median(delta)),readout_lower_count=int((delta<0).sum()),n=len(rr))
            summaries[name]=result
        report[pair]=dict(predictions_sha256=digest(path),groups=summaries)
    atomic_json(HERE/'checks/validation_subgroups.json',dict(descriptive_only=True,not_used_for_selection=True,
        formulas='KL/CE per continuation token within each example; reconstruction error divided by native teacher energy within each example; averages over examples',pairs=report))
    print(json.dumps({p:v['groups']['all']['kl_sum'] for p,v in report.items()},indent=2))

if __name__=='__main__':main()
