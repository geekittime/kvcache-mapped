"""Recheck model provenance and held-out records independently of live numerical code."""
import argparse,json,math
from pathlib import Path
from core import atomic_json,digest
from fit import paths,rows
from validate import audit as original_audit,DATA

HERE=Path(__file__).resolve().parent

def provenance():
    p=json.loads((HERE/'checks/provenance.json').read_text());assert p['passed']
    for files in p['models'].values():
        for name,value in files.items():assert digest(name)==value, name
    frozen=json.loads((HERE/'configs/frozen_protocol_v1.json').read_text())
    for name in ['fit.py','readout.py','core.py','evaluate.py','validate.py','prepare_eval.py']:
        assert digest(HERE/name)==frozen['files'][name],name
    return True

def validation(out,formal=True):
    out=Path(out);original_audit(out)
    r=json.loads((out/'results.json').read_text());p=r['protocol']
    assert p==json.loads((out/'protocol.json').read_text())
    assert (p['source'],p['target'])==paths(p['pair']) and Path(p['data']).resolve()==DATA
    if formal:
        assert p['limit']==0 and r['n']==64
        assert Path(p['artifact']).resolve()==HERE/'artifacts'/p['pair']/'full_v1'
    expected=rows(DATA);expected=expected[:p['limit']] if p['limit'] else expected
    records=rows(out/'predictions.jsonl')
    assert [x['id'] for x in records]==[x['id'] for x in expected]
    for x in records:
        assert set(x['values'])=={'weighted','readout'}
        assert set(x['target_energy'])=={'weighted_v','readout_v'}
        assert all(math.isfinite(v) and v>0 for v in x['target_energy'].values())
        assert x['values']['weighted']['native_nll_sum']==x['values']['readout']['native_nll_sum']
    result=dict(passed=True,n=r['n'],formal=formal,results_sha256=digest(out/'results.json'),auditor_sha256=digest(__file__))
    atomic_json(out/'independent_validation_audit.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validation');parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();print(validation(a.validation,not a.smoke) if a.validation else provenance())
