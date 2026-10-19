"""Bind formal runs to the pre-inference scientific protocol and prepared inputs."""
import json
from pathlib import Path
from core import digest
from fit import paths

HERE=Path(__file__).resolve().parent

def validate_case(protocol, case, count):
    p=protocol
    assert p['task']==case['task'] and p['length']==case['length'] and p['thinking']==case['thinking']
    assert p['mode'] in ('weighted','readout')
    assert p['max_new']==case['max_new'], 'Generation budget differs from frozen case'
    assert p['batch_size']==case['batch_size'], 'Batch geometry differs from frozen case'
    assert p['shards']==1 and p['shard']==0, 'Formal cells must cover the entire prepared case'
    assert p['limit'] in (0,case['batch_size']), 'Only full or one-batch smoke is allowed'
    assert count==(p['limit'] or case['n']), 'Wrong complete sample count'
    assert Path(p['data']).resolve()==Path(case['data']).resolve()
    assert p['data_sha256']==case['data_sha256'], 'Dataset changed after preparation'

def formal_protocol(p, count, run):
    prepared=json.loads((HERE/'checks/evaluation_inputs.json').read_text());assert prepared['passed']
    for name,value in prepared['code_sha256'].items():assert digest(HERE/name)==value
    case=next(c for c in prepared['cases'] if (c['task'],c['length'],c['thinking'])==(p['task'],p['length'],p['thinking']))
    validate_case(p,case,count)
    pair=run['metadata']['pair'];sp,tp=paths(pair)
    assert Path(p['source_model']).resolve()==Path(sp).resolve()
    assert Path(p['target_model']).resolve()==Path(tp).resolve()
    assert Path(p['artifact']).resolve()==HERE/'artifacts'/pair/'full_v1'
    assert run['train_count']==400 and run['metadata']['limit']==0
    assert run['metadata']['bias'] is False and run['metadata']['ridge_lambda']==.01
    assert run['metadata']['attention_normalization']=='prefix'
    return case

def matching_pair(a,b):
    # The condition name/output directory are the only intended semantic differences.
    ignored={'mode','output','device'}
    for key in a.keys()|b.keys():
        if key not in ignored:assert a.get(key)==b.get(key),f'Unpaired protocol setting: {key}'
