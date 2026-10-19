"""Fresh record-level scoring and input/output identity checks for every completed job."""
import argparse,json,hashlib,math
from pathlib import Path
from transformers import AutoTokenizer
from core import digest,atomic_json
from evaluate import dataset,user_prompt,chat_ids,read_jsonl,score_prediction,summarize
HERE=Path(__file__).resolve().parent

def audit(out):
    out=Path(out);r=json.loads((out/'results.json').read_text());p=r['protocol']
    assert r['complete'] and p==json.loads((out/'protocol.json').read_text())
    for name,value in p['code_sha256'].items():assert digest(HERE/name)==value,name
    assert digest(Path(p['artifact'])/'run.json')==p['mapper_run_sha256']
    run=json.loads((Path(p['artifact'])/'run.json').read_text())
    for name,value in run['metadata']['code_sha256'].items():assert digest(HERE/name)==value,name
    for name,value in run['artifact_sha256'].items():assert digest(Path(p['artifact'])/name)==value,name
    assert digest(Path(p['target_model'])/'config.json')==p['model_config_sha256']
    assert digest(Path(p['source_model'])/'config.json')==p['source_config_sha256']
    assert digest(Path(p['target_model'])/'tokenizer_config.json')==p['tokenizer_config_sha256']
    data,rows=dataset(p['task'],p['length']);assert digest(data)==p['data_sha256']
    if p['limit']:rows=rows[:p['limit']]
    expected={i:row for i,row in enumerate(rows) if i%p['shards']==p['shard']}
    preds=read_jsonl(out/'predictions.jsonl')
    assert len(preds)==len(expected) and {x['index'] for x in preds}==set(expected)
    tok=AutoTokenizer.from_pretrained(p['target_model'],local_files_only=True)
    for x in preds:
        row=expected[x['index']];ids=chat_ids(tok,user_prompt(row,p['task']),p['thinking'])
        assert len(ids)==x['prompt_tokens']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==x['prompt_ids_sha256']
        assert tok.decode(x['token_ids'],skip_special_tokens=True)==x['text']
        assert len(x['token_ids'])==x['generated_tokens']<=p['max_new']
        assert x['stop_reason'] in ('length','eos')
        if x['stop_reason']=='length':assert x['generated_tokens']==p['max_new']
        else:assert x['eos_token_id'] is not None
        assert all(x[k]==v for k,v in score_prediction(row,p['task'],x['text']).items())
    fresh=summarize(preds,p['task'])
    assert all(r[k]==v for k,v in fresh.items())
    assert digest(out/'predictions.jsonl')==r['predictions_sha256']
    proof=dict(passed=True,n=len(preds),complete_dataset=not p['limit'] and p['shards']==1,
               predictions_sha256=r['predictions_sha256'],results_sha256=digest(out/'results.json'),
               checks=['exact row coverage','source/mapper/data fingerprints','retokenized prompts',
                       'decoded saved output tokens','fresh score and aggregate','budget and stop accounting'])
    atomic_json(out/'audit.json',proof);print(json.dumps(proof),flush=True);return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');audit(p.parse_args().output)
