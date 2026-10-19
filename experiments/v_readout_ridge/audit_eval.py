"""Fresh record-level scoring and input/output identity checks for every completed job."""
import argparse,json,hashlib,math
from pathlib import Path
from transformers import AutoTokenizer
from core import digest,atomic_json
from evaluate import dataset,user_prompt,chat_ids,read_jsonl,score_prediction,summarize
from protocol_guards import formal_protocol
HERE=Path(__file__).resolve().parent

def audit(out,require_formal=False):
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
    formal=require_formal or run['train_count']==400
    case=formal_protocol(p,len(preds),run) if formal else None
    if formal:
        runtime=json.loads((out/'runtime.json').read_text())
        assert runtime['cublas_workspace_config']==':4096:8' and runtime['deterministic_algorithms'] is True
        assert runtime['seed']==42 and runtime['wrapper_sha256']==digest(HERE/'deterministic_eval.py')
        assert runtime['evaluate_sha256']==digest(HERE/'evaluate.py')
    frozen_hashes=json.loads((HERE/'checks/prompt_hashes.json').read_text())[case['name']] if formal else None
    if formal:
        assert hashlib.sha256(json.dumps(frozen_hashes).encode()).hexdigest()==case['prompt_hash_list_sha256']
    gc=json.loads((Path(p['target_model'])/'generation_config.json').read_text())
    model_vocab=json.loads((Path(p['target_model'])/'config.json').read_text())['vocab_size']
    eos=gc.get('eos_token_id') or tok_eos_from_config(p)
    eos=eos if isinstance(eos,list) else [eos]
    tok=AutoTokenizer.from_pretrained(p['target_model'],local_files_only=True)
    for x in preds:
        row=expected[x['index']];ids=chat_ids(tok,user_prompt(row,p['task']),p['thinking'])
        assert len(ids)==x['prompt_tokens']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==x['prompt_ids_sha256']
        if formal:assert x['prompt_ids_sha256']==frozen_hashes[x['index']]
        assert x['id']==str(row.get('id',row.get('question_id',x['index'])))
        assert tok.decode(x['token_ids'],skip_special_tokens=True)==x['text']
        assert len(x['token_ids'])==x['generated_tokens']<=p['max_new']
        # Added special tokens lie outside tokenizer.vocab_size; model logits
        # also include padded vocabulary entries. Check the actual model domain.
        assert all(isinstance(t,int) and 0<=t<model_vocab and t not in eos for t in x['token_ids'])
        assert x['stop_reason'] in ('length','eos')
        if x['stop_reason']=='length':assert x['generated_tokens']==p['max_new']
        else:
            assert x['eos_token_id'] in eos
            assert x['generated_tokens']<p['max_new']
        assert all(x[k]==v for k,v in score_prediction(row,p['task'],x['text']).items())
    fresh=summarize(preds,p['task'])
    assert all(r[k]==v for k,v in fresh.items())
    assert digest(out/'predictions.jsonl')==r['predictions_sha256']
    proof=dict(passed=True,n=len(preds),complete_dataset=formal and not p['limit'] and p['shards']==1,
               frozen_case_verified=formal,auditor_sha256=digest(Path(__file__)),
               predictions_sha256=r['predictions_sha256'],results_sha256=digest(out/'results.json'),
               checks=['exact row coverage','source/mapper/data fingerprints','retokenized prompts',
                       'decoded saved output tokens','fresh score and aggregate','budget and stop accounting'])
    atomic_json(out/'audit.json',proof);print(json.dumps(proof),flush=True);return proof

def tok_eos_from_config(p):
    return json.loads((Path(p['target_model'])/'config.json').read_text())['eos_token_id']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');audit(p.parse_args().output)
