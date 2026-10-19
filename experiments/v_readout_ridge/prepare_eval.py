"""Freeze every evaluation prompt before inference and compare paired tokenizer IDs."""
import collections,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from core import digest,atomic_json
from evaluate import dataset,user_prompt,chat_ids,mc_item
HERE=Path(__file__).resolve().parent

def cases():
    result=[]
    for thinking in [False,True]:
        for task in ['arc','mmlu']:
            result.append(dict(task=task,length=None,thinking=thinking,batch_size=32))
        for length in ['4k','8k','16k','32k']:
            result.append(dict(task='ruler',length=length,thinking=thinking,batch_size=8 if length!='32k' else 4))
    result.append(dict(task='hotpot',length=None,thinking=False,batch_size=16))
    return result

def case_name(case):
    return case['task']+('_'+case['length'] if case['length'] else '')+('_on' if case['thinking'] else '_off')

def main():
    tok=[AutoTokenizer.from_pretrained('/home/tiger/models/Qwen3-'+name,local_files_only=True) for name in ['0.6B','1.7B']]
    assert tok[0].get_vocab()==tok[1].get_vocab() and tok[0].chat_template==tok[1].chat_template
    summary=[];all_hashes={}
    for c in cases():
        data,rows=dataset(c['task'],c['length']);lengths=[];hashes=[]
        count={'arc':1172,'mmlu':1400,'ruler':400,'hotpot':7405}[c['task']];assert len(rows)==count
        if c['task']=='mmlu':assert set(collections.Counter(x['category'] for x in rows).values())=={100}
        for i,row in enumerate(rows):
            ids=chat_ids(tok[0],user_prompt(row,c['task']),c['thinking'])
            other=chat_ids(tok[1],user_prompt(row,c['task']),c['thinking']);assert ids==other
            budget=256 if c['task']=='hotpot' else 2048 if c['task']=='ruler' else 8192 if c['thinking'] else 1024
            assert len(ids)+budget<=40960
            if c['task'] in ['arc','mmlu']:
                assert len(ids)<=3000
                mc_item(row,c['task'])
            lengths.append(len(ids));hashes.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        name=case_name(c);all_hashes[name]=hashes
        summary.append(dict(**c,name=name,n=len(rows),data=str(data),data_sha256=digest(data),
            min_prompt_tokens=min(lengths),max_prompt_tokens=max(lengths),max_new=budget,
            prompt_hash_list_sha256=hashlib.sha256(json.dumps(hashes).encode()).hexdigest()))
        print(name,len(rows),min(lengths),max(lengths),flush=True)
    atomic_json(HERE/'checks/evaluation_inputs.json',dict(passed=True,cases=summary,
        tokenizer_identity_all_prompts=True,code_sha256={p.name:digest(p) for p in [Path(__file__),HERE/'evaluate.py']}))
    atomic_json(HERE/'checks/prompt_hashes.json',all_hashes)

if __name__=='__main__':main()
