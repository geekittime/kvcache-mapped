"""Exact normalized question-overlap screen; not a semantic contamination guarantee."""
import json,re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from core import atomic_json,digest
from fit import rows,DATA
from evaluate import dataset

HERE=Path(__file__).resolve().parent
def normalize(text):return ' '.join(re.findall(r'\w+',text.lower()))

def main():
    tokenizer=AutoTokenizer.from_pretrained('/home/tiger/models/Qwen3-1.7B',local_files_only=True)
    records=rows(DATA)
    # Include continuations and all packed filler text, not just target prompts.
    texts=[normalize(tokenizer.decode(r['prefix_ids']+r['continuation_ids'])) for r in records]
    joined='\n'.join(texts);checks=[]
    for task in ['arc','mmlu','hotpot']:
        path,data=dataset(task)
        queries=[]
        for i,r in enumerate(data):
            q=normalize(r['question'])
            queries.append((i,q))
        def locate(item):
            i,q=item
            if len(q)<40 or len(q.split())<8:return ('too_short',i,[])
            if q not in joined:return ('absent',i,[])
            matched=[r['id'] for r,t in zip(records,texts) if q in t]
            return ('matched',i,matched)
        with ThreadPoolExecutor(max_workers=8) as pool:found=list(pool.map(locate,queries))
        checks.append(dict(task=task,dataset_sha256=digest(path),n=len(data),
            screened=sum(s!='too_short' for s,_,_ in found),short_question_excluded=sum(s=='too_short' for s,_,_ in found),
            normalized_exact_matches=[dict(index=i,training_ids=ids) for s,i,ids in found if s=='matched']))
    proof=dict(completed=True,training_sha256=digest(DATA),train_count=400,screen='Full normalized question substring in training prefixes+continuations, >=40 characters and >=8 words',
        limitation='No semantic/paraphrase detection; short ambiguous questions excluded. This cannot certify the original model pretraining corpus.',checks=checks)
    atomic_json(HERE/'checks/benchmark_overlap_screen.json',proof)
    print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
