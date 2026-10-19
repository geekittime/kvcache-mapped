#!/usr/bin/env python3
"""Freeze auditable Nemotron pairs following KV-Lingo Appendix C.

The authors do not release sample IDs or their packing implementation. All
sampling choices below are deterministic, stored, and explicitly local choices.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, multiprocessing
from pathlib import Path
import requests
from transformers import AutoTokenizer
from core import ROOT, atomic_json, digest

REVISION = '1a9454ed054b8544503ab8d8c0a519d141a44c5b'
DATASET = 'nvidia/Nemotron-SFT-Instruction-Following-Chat-v2'

def download(mode, count, directory):
    path = directory / f'{mode}_{count}.jsonl'
    if path.exists():
        return [json.loads(s) for s in path.open()]
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/reasoning_{mode}.jsonl'
    records = []
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        # Binary iter_lines only treats actual LF as a separator.
        for line in response.iter_lines(chunk_size=1<<20):
            if line:
                records.append(json.loads(line))
            if len(records) == count:
                break
            if len(records) and len(records)%10000==0:
                print(f'download {mode}: {len(records)}/{count}',flush=True)
    if len(records) != count:
        raise RuntimeError(f'incomplete download: {len(records)}/{count}')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    return records

def strip_reasoning(content):
    return re.sub(r'<think>.*?</think>\s*', '', content, flags=re.S)

def conversation_pair(tokenizer, record, rng, max_cont=2048):
    messages = record['messages']
    turns = [i for i,m in enumerate(messages) if m['role']=='assistant' and i>0
             and messages[i-1]['role']=='user' and isinstance(m.get('content'),str)]
    if not turns:
        return None
    turn = rng.choice(turns)
    answer = messages[turn]
    trace = answer.get('reasoning_content') or ''
    content = answer['content']
    if trace and '<think>' not in content:
        content = '<think>\n'+trace+'\n</think>\n\n'+content
    reasoning = '<think>' in content and bool(re.search(r'<think>\s*\S',content))
    history = [{'role':m['role'], 'content':strip_reasoning(m.get('content') or '')}
               for m in messages[:turn]]
    prefix = tokenizer.apply_chat_template(history, tokenize=False,
                        add_generation_prompt=True, enable_thinking=reasoning)
    continuation = content if reasoning else strip_reasoning(content)
    continuation += '<|im_end|>'
    x = tokenizer.encode(prefix, add_special_tokens=False)
    y = tokenizer.encode(continuation, add_special_tokens=False)[:max_cont]
    if not y or not 10<=len(x)<=10000:
        return None
    # Closed conversation for distractor context: answers, not just user prompts.
    complete = history + [{'role':'assistant','content':strip_reasoning(answer['content'])}]
    filler = tokenizer.apply_chat_template(complete, tokenize=False, add_generation_prompt=False)
    return dict(uuid=record['uuid'], assistant_turn=turn, reasoning='on' if reasoning else 'off',
                prefix_ids=x, continuation_ids=y, filler_ids=tokenizer.encode(filler,add_special_tokens=False))

def select_natural(pool, n, rng):
    # Stratify approximately uniformly over the paper's 10--10k token range. Exact author bins/record sampler are unpublished.
    selected=[]
    used=set()
    for i in range(n):
        mode='off' if i%2==0 else 'on'
        choices=[p for p in pool if p['reasoning']==mode and p['uuid'] not in used]
        target=10+(10000-10)*((i//2+.5)/(n/2))
        if not choices:
            raise ValueError('insufficient distinct reasoning records')
        bin_index=min(int(target//1000),9)
        choices=[p for p in choices if min(len(p['prefix_ids'])//1000,9)==bin_index]
        if not choices:
            raise ValueError(f'insufficient {mode} records in bin {bin_index}; increase --candidates')
        nearest=sorted(choices,key=lambda p:abs(len(p['prefix_ids'])-target))[:4]
        chosen=rng.choice(nearest);selected.append(chosen);used.add(chosen['uuid'])
    return selected

def construct(pool, n, rng):
    assert n%4==0
    natural=select_natural(pool,n//2,rng)
    used={p['uuid'] for p in natural}
    # Packed targets need not follow natural-length bins: their final prefixes
    # are controlled by complete-conversation packing to 12--16k.
    targets=[]
    for i in range(n//2):
        mode='off' if i%2==0 else 'on'
        choices=[p for p in pool if p['uuid'] not in used and p['reasoning']==mode]
        chosen=rng.choice(choices);targets.append(chosen);used.add(chosen['uuid'])
    result=[]
    for kind, group in [('natural',natural),('packed',targets)]:
        for p in group:
            prefix=list(p['prefix_ids']); members=[p['uuid']]
            if kind=='packed':
                budget=rng.randint(12000,16000)
                fillers=[q for q in pool if q['uuid']!=p['uuid']]
                rng.shuffle(fillers)
                chunks=[]; length=len(prefix)
                for q in fillers:
                    k=len(q['filler_ids'])
                    if length+k<=budget:
                        chunks.append(q['filler_ids']);length+=k;members.append(q['uuid'])
                    if length>=budget:break
                # Discrete conversation lengths can leave a small gap below
                # 12k (e.g. 11,998). Add the smallest remaining conversation
                # that keeps the required 16k ceiling.
                if length<12000:
                    candidates=[q for q in fillers if q['uuid'] not in members and length+len(q['filler_ids'])<=16000]
                    if candidates:
                        q=min(candidates,key=lambda z:len(z['filler_ids']))
                        chunks.append(q['filler_ids']);length+=len(q['filler_ids']);members.append(q['uuid'])
                if not 12000<=length<=16000:
                    raise ValueError(f'cannot pack complete conversations to budget {budget}: {length}')
                # Evaluated query is at the very end; its answer is NEVER a filler.
                prefix=[token for chunk in chunks for token in chunk]+prefix
            row={k:v for k,v in p.items() if k not in ['filler_ids','prefix_ids']}
            row.update(kind=kind,prefix_ids=prefix,member_uuids=members,
                       prefix_tokens=len(prefix),continuation_tokens=len(p['continuation_ids']))
            result.append(row)
    for i,r in enumerate(result):
        r['id']=i
        assert r['reasoning']==('off' if i%2==0 else 'on')
    return result

def init_tokenizer(seed):
    global worker_tokenizer,worker_seed
    worker_tokenizer=AutoTokenizer.from_pretrained('/home/tiger/models/Qwen3-1.7B',local_files_only=True)
    worker_seed=seed

def tokenize_record(line):
    record=json.loads(line)
    # A UUID-specific RNG keeps assistant-turn choice invariant to worker count.
    seed=int(hashlib.sha256((str(worker_seed)+':'+record['uuid']).encode()).hexdigest()[:16],16)
    return conversation_pair(worker_tokenizer,record,random.Random(seed))

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('/home/tiger/models/Qwen3-1.7B',local_files_only=True)
    rng=random.Random(a.seed);pool=[]
    raw_manifest=json.loads((Path(a.raw_dir)/'manifest.json').read_text())
    assert raw_manifest['dataset']==DATASET and raw_manifest['revision']==REVISION
    for mode in ['off','on']:
        raw_path=Path(raw_manifest['files'][mode]['path'])
        assert digest(raw_path)==raw_manifest['files'][mode]['sha256']
        assert raw_manifest['files'][mode]['records']>=a.candidates
        from itertools import islice
        with multiprocessing.Pool(a.workers,initializer=init_tokenizer,initargs=(a.seed,)) as workers:
            for index,p in enumerate(workers.imap(tokenize_record,islice(raw_path.open(),a.candidates),chunksize=32)):
                if p is not None:pool.append(p)
                if (index+1)%5000==0:print(f'tokenized {mode}: {index+1}/{a.candidates}',flush=True)
        print(f'tokenized {mode}: pool={len(pool)}',flush=True)
    # Partition at UUID level before packing; fillers are disjoint as well.
    pool=list({p['uuid']:p for p in pool}.values());rng.shuffle(pool)
    print('unique candidate pairs',len(pool),flush=True)
    cut=int(len(pool)*.8);train_pool,valid_pool=pool[:cut],pool[cut:]
    manifests={}
    for name,pool,count in [('train',train_pool,a.train_count),('validation',valid_pool,64)]:
        rows=construct(pool,count,rng);path=out/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        manifests[name]={'count':len(rows),'sha256':digest(path),'prefix_tokens':sum(r['prefix_tokens'] for r in rows),
            'natural_histogram_1k':[sum(r['kind']=='natural' and min(r['prefix_tokens']//1000,9)==i for r in rows) for i in range(10)],
            'by_kind':{k:{'count':sum(r['kind']==k for r in rows),
              'min':min(r['prefix_tokens'] for r in rows if r['kind']==k),
              'max':max(r['prefix_tokens'] for r in rows if r['kind']==k)} for k in ['natural','packed']},
            'on':sum(r['reasoning']=='on' for r in rows)}
    tr=[json.loads(s) for s in (out/'train.jsonl').open()]
    va=[json.loads(s) for s in (out/'validation.jsonl').open()]
    assert not {u for r in tr for u in r['member_uuids']} & {u for r in va for u in r['member_uuids']}
    atomic_json(out/'manifest.json',dict(dataset=DATASET,revision=REVISION,seed=a.seed,
        raw_manifest=raw_manifest,raw_manifest_sha256=digest(Path(a.raw_dir)/'manifest.json'),
        prepare_code_sha256=digest(__file__),
        candidates=a.candidates,local_choices=['first N records per pinned stream','random assistant turn',
        'strict 1k natural length bins with alternating reasoning modes','complete rendered conversations as distractors',
        'UUID-disjoint 80/20 pool split before packing'],splits=manifests))
    print(json.dumps(manifests,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=int,default=100000)
    p.add_argument('--train-count',type=int,default=400);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--raw-dir',default='/home/tiger/datasets/kv_lingo/Nemotron-SFT-Instruction-Following-Chat-v2/'+REVISION)
    p.add_argument('--output',default=str(ROOT/'data/kv_lingo/training_paper_v7'))
    main(p.parse_args())
