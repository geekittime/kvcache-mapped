#!/usr/bin/env python3
"""Audited, resumable Stage-1 greedy generation; left-padded batches."""
from __future__ import annotations
import argparse, collections, json, math, re, string, fcntl
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from core import ROOT, load_model, Trace, project_cache, atomic_json, digest


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def check_resume_protocol(previous, current):
    """Prevent silently combining predictions from different implementations."""
    for key in previous.keys() | current.keys():
        if key == 'batch_size':
            continue
        if previous.get(key) != current.get(key):
            raise ValueError(f'incompatible resume: {key}; use a new output directory')


def load_weights(root, mode, device, source=None, target=None):
    root=Path(root); run=json.loads((root/'run.json').read_text())
    proof=json.loads((root/'audit.json').read_text())
    assert proof['passed'] and proof['K_bitwise_equal'] and proof['run_sha256']==digest(root/'run.json')
    for key,value in [('source',source),('target',target)]:
        if value is not None:
            assert Path(run['metadata'][key]).resolve()==Path(value).resolve(), 'mapper model mismatch'
    z={}
    for p in sorted((root/mode).glob('layer_*.npz')):
        assert digest(p)==run['artifact_sha256'][str(p.relative_to(root))],p
        _,layer,side=p.stem.split('_')
        v=np.load(p)['weight']; assert np.isfinite(v).all() and v.shape==(1024,1024) and v.dtype==np.float32
        z[f'{side}.{int(layer)}']=torch.tensor(v,device=device)
    mapping=run['metadata']['source_layer_map']
    assert set(z)=={f'{s}.{i}' for i in range(len(mapping)) for s in ['k','v']}
    return z,mapping


def answer_body(text):
    if '</think>' in text: return text.rsplit('</think>',1)[-1].strip()
    if '<think>' in text: return ''  # unfinished reasoning is not a final answer
    return text.strip()


def parse_answer(text, n_options=10):
    t=answer_body(text)
    letters='ABCDEFGHIJ'[:n_options]
    # Explicit verdicts beat discussion. Never infer from a word's first letter,
    # or from the English article "a" in free prose.
    patterns=[r'\\boxed\s*\{\s*\(?([A-J])\)?\s*\}',
              r'\b(?:the\s+)?(?:correct\s+answer|final\s+answer|answer)\s*(?:is\s*:?\s*|[:：]\s*)(?:\*\*|\*|`)?\s*(?:option\s+)?\(?([A-J])\)?(?![A-Za-z])',
              r'^\s*(?:\*\*|`)?\(?([A-J])\)?(?:[.)]|(?:\*\*|`))?\s*$']
    for pat in patterns:
        found=re.findall(pat,t,flags=re.I|re.M)
        if found: return found[-1].upper() if found[-1].upper() in letters else ''
    # A single distinct standalone option line is explicit even if it also
    # includes option text. Multiple alternatives without a verdict are rejected.
    found=re.findall(r'^\s*(?:\*\*|`)?\(([A-J])\)\s',t,re.M)
    if len(set(found))==1:return found[0] if found[0] in letters else ''
    if len(set(found))>1:return ''
    # A leading uppercase option followed by explanation is also explicit.
    m=re.match(r'^\s*\(?([A-J])\)?[.)](?:\s|$)',t)
    return m.group(1) if m and m.group(1) in letters else ''


def norm_text(s):
    s=s.lower(); s=''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(w for w in s.split() if w not in {'a','an','the'})


def token_f1(pred,gold):
    a,b=norm_text(pred),norm_text(gold)
    # Official HotpotQA answer F1 special cases.
    if (a in {'yes','no','noanswer'} or b in {'yes','no','noanswer'}) and a!=b:return 0.0
    p,g=a.split(),b.split()
    if not p or not g:return 0.0
    common=sum((collections.Counter(p)&collections.Counter(g)).values())
    return 2*common/(len(p)+len(g)) if common else 0.0


def extract_hotpot_answer(text):
    """Prediction-only extraction; never accepts reference/type/question input."""
    t=answer_body(text)
    boxes=re.findall(r'\\boxed\s*\{([^{}]+)\}',t)
    if boxes:t=boxes[-1].strip()
    else:
        labelled=re.findall(r'(?:^|\n)\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*(.+)',t,re.I)
        if labelled:t=labelled[-1]
    t=t.strip().strip('*`').strip()
    if not t:return ''
    t=t.split('\n')[0].strip().strip('*`').strip()
    m=re.match(r'^(yes|no|noanswer)(?:[,.!\s]|$)',t,re.I)
    if m:return m.group(1).lower()
    return t.strip('"').strip()


def mc_item(row,task):
    letters='ABCDEFGHIJ'
    if task=='mmlu':
        opts=row['options']; gold=letters[int(row['answer_index'])]
    else:
        ch=row['choices']; opts=ch['text']; labels=list(ch['label'])
        gold=letters[labels.index(str(row['answerKey']))]
    prompt=row['question']+'\n'+''.join(f'({letters[i]}) {v}\n' for i,v in enumerate(opts))+'\nAnswer with only the letter of the correct option.'
    return prompt,gold,len(opts)


def chat_ids(tok,user,thinking=False):
    return tok.apply_chat_template([{'role':'user','content':user}],tokenize=True,
                    add_generation_prompt=True,enable_thinking=thinking)


@torch.inference_mode()
def generate_batch(src,tgt,sequences,weights,max_new,pad_id,source_layer_map=None):
    assert max_new>0 and sequences and all(len(s)>=2 for s in sequences)
    dev=next(tgt.parameters()).device; B=len(sequences); T=max(map(len,sequences))
    ids=torch.full((B,T),pad_id,device=dev,dtype=torch.long); mask=torch.zeros_like(ids)
    for i,s in enumerate(sequences): ids[i,-len(s):]=torch.tensor(s,device=dev);mask[i,-len(s):]=1
    pos=(mask.cumsum(-1)-1).clamp_min(0)
    if weights is None:
        out=tgt(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=True,logits_to_keep=1)
    else:
        sdev=next(src.parameters()).device
        with Trace(src,T-1) as tr:
            src.model(input_ids=ids[:,:-1].to(sdev),attention_mask=mask[:,:-1].to(sdev),
                      position_ids=pos[:,:-1].to(sdev),use_cache=False)
        cache=project_cache(tr.k,tr.v,tgt,weights,position_ids=pos[:,:-1],source_layer_map=source_layer_map)
        out=tgt(input_ids=ids[:,-1:],past_key_values=cache,attention_mask=mask,
                position_ids=pos[:,-1:],cache_position=torch.tensor([T-1],device=dev),
                use_cache=True,logits_to_keep=1)
        del tr,cache
    eos_ids=tgt.generation_config.eos_token_id or tgt.config.eos_token_id
    eos_ids=eos_ids if isinstance(eos_ids,list) else [eos_ids]
    eos=torch.tensor(eos_ids,device=dev); active=torch.ones(B,device=dev,dtype=torch.bool)
    tokens=[]
    for step in range(max_new):
        nxt=out.logits[:,-1].argmax(-1)
        is_eos=(nxt[:,None]==eos[None,:]).any(-1)
        saved=torch.where(active,nxt,torch.full_like(nxt,-1));tokens.append(saved)
        active=active & ~is_eos
        if not active.any() or step+1==max_new:break
        # Finished rows are kept in the batch but excluded from outputs. Their
        # later states cannot influence any other row's attention.
        nxt=torch.where(active,nxt,torch.full_like(nxt,pad_id))
        mask=torch.cat([mask,torch.ones((B,1),device=dev,dtype=mask.dtype)],dim=1)
        cur=out.past_key_values.get_seq_length()
        out=tgt(input_ids=nxt[:,None],past_key_values=out.past_key_values,attention_mask=mask,
                position_ids=(pos[:,-1:]+step+1),cache_position=torch.tensor([cur],device=dev),
                use_cache=True,logits_to_keep=1)
    matrix=torch.stack(tokens,1).cpu().tolist(); result=[]
    for row in matrix:
        vals=[]; stop='length';finish=None
        for token in row:
            if token<0:break
            if token in eos_ids: stop='eos';finish=token;break
            vals.append(token)
        result.append({'token_ids':vals,'generated_tokens':len(vals),'stop_reason':stop,'eos_token_id':finish})
    return result


def score_prediction(row,task,text):
    if task=='ruler':
        body=re.sub(r'[\x00-\x1f]', '\n', answer_body(text).strip()).strip().lower()
        refs=row['outputs'];assert refs
        return {'gold':refs,'pred':body,'recall':sum(str(r).lower() in body for r in refs)/len(refs)}
    if task in ('mmlu','arc'):
        _,gold,n=mc_item(row,task); pred=parse_answer(text,n)
        return {'gold':gold,'pred':pred,'correct':int(pred==gold),'category':row.get('category','all')}
    gold=row['answer']; pred=extract_hotpot_answer(text)
    return {'gold':gold,'pred':pred,'em':int(norm_text(pred)==norm_text(gold)),
            'f1':token_f1(pred,gold),'raw_em':int(norm_text(text)==norm_text(gold)),
            'raw_f1':token_f1(text,gold),'type':row.get('type'),'level':row.get('level')}


def summarize(preds,task):
    n=len(preds); result={'n':n,'length_stops':sum(r['stop_reason']=='length' for r in preds),
              'unparsed':sum(not r['pred'] for r in preds),'mean_generated_tokens':sum(r['generated_tokens'] for r in preds)/n}
    if task in ('mmlu','arc'):
        groups=collections.defaultdict(list)
        for r in preds:groups[r['category']].append(r['correct'])
        cats={k:sum(v)/len(v) for k,v in groups.items()}
        result.update(accuracy=sum(r['correct'] for r in preds)/n,
                      macro_accuracy=sum(cats.values())/len(cats),by_category=cats)
    elif task=='ruler':
        result['recall']=sum(r['recall'] for r in preds)/n
    else:
        for key in ['em','f1','raw_em','raw_f1']:result[key]=sum(r[key] for r in preds)/n
    return result


def dataset(task,length=None):
    if task=='ruler':
        data=Path('/home/tiger/datasets/RULER_generated')/length/'niah_single_2/validation.jsonl'
        return data,read_jsonl(data)[:400]
    files={'mmlu':'mmlu_pro.jsonl','arc':'arc_c.jsonl','hotpot':'hotpotqa_distractor_validation.jsonl'}
    data=ROOT/'data/kv_lingo/benchmarks'/files[task]
    return data,read_jsonl(data)


def user_prompt(row,task):
    if task=='ruler':return row['input']+row.get('answer_prefix','')
    return row['prompt'] if task=='hotpot' else mc_item(row,task)[0]


def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    data,rows=dataset(a.task,a.length)
    if a.limit:rows=rows[:a.limit]
    indexed=[(i,r) for i,r in enumerate(rows) if i%a.shards==a.shard]
    protocol={**vars(a),'data':str(data),'data_sha256':digest(data),'code_sha256':{p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('core.py')]},
              'scoring':'generation, prediction-only extraction; official Hotpot answer normalization/F1',
              'max_new':a.max_new or (256 if a.task=='hotpot' else 2048 if a.task=='ruler' else 8192 if a.thinking else 1024)}
    if a.mode!='native':protocol['mapper_run_sha256']=digest(Path(a.artifact)/'run.json')
    protocol['model_config_sha256']=digest(Path(a.target_model)/'config.json')
    protocol['source_config_sha256']=digest(Path(a.source_model)/'config.json')
    protocol['tokenizer_config_sha256']=digest(Path(a.target_model)/'tokenizer_config.json')
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        old_protocol=json.loads(protocol_path.read_text())
        # Batch size is a scheduling parameter; changing it preserves the
        # sample protocol and allows a safe resumable speed adjustment.
        check_resume_protocol(old_protocol, protocol)
        protocol=old_protocol
    else:atomic_json(protocol_path,protocol)
    path=out/'predictions.jsonl'
    lock_file=(out/'predictions.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    done=read_jsonl(path) if path.exists() else []
    assert len({r['index'] for r in done})==len(done)
    completed={r['index'] for r in done}; expected={i for i,_ in indexed}; assert completed<=expected
    todo=[(i,r) for i,r in indexed if i not in completed]
    if todo:
        tgt=load_model(a.target_model,a.device);src=None;w=None;mapping=None
        if a.mode!='native':
            src=load_model(a.source_model,a.device)
            w,mapping=load_weights(a.artifact,a.mode,a.device,a.source_model,a.target_model)
        tok=AutoTokenizer.from_pretrained(a.target_model,local_files_only=True)
        prepared=[]
        for i,r in todo:
            user=user_prompt(r,a.task)
            ids=chat_ids(tok,user,a.thinking)
            assert len(ids)+protocol['max_new']<=tgt.config.max_position_embeddings
            if a.task in ('mmlu','arc'):assert len(ids)<=3000
            prepared.append((i,r,ids))
        # Stable original order keeps smoke/full batches identical.
        with path.open('a',buffering=1) as f:
            for j in range(0,len(prepared),a.batch_size):
                batch=prepared[j:j+a.batch_size]
                generated=generate_batch(src,tgt,[r[2] for r in batch],w,protocol['max_new'],tok.pad_token_id,mapping)
                for (i,r,ids),g in zip(batch,generated):
                    text=tok.decode(g['token_ids'],skip_special_tokens=True)
                    pred={'index':i,'id':str(r.get('id',r.get('question_id',i))),
                          'prompt_tokens':len(ids),'prompt_ids_sha256':__import__('hashlib').sha256(json.dumps(ids).encode()).hexdigest(),
                          'text':text,**g,**score_prediction(r,a.task,text)}
                    f.write(json.dumps(pred,ensure_ascii=False,allow_nan=False)+'\n');done.append(pred)
                print(json.dumps({'task':a.task,'mode':a.mode,'done':len(done),'total':len(indexed),**summarize(done,a.task)}),flush=True)
    assert {r['index'] for r in done}==expected
    done.sort(key=lambda r:r['index']); res=summarize(done,a.task)
    atomic_json(out/'results.json',{'complete':True,'protocol':protocol,'predictions_sha256':digest(path),**res})
    print(json.dumps(res),flush=True)
    fcntl.flock(lock_file, fcntl.LOCK_UN); lock_file.close()
    from audit_eval import audit
    audit(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact');p.add_argument('--output',required=True)
    p.add_argument('--task',choices=['mmlu','arc','hotpot','ruler'],required=True);p.add_argument('--mode',choices=['weighted','readout'],required=True)
    p.add_argument('--length',choices=['4k','8k','16k','32k'])
    p.add_argument('--source-model');p.add_argument('--target-model',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-new',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--thinking',action='store_true');main(p.parse_args())
