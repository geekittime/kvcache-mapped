"""Reconstruct every selected prefix/continuation from pinned raw conversations."""
import hashlib,json,random,re
from pathlib import Path
from transformers import AutoTokenizer
from core import atomic_json,digest
from fit import rows

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'data/kv_lingo/training_paper_v7'

def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    selected={name:rows(DATA/(name+'.jsonl')) for name in ['train','validation']}
    members={name:{u for r in rr for u in r['member_uuids']} for name,rr in selected.items()}
    assert not members['train']&members['validation']
    wanted=members['train']|members['validation'];raw={}
    for entry in manifest['raw_manifest']['files'].values():
        assert digest(entry['path'])==entry['sha256']
        for line in Path(entry['path']).open():
            record=json.loads(line)
            if record['uuid'] in wanted:raw[record['uuid']]=record
    assert set(raw)==wanted
    tok=AutoTokenizer.from_pretrained('/home/tiger/models/Qwen3-1.7B',local_files_only=True)
    encode=lambda s:tok.encode(s,add_special_tokens=False)
    strip=lambda s:re.sub(r'<think>.*?</think>\s*','',s,flags=re.S)
    reconstructed={}
    for uuid,record in raw.items():
        messages=record['messages']
        candidates=[i for i,m in enumerate(messages) if i>0 and m['role']=='assistant' and messages[i-1]['role']=='user' and isinstance(m.get('content'),str)]
        seed=int(hashlib.sha256((str(manifest['seed'])+':'+uuid).encode()).hexdigest()[:16],16)
        turn=random.Random(seed).choice(candidates);answer=messages[turn]
        content=answer['content'];trace=answer.get('reasoning_content') or ''
        if trace and '<think>' not in content:content='<think>\n'+trace+'\n</think>\n\n'+content
        reasoning='<think>' in content and bool(re.search(r'<think>\s*\S',content))
        history=[dict(role=m['role'],content=strip(m.get('content') or '')) for m in messages[:turn]]
        prefix=tok.apply_chat_template(history,tokenize=False,add_generation_prompt=True,enable_thinking=reasoning)
        filler=tok.apply_chat_template(history+[dict(role='assistant',content=strip(answer['content']))],tokenize=False,add_generation_prompt=False)
        continuation=(content if reasoning else strip(content))+'<|im_end|>'
        reconstructed[uuid]=dict(turn=turn,x=encode(prefix),y=encode(continuation)[:2048],filler=encode(filler),reasoning='on' if reasoning else 'off')
    for name,rr in selected.items():
        assert digest(DATA/(name+'.jsonl'))==manifest['splits'][name]['sha256']
        for r in rr:
            p=reconstructed[r['uuid']];uuids=r['member_uuids']
            assert uuids[0]==r['uuid'] and len(uuids)==len(set(uuids))
            x=[t for u in uuids[1:] for t in reconstructed[u]['filler']]+p['x']
            assert x==r['prefix_ids'] and p['y']==r['continuation_ids'],(name,r['id'])
            assert p['turn']==r['assistant_turn'] and p['reasoning']==r['reasoning']
            assert r['kind']=='packed' or len(uuids)==1
    proof=dict(passed=True,train=len(selected['train']),validation=len(selected['validation']),raw_members=len(raw),
        exact_prefix_and_continuation_token_equality=True,raw_hashes_verified=True,member_split_disjoint=True,
        target_answer_excluded_from_own_fillers=True,auditor_sha256=digest(__file__),
        limitation='Verifies the frozen local construction; does not recover unpublished author sample IDs or packing code.')
    atomic_json(HERE/'checks/training_source_reconstruction.json',proof);print(json.dumps(proof))

if __name__=='__main__':main()
