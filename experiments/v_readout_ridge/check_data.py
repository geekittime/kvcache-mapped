#!/usr/bin/env python3
"""Fail closed on corpus, template, length, UUID and benchmark invariants."""
import argparse,collections,json
from pathlib import Path
from transformers import AutoTokenizer
from core import atomic_json,digest
from evaluate import read_jsonl,mc_item

def main(a):
 root=Path(a.data);manifest=json.loads((root/'manifest.json').read_text())
 assert manifest['dataset']=='nvidia/Nemotron-SFT-Instruction-Following-Chat-v2'
 t=AutoTokenizer.from_pretrained('/home/tiger/models/Qwen3-1.7B',local_files_only=True)
 s=AutoTokenizer.from_pretrained('/home/tiger/models/Qwen3-0.6B',local_files_only=True)
 assert t.get_vocab()==s.get_vocab() and t.chat_template==s.chat_template
 special={i for i in [t.eos_token_id,t.pad_token_id] if i is not None}
 summary={};members={}
 for name,total in [('train',400),('validation',64)]:
  path=root/(name+'.jsonl');rows=read_jsonl(path)
  assert len(rows)==total and len({r['id'] for r in rows})==total and len({r['uuid'] for r in rows})==total
  assert digest(path)==manifest['splits'][name]['sha256']
  kinds=collections.Counter(r['kind'] for r in rows)
  assert kinds=={'natural':total//2,'packed':total//2}
  for kind in kinds:
   modes=collections.Counter(r['reasoning'] for r in rows if r['kind']==kind)
   assert modes=={'on':total//4,'off':total//4}
  for i,r in enumerate(rows):
   assert r['id']==i and r['reasoning']==('off' if i%2==0 else 'on')
   assert r['prefix_tokens']==len(r['prefix_ids']) and r['continuation_tokens']==len(r['continuation_ids'])
   assert 1<=r['continuation_tokens']<=2048
   assert (10<=r['prefix_tokens']<=10000) if r['kind']=='natural' else (12000<=r['prefix_tokens']<=16000)
   assert r['member_uuids'][0]==r['uuid'] and r['member_uuids'].count(r['uuid'])==1
   assert len(r['member_uuids'])==len(set(r['member_uuids']))
   tail=t.decode(r['prefix_ids'][-25:])
   if r['reasoning']=='on':
    assert tail.endswith('<|im_start|>assistant\n')
    assert '<think>' in t.decode(r['continuation_ids'][:15])
   else:assert tail.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
  bins=[sum(r['kind']=='natural' and min(r['prefix_tokens']//1000,9)==i for r in rows) for i in range(10)]
  if name=='train':assert bins==[20]*10,bins
  else:assert min(bins)>=2 and max(bins)<=4,bins
  members[name]={u for r in rows for u in r['member_uuids']}
  summary[name]={'n':total,'prefix_tokens':sum(r['prefix_tokens'] for r in rows),'natural_bins':bins,
                 'member_uuids':len(members[name]),'sha256':digest(path)}
 assert not members['train']&members['validation']
 bench=Path('data/kv_lingo/benchmarks')
 for kind,count in [('mmlu_pro',1400),('arc_c',1172),('hotpotqa_distractor_validation',7405)]:
  rs=read_jsonl(bench/(kind+'.jsonl')); assert len(rs)==count
  if kind=='mmlu_pro':assert set(collections.Counter(r['category'] for r in rs).values())=={100}
  if kind=='arc_c':
   for r in rs:mc_item(r,'arc')
  summary[kind]={'n':len(rs),'sha256':digest(bench/(kind+'.jsonl'))}
 summary.update(passed=True,tokenizers_identical=True,member_overlap=0,raw_manifest=manifest['raw_manifest'])
 atomic_json(a.output,summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);main(p.parse_args())
