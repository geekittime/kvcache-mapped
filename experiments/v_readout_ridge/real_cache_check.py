import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from core import load_model,Trace,project_cache,atomic_json
from evaluate import generate_batch,read_jsonl,mc_item,chat_ids

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path(__file__).parent/'checks/real_cache_check.json')
args=parser.parse_args()
records=[]
for name in ['Qwen3-0.6B','Qwen3-1.7B']:
 m=load_model('/home/tiger/models/'+name,'cuda:0');t=AutoTokenizer.from_pretrained('/home/tiger/models/'+name,local_files_only=True)
 rs=read_jsonl('data/kv_lingo/benchmarks/arc_c.jsonl')[:3]
 seqs=[chat_ids(t,mc_item(r,'arc')[0]) for r in rs]
 weights={f'{s}.{i}':torch.eye(1024,device='cuda:0') for s in ['k','v'] for i in range(28)}
 native=generate_batch(None,m,seqs,None,12,t.pad_token_id)
 replay=generate_batch(m,m,seqs,weights,12,t.pad_token_id,list(range(28)))
 individual=[generate_batch(None,m,[s],None,12,t.pad_token_id)[0] for s in seqs]
 assert native==replay, 'real BF16 identity replay generation mismatch'
 assert native==individual,'real BF16 batched/unbatched generation mismatch'
 ids=torch.tensor([seqs[0]],device='cuda:0')
 with torch.inference_mode():
  full=m(ids,use_cache=True,logits_to_keep=1)
  with Trace(m,ids.shape[1]-1) as tr:m.model(ids[:,:-1],use_cache=False)
  cache=project_cache(tr.k,tr.v,m)
  native_prefix=m.model(ids[:,:-1],use_cache=True).past_key_values
  for layer in range(28):
   for side in range(2):
    torch.testing.assert_close(cache[layer][side],native_prefix[layer][side],rtol=0,atol=0)
  n=ids.shape[1]-1
  got=m(ids[:,-1:],past_key_values=cache,
        attention_mask=torch.ones((1,n+1),device='cuda:0',dtype=torch.long),
        position_ids=torch.tensor([[n]],device='cuda:0'),cache_position=torch.tensor([n],device='cuda:0'),
        use_cache=True,logits_to_keep=1)
  err=float((full.logits-got.logits).abs().max());rms=float((full.logits.float()-got.logits.float()).square().mean().sqrt())
  split=m(ids[:,-1:],past_key_values=native_prefix,attention_mask=torch.ones_like(ids),position_ids=torch.tensor([[n]],device='cuda:0'),cache_position=torch.tensor([n],device='cuda:0'),use_cache=True,logits_to_keep=1)
  torch.testing.assert_close(split.logits,got.logits,atol=0,rtol=0)
  assert full.logits.argmax(-1).equal(got.logits.argmax(-1))
 records.append({'model':name,'reconstructed_vs_native_split_cache_exact':True,'split_logits_exact':True,'batch_identity_exact':True,'batch_vs_single_exact':True,'max_logit_error_bf16':err,'rms_logit_error_bf16':rms,'sequences':len(seqs)})
 del m,weights,full,got,tr,cache
 torch.cuda.empty_cache()
atomic_json(args.output,records)
print(json.dumps(records,indent=2))
