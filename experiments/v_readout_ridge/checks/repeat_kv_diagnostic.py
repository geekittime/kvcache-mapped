import fcntl,time,json
from pathlib import Path
import torch
from core import atomic_json
from transformers.integrations.sdpa_attention import repeat_kv
h=Path(__file__).resolve().parent
with (h/'cuda_0.schedule.lock').open('a') as lock:
 print('waiting for GPU0 shared lease',flush=True)
 fcntl.flock(lock,fcntl.LOCK_SH)
 torch.use_deterministic_algorithms(True)
 torch.manual_seed(42)
 def cat_heads(x):
  b,h,n,d=x.shape
  return torch.cat((x[:,:,None],x[:,:,None]),dim=2).reshape(b,h*2,n,d)
 rows=[]
 for n in [512,4096,8192]:
  x=torch.randn(32,8,n,128,device='cuda:0',dtype=torch.bfloat16)
  a=repeat_kv(x,2);b=cat_heads(x)
  assert torch.equal(a,b) and a.stride()==b.stride()
  row=dict(n=n,exact=True,stride=list(a.stride()))
  del a,b
  for name,fn in [('expand_reshape',lambda z:repeat_kv(z,2)),('cat_heads',cat_heads),('repeat_interleave',lambda z:z.repeat_interleave(2,dim=1))]:
   for _ in range(3): y=fn(x)
   torch.cuda.synchronize();start=time.perf_counter()
   for _ in range(20):y=fn(x)
   torch.cuda.synchronize();row[name]=(time.perf_counter()-start)/20
   del y
  rows.append(row);del x;torch.cuda.empty_cache()
 atomic_json(h/'repeat_kv_diagnostic.json',dict(synthetic_only=True,production_changed=False,rows=rows))
 print(json.dumps(rows),flush=True)
