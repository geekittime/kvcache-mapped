import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol_guards import validate_case,matching_pair

def fixture():
    case=dict(task='mmlu',length=None,thinking=True,max_new=8192,batch_size=32,n=1400,
              data='/home/tiger/fixed-inputs.jsonl',data_sha256='prepared-hash')
    protocol=dict(case,mode='weighted',shards=1,shard=0,limit=0)
    return case,protocol

@pytest.mark.parametrize('key,value',[
 ('max_new',1024),('thinking',False),('batch_size',16),('shards',2),('shard',1),
 ('limit',100),('data_sha256','new-unprepared-inputs'),('mode','native')])
def test_formal_case_rejects_silent_drift(key,value):
    case,p=fixture();p[key]=value
    with pytest.raises(AssertionError):validate_case(p,case,1400)

def test_full_count_and_smoke_contract():
    case,p=fixture();validate_case(p,case,1400)
    with pytest.raises(AssertionError):validate_case(p,case,100)
    p['limit']=32;validate_case(p,case,32)
    with pytest.raises(AssertionError):validate_case(p,case,16)

def test_paired_budget_mismatch_rejected():
    _,a=fixture();b=copy.deepcopy(a);b.update(mode='readout',device='cuda:1',output='/home/tiger/second')
    a.update(device='cuda:0',output='/home/tiger/first');matching_pair(a,b)
    b['max_new']=1024
    with pytest.raises(AssertionError,match='max_new'):matching_pair(a,b)
