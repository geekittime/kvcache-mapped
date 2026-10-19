import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol_guards import validate_case,matching_pair,generation_parity

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

def test_greedy_tail_drift_is_recorded_but_input_and_first_token_are_required():
    a=dict(index=0,id='0',prompt_tokens=4,prompt_ids_sha256='same',gold='A',
           token_ids=[8,9,10],pred='A')
    b=dict(index=0,id='0',prompt_tokens=4,prompt_ids_sha256='same',gold='A',
           token_ids=[8,9,11],pred='A')
    result=generation_parity([a],[b])
    assert result['first_token_exact']==1 and result['full_token_sequence_exact']==0
    assert result['parsed_answer_exact']==1 and result['first_divergence_token']==[2]
    b['prompt_ids_sha256']='different'
    with pytest.raises(AssertionError):generation_parity([a],[b])
    b['prompt_ids_sha256']='same';b['token_ids'][0]=7
    with pytest.raises(AssertionError,match='First-token'):generation_parity([a],[b])
