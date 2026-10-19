import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate import parse_answer,score_prediction,token_f1,check_resume_protocol,answer_body

@pytest.mark.parametrize('text,gold',[
 ('<think>I considered A.</think>\nThe answer is (B).','B'),
 ('<think>Answer: C',''),('The final answer is J.','J'),
 ('A reasonable explanation is needed.',''),('Answer: (D)','D')])
def test_mc_extraction(text,gold):assert parse_answer(text)==gold

def test_ruler_does_not_score_answer_seen_only_in_reasoning():
    row={'outputs':['91823']}
    assert score_prediction(row,'ruler','<think>91823</think>Unknown')['recall']==0
    assert score_prediction(row,'ruler','<think>Unknown</think>91823')['recall']==1

def test_hotpot_yes_no_rules():
    assert token_f1('yes','no')==0
    assert token_f1('yes','yes')==1
    assert token_f1('The Eiffel Tower','Eiffel Tower')==1

def test_resume_rejects_semantic_or_code_changes():
    with pytest.raises(ValueError):check_resume_protocol({'code_sha256':'a'},{'code_sha256':'b'})
    with pytest.raises(ValueError):check_resume_protocol({'thinking':False},{'thinking':True})
