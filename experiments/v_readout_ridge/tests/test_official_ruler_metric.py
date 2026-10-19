"""Check local NIAH aggregation against the downloaded upstream metric."""
import importlib.util
from pathlib import Path
import pytest
from evaluate import score_prediction

def test_niah_matches_official_recall_after_documented_reasoning_removal():
    path=Path('/home/tiger/datasets/RULER/scripts/eval/synthetic/constants.py')
    if not path.exists():pytest.skip('downloaded upstream RULER required')
    spec=importlib.util.spec_from_file_location('official_ruler_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    outputs=[['ABC','XYZ'],['brown'],['12345'],['needle']]
    text=['abc only','BROWN','wrong','<think>needle</think>absent']
    scores=[score_prediction({'outputs':ref},'ruler',prediction) for ref,prediction in zip(outputs,text)]
    official=module.TASKS['niah']['metric_fn']([x['pred'] for x in scores],outputs)
    assert round(sum(x['recall'] for x in scores)/len(scores)*100,2)==official==37.5
