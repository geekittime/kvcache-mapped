"""Format-only MC parser correction; frozen generated tokens are never changed."""
import argparse,json,re
from pathlib import Path
from core import atomic_json,digest
from evaluate import answer_body,parse_answer,dataset,mc_item,read_jsonl,summarize

def parse_v2(text,n_options=10):
    body=answer_body(text)
    pattern=r'\\(?:text|mathrm|mathbf|textbf|textrm|mathit)\s*\{\s*(\(?[A-J]\)?)\s*\}'
    body=re.sub(pattern,r'\1',body)
    body=body.replace('**','').replace(chr(96),'')
    return parse_answer(body,n_options)

def rescore(out):
    out=Path(out);original=json.loads((out/'results.json').read_text())
    task=original['protocol']['task'];assert task in ['arc','mmlu']
    _,data=dataset(task,original['protocol']['length'])
    records=read_jsonl(out/'predictions.jsonl');scored=[];changes=[]
    for row in records:
        ref=data[row['index']];_,gold,n=mc_item(ref,task);assert row['gold']==gold
        pred=parse_v2(row['text'],n);value={**row,'pred':pred,'correct':int(pred==gold)}
        scored.append(value)
        if pred!=row['pred']:changes.append(dict(index=row['index'],old=row['pred'],new=pred,
            old_correct=row['correct'],new_correct=value['correct']))
    result={**original,**summarize(scored,task)}
    result['scoring_revision']='v2: LaTeX single-option wrappers and Markdown presentation normalization'
    result['scoring_v1']={k:original[k] for k in ['accuracy','macro_accuracy','unparsed','by_category']}
    proof=dict(passed=True,n=len(records),results_sha256=digest(out/'results.json'),
        predictions_sha256=digest(out/'predictions.jsonl'),scorer_sha256=digest(__file__),
        scoring_v1=result['scoring_v1'],scoring_v2=summarize(scored,task),changes=changes,
        no_generation_changes=True,reference_independent_parser=True)
    atomic_json(out/'mc_scoring_v2.json',proof)
    return result,scored

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
    r,_=rescore(a.output);print(json.dumps(dict(n=r['n'],v1=r['scoring_v1'],v2=r['macro_accuracy'])))
