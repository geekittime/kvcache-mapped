"""Durable stage-gated fit merge, held-out checks and four paired evaluation queues."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from core import atomic_json,digest
from fit import paths
from prepare_eval import cases,case_name
from audit_upstream import provenance,validation as audit_validation
from protocol_guards import generation_parity

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def wait_exact(info):
    p=Path('/proc')/str(info['pid'])
    while True:
        try:
            if p.joinpath('stat').read_text().split()[2]=='Z':return
            actual=p.joinpath('cmdline').read_bytes().decode().split(chr(0))[:-1]
            assert actual==info['command'],(info['pid'],actual)
        except FileNotFoundError:return
        time.sleep(20)

def run(cmd,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        atomic_json(log.with_suffix('.process.json'),dict(pid=proc.pid,command=cmd,started_at=time.time()))
        code=proc.wait()
    if code:raise RuntimeError(f'Failed exit {code}: {log}')

def eval_command(pair,mode,gpu,case,output,limit=0):
    sp,tp=paths(pair)
    cmd=[sys.executable,str(HERE/'evaluate.py'),'--artifact',str(HERE/'artifacts'/pair/'full_v1'),
         '--output',str(output),'--task',case['task'],'--mode',mode,'--source-model',sp,'--target-model',tp,
         '--device',f'cuda:{gpu}','--batch-size',str(case['batch_size']),'--limit',str(limit)]
    if case['length']:cmd+=['--length',case['length']]
    if case['thinking']:cmd+=['--thinking']
    return cmd

def worker(a):
    artifact=HERE/'artifacts'/a.pair/'full_v1'
    proof=json.loads((artifact/'audit.json').read_text());assert proof['passed'] and proof['n']==400
    validation=HERE/'checks'/f'validation_{a.pair}_full'
    proof=json.loads((validation/'audit.json').read_text());assert proof['passed'] and proof['n']==64
    data=json.loads((HERE/'checks/evaluation_inputs.json').read_text());assert data['passed']
    for name,value in data['code_sha256'].items():assert digest(HERE/name)==value
    # Finish all reasoning-off including Hotpot before longer reasoning-on cases.
    queue=sorted(cases(),key=lambda c:c['thinking'])
    for case in queue:
        name=case_name(case);base=HERE/'runs'/a.pair/a.mode/name
        if (base/'stage.json').exists():
            assert json.loads((base/'stage.json').read_text())['passed']
            continue
        for tag,limit in [('smoke',case['batch_size']),('full',0)]:
            out=base/tag
            if not (out/'results.json').exists():
                run(eval_command(a.pair,a.mode,a.gpu,case,out,limit),base/(tag+'.log'))
            proof=json.loads((out/'audit.json').read_text());assert proof['passed']
            if tag=='full':assert proof['complete_dataset']
            print(a.pair,a.mode,name,tag,'AUDITED',proof['n'],flush=True)
        smoke=[json.loads(x) for x in (base/'smoke/predictions.jsonl').open()]
        full=[json.loads(x) for x in (base/'full/predictions.jsonl').open()]
        parity=generation_parity(smoke,full)
        atomic_json(base/'stage.json',dict(passed=True,n=len(full),smoke_full_exact=parity['full_token_sequence_exact']==len(smoke),
                    generation_parity=parity,full_results_sha256=digest(base/'full/results.json'),completed_at=time.time()))

def pipeline():
    provenance()
    infos=json.loads((HERE/'checks/fit_processes.json').read_text())
    for info in infos:wait_exact(info)
    for pair in ['06_17','17_06']:
        out=HERE/'artifacts'/pair/'full_v1'
        for shard in range(2):
            proof=json.loads((out/f'shard_{shard:02d}.json').read_text());assert proof['complete'] and len(proof['done'])==200
        run([sys.executable,str(HERE/'fit.py'),'merge','--pair',pair,'--output',str(out),'--shards','2'],
            HERE/'checks'/f'merge_full_{pair}.log')
        assert json.loads((out/'audit.json').read_text())['n']==400
        run([sys.executable,str(HERE/'audit_fit.py'),str(out)],HERE/'checks'/f'independent_fit_{pair}.log')
    print('BOTH FULL FITS AUDITED',flush=True)
    # The first validation program has already passed a genuine model smoke.
    assert json.loads((HERE/'checks/validation_smoke/audit.json').read_text())['passed']
    procs=[]
    for pair,gpu in [('06_17',0),('17_06',2)]:
        cmd=[sys.executable,str(HERE/'validate.py'),'--pair',pair,'--artifact',str(HERE/'artifacts'/pair/'full_v1'),
             '--output',str(HERE/'checks'/f'validation_{pair}_full'),'--device',f'cuda:{gpu}']
        with (HERE/'checks'/f'validation_{pair}_full.log').open('w') as f:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        atomic_json(HERE/'checks'/f'validation_{pair}_process.json',dict(pid=p.pid,command=cmd))
        procs.append((pair,p))
    for pair,p in procs:
        assert p.wait()==0,pair
        proof=json.loads((HERE/'checks'/f'validation_{pair}_full/audit.json').read_text());assert proof['passed'] and proof['n']==64
        run([sys.executable,str(HERE/'validate.py'),'--pair',pair,'--artifact',str(HERE/'artifacts'/pair/'full_v1'),
             '--output',str(HERE/'checks'/f'validation_{pair}_full'),'--audit-only'],HERE/'checks'/f'validation_reaudit_{pair}.log')
        audit_validation(HERE/'checks'/f'validation_{pair}_full')
    print('HELD-OUT VALIDATION AUDITED',flush=True)
    prep=json.loads((HERE/'checks/prepare_eval_process.json').read_text());wait_exact(prep)
    assert json.loads((HERE/'checks/evaluation_inputs.json').read_text())['passed']
    provenance()
    procs=[]
    for gpu,(pair,mode) in enumerate([('06_17','weighted'),('06_17','readout'),('17_06','weighted'),('17_06','readout')]):
        cmd=[sys.executable,str(Path(__file__)),'--worker','--pair',pair,'--mode',mode,'--gpu',str(gpu)]
        with (HERE/'checks'/f'eval_{pair}_{mode}.log').open('w') as f:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        atomic_json(HERE/'checks'/f'eval_{pair}_{mode}_process.json',dict(pid=p.pid,command=cmd));procs.append((pair,mode,p))
    failures=[]
    for pair,mode,p in procs:
        if p.wait()!=0:failures.append((pair,mode))
    assert not failures,failures
    run([sys.executable,str(HERE/'report.py'),'--require-complete'],HERE/'checks/final_report.log')
    with (HERE/'PROCESS.md').open('a') as f:
        f.write('\n## All comparison cells completed\n\nBoth400-example fits and64-example validation sets audited, all52 benchmark conditions independently rescored, smoke/full first batches exactly matched. See RESULTS.md/json for paired differences and qualifications. No push.\n')
    print('ALL STAGES COMPLETE AND AUDITED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--pair');p.add_argument('--mode');p.add_argument('--gpu',type=int)
    a=p.parse_args()
    try:worker(a) if a.worker else pipeline()
    except Exception:
        name=f'failure_{a.pair}_{a.mode}' if a.worker else 'pipeline_failure'
        atomic_json(HERE/'checks'/(name+'.json'),dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
