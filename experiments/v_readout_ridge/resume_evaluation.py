"""Resume four evaluation workers after the fit/validation gates already passed."""
import json,subprocess,sys,traceback,time
from pathlib import Path
from core import atomic_json
from audit_upstream import provenance
from audit_fit import audit as audit_fit
from audit_upstream import validation

HERE=Path(__file__).resolve().parent
def main():
    provenance()
    for pair in ['06_17','17_06']:
        artifact=HERE/'artifacts'/pair/'full_v1'
        assert audit_fit(artifact,deep=False)['n']==400
        assert validation(HERE/'checks'/f'validation_{pair}_full')['n']==64
    procs=[]
    for gpu,(pair,mode) in enumerate([('06_17','weighted'),('06_17','readout'),('17_06','weighted'),('17_06','readout')]):
        for lane in [0,1]:
            cmd=[sys.executable,str(HERE/'run_pipeline.py'),'--worker','--pair',pair,'--mode',mode,'--gpu',str(gpu),
                 '--lane',str(lane),'--lanes','2']
            log=(HERE/'checks'/f'eval_{pair}_{mode}_lane{lane}.log').open('w')
            p=subprocess.Popen(cmd,cwd=HERE.parents[1],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(HERE/'checks'/f'eval_{pair}_{mode}_lane{lane}_process.json',dict(pid=p.pid,command=cmd,started_at=time.time(),resume=True))
            procs.append((pair,mode,p,log))
    failures=[]
    for pair,mode,p,log in procs:
        code=p.wait();log.close()
        if code:failures.append((pair,mode,code))
    if failures:raise RuntimeError(f'Worker failures: {failures}')
    cmd=[sys.executable,str(HERE/'report.py'),'--require-complete']
    with (HERE/'checks/final_report.log').open('w') as log:
        code=subprocess.call(cmd,cwd=HERE.parents[1],stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError(f'Final report rejected, exit={code}')
    with (HERE/'PROCESS.md').open('a') as f:
        f.write('\n## Deterministic evaluation completed\n\nAll52 formal cells passed independent auditing and exact smoke/full record parity under deterministic CUDA settings. Two disjoint queues per GPU preserved every case and its batch geometry. See RESULTS.md. Archived nondeterministic outputs are excluded.\n')
    print('ALL EVALUATION CELLS COMPLETED AND AUDITED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(HERE/'checks/evaluation_resume_failure.json',dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
