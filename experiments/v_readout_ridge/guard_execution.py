"""Monitor durable evaluation; only resource OOM permits automatic serial retry."""
import json,os,signal,subprocess,sys,time,traceback,shutil
from pathlib import Path
from core import atomic_json

HERE=Path(__file__).resolve().parent
CHECKS=HERE/'checks'

def live(info):
    path=Path('/proc')/str(info['pid'])
    if not path.exists():return False
    state=path.joinpath('stat').read_text().split()[2]
    cmd=path.joinpath('cmdline').read_bytes().decode().split(chr(0))[:-1]
    if state=='Z' or not cmd:return False
    assert cmd==info['command'],('Process identity mismatch',info['pid'])
    return True

def stop_owned(info):
    if not live(info):return
    assert os.getpgid(info['pid'])==info['pid']
    os.killpg(info['pid'],signal.SIGTERM)

def failures():return sorted(CHECKS.glob('failure_*.json'))+list(CHECKS.glob('evaluation_resume_failure.json'))

def is_resource_failure(path):
    trace=json.loads(path.read_text())['traceback']
    if 'OutOfMemoryError' in trace:return True
    marker=': '+str(HERE/'runs')
    if marker not in trace:return False
    value=str(HERE/'runs')+trace.rsplit(marker,1)[1].splitlines()[0]
    log=Path(value.strip())
    assert log.is_relative_to(HERE/'runs')
    return log.exists() and 'torch.OutOfMemoryError:' in log.read_text()

def retry_serial(attempt,issues):
    archive=CHECKS/f'autorecovery_serial_{attempt}';archive.mkdir(exist_ok=False)
    coordinator=json.loads((CHECKS/'resume_evaluation_process.json').read_text())
    workers=[json.loads(p.read_text()) for p in CHECKS.glob('eval_*_lane*_process.json')]
    for info in [coordinator]+workers:stop_owned(info)
    for _ in range(20):
        if not any(live(p) for p in [coordinator]+workers):break
        time.sleep(.5)
    assert not any(live(p) for p in [coordinator]+workers),'Owned process did not stop'
    for p in (HERE/'runs').glob('*/*/*/*.log'):
        target=archive/'logs'/p.relative_to(HERE/'runs');target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    for p in failures():p.rename(archive/p.name)
    (CHECKS/'resume_evaluation_process.json').rename(archive/'coordinator.json')
    atomic_json(archive/'reason.json',dict(oom_failure_paths=[str(p) for p in issues],
        policy='Retry saved complete batches, unchanged geometry/budgets, all GPU calls exclusive',stopped=[coordinator]+workers))
    env=dict(os.environ,KV_LINGO_SERIALIZE_ALL='1')
    cmd=[sys.executable,str(HERE/'resume_evaluation.py')]
    with (CHECKS/'resume_evaluation.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=HERE.parents[1],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(CHECKS/'resume_evaluation_process.json',dict(pid=proc.pid,command=cmd,
        started_at=time.time(),automatic_resource_retry=attempt,scheduling='all calls exclusive per GPU'))

def main():
    attempt=0;last_status=0
    while True:
        now=time.time();issues=failures()
        # Aggregate coordinator error can accompany the underlying worker error.
        specific=[p for p in issues if p.name.startswith('failure_')]
        if issues:
            assert specific and all(is_resource_failure(p) for p in specific), 'Non-resource failure requires independent investigation; automatic continuation refused'
            attempt+=1;assert attempt<=2,'Resource retry limit exceeded'
            retry_serial(attempt,issues);continue
        info=json.loads((CHECKS/'resume_evaluation_process.json').read_text())
        if not live(info):
            results=json.loads((HERE/'RESULTS.json').read_text())
            assert results['complete'] and results['accepted_cells']==results['expected_cells']==52
            atomic_json(CHECKS/'guard_complete.json',dict(passed=True,completed_at=now,
                accepted_cells=52,resource_retries=attempt,coordinator=info))
            return
        if now-last_status>=60:
            status=subprocess.check_output([sys.executable,str(HERE/'brief_status.py')],cwd=HERE.parents[1],text=True)
            atomic_json(CHECKS/'guard_status.json',dict(time=now,coordinator=info,resource_retries=attempt,
                status=status,complete=False))
            last_status=now
        time.sleep(15)

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(CHECKS/'guard_failure.json',dict(time=time.time(),traceback=traceback.format_exc(),
            no_failed_result_accepted=True))
        raise
