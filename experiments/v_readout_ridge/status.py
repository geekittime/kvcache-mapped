"""Read-only live progress; never interpret partial scores as completed results."""
import json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def last_progress(path):
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-65536));lines=f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            obj=json.loads(line)
            if isinstance(obj,dict) and 'done' in obj:return {k:obj[k] for k in ['done','total','seconds'] if k in obj}
        except ValueError:pass
    return None

def main():
    result=dict(time=time.time(),fit=[],validation=[],evaluation=[],failures=[])
    for pair in ['06_17','17_06']:
        for shard in [0,1]:
            complete=(HERE/'artifacts'/pair/'full_v1'/f'shard_{shard:02d}.json').exists()
            result['fit'].append(dict(pair=pair,shard=shard,complete=complete,
                progress=last_progress(HERE/'checks'/f'fit_{pair}_shard{shard}.log')))
        val=HERE/'checks'/f'validation_{pair}_full'
        result['validation'].append(dict(pair=pair,audited=(val/'audit.json').exists(),
                                        progress=last_progress(HERE/'checks'/f'validation_{pair}_full.log')))
        for mode in ['weighted','readout']:
            for path in sorted((HERE/'runs'/pair/mode).glob('*')):
                result['evaluation'].append(dict(pair=pair,mode=mode,case=path.name,
                    stage_audited=(path/'stage.json').exists(),full_progress=last_progress(path/'full.log'),
                    smoke_progress=last_progress(path/'smoke.log')))
    for p in list((HERE/'checks').glob('failure_*.json'))+list((HERE/'checks').glob('*_failure.json')):
        result['failures'].append(dict(path=str(p),details=json.loads(p.read_text())))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
