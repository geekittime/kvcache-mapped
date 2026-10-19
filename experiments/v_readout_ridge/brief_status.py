"""Compact read-only monitoring; no partial accuracy reporting."""
import json
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent

def running(receipt):
    if not receipt.exists():return False
    info=json.loads(receipt.read_text())
    try:
        proc=Path('/proc')/str(info['pid'])
        cmd=proc.joinpath('cmdline').read_bytes().decode().split(chr(0))[:-1]
        return cmd==info['command'] and proc.joinpath('stat').read_text().split()[2]!='Z'
    except (FileNotFoundError, ProcessLookupError):
        return False

done=[];active=[]
for pair in ['06_17','17_06']:
    for mode in ['weighted','readout']:
        for p in sorted((HERE/'runs'/pair/mode).glob('*')):
            if (p/'stage.json').exists():
                assert json.loads((p/'stage.json').read_text())['passed']
                done.append(str(p.relative_to(HERE/'runs')));continue
            tag='full' if (p/'smoke/audit.json').exists() else 'smoke'
            live=running(p/f'{tag}.process.json')
            f=p/tag/'predictions.jsonl'
            n=sum(1 for line in f.open() if line.endswith('\n')) if f.exists() else 0
            state='running' if live else 'queued or stopped'
            if (p/tag/'results.json').exists():state='awaiting audit/stage acceptance'
            active.append(f'{pair}/{mode}/{p.name}: {tag} {state}, saved={n}')
failures=[p.name for p in (HERE/'checks').glob('*failure*.json')]
print(datetime.now().isoformat(timespec='seconds'),f'complete={len(done)}/52',f'failures={failures}')
print('\n'.join(active))
