"""Compact read-only monitoring; no partial accuracy reporting."""
import json
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
done=[];active=[]
for pair in ['06_17','17_06']:
    for mode in ['weighted','readout']:
        for p in sorted((HERE/'runs'/pair/mode).glob('*')):
            if (p/'stage.json').exists():done.append(str(p.relative_to(HERE/'runs')));continue
            f=p/'full/predictions.jsonl'
            n=sum(1 for _ in f.open()) if f.exists() else 'smoke'
            active.append(f'{pair}/{mode}/{p.name}: {n}')
failures=[p.name for p in (HERE/'checks').glob('*failure*.json')]
print(datetime.now().isoformat(timespec='seconds'),f'complete={len(done)}/52',f'failures={failures}')
print('\n'.join(active))
