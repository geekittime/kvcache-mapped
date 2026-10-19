"""Run frozen evaluation under an explicit reproducible CUDA runtime."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,runpy
from pathlib import Path
import torch
from core import atomic_json,digest
HERE=Path(__file__).resolve().parent

def runtime():
    return dict(cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        seed=42,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        wrapper_sha256=digest(__file__),evaluate_sha256=digest(HERE/'evaluate.py'))

if __name__=='__main__':
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    out=Path(sys.argv[sys.argv.index('--output')+1])
    proof=runtime();receipt=out/'runtime.json'
    if receipt.exists():
        import json
        assert json.loads(receipt.read_text())==proof
    atomic_json(receipt,proof)
    runpy.run_path(str(HERE/'evaluate.py'),run_name='__main__')
