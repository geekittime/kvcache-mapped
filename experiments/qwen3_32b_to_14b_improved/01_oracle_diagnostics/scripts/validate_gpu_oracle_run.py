#!/usr/bin/env python3
"""CPU-only integrity validator for a completed A1 v2 GPU oracle run."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gpu_oracle import CONFIG,validate_run
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--profile",choices=("smoke","screen"),required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 print(json.dumps(validate_run(a.config,profile=a.profile,output_dir=a.output_dir),ensure_ascii=False,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
