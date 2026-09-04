#!/usr/bin/env python3
"""CPU preflight/dry run or explicitly authorized A1 GPU oracle execution."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gpu_oracle import CONFIG,contract_preflight,execute,load_v2
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--profile",choices=("smoke","screen"),required=True);p.add_argument("--output-dir",type=Path);g=p.add_mutually_exclusive_group(required=True);g.add_argument("--preflight-only",action="store_true");g.add_argument("--dry-run",action="store_true");g.add_argument("--execute",action="store_true");a=p.parse_args();cfg=load_v2(a.config);pre,_=contract_preflight(cfg,profile=a.profile)
 if a.preflight_only or a.dry_run: print(json.dumps(pre,ensure_ascii=False,indent=2,sort_keys=True));return 0
 if a.output_dir is None:raise SystemExit("--output-dir is required with --execute")
 return execute(a.config,profile=a.profile,output_dir=a.output_dir)
if __name__=="__main__":raise SystemExit(main())
