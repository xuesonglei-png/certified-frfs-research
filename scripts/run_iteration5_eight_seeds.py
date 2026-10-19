"""Copy into the prepared bundle, then run unchanged engine across eight seeds."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parent
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
sys.path.insert(0,str(ROOT))


def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def main():
    import engine
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--output-name',default='run-001')
    ap.add_argument('--resume',action='store_true',help='Resume only missing splits with identical frozen inputs.')
    args=ap.parse_args()
    if args.workers<1 or not args.output_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.output_name):ap.error('invalid arguments')
    engine.verify_inputs()
    splits=json.loads((ROOT/'splits.json').read_text())
    out=ROOT/'runs'/args.output_name
    signature=hashlib.sha256((ROOT/'frozen-inputs-sha256.json').read_bytes()).hexdigest()
    if args.resume:
        config=json.loads((out/'run-config.json').read_text())
        if config['manifest_sha256']!=signature:raise ValueError('Manifest changed')
        if (out/'completion.json').exists():raise ValueError('Run already completed; preserve outputs')
    else:
        out.mkdir(parents=True,exist_ok=False)
        engine.dump_new(out/'run-config.json',{'started_unix':time.time(),'workers':args.workers,
                          'manifest_sha256':signature,'python':sys.version,'expected_splits':len(splits)})
    pending=[];completed=[]
    for s in splits:
        p=out/f"{s['dataset']}-s{s['seed']}-f{s['fold']}.json"
        if p.exists():
            obj=json.loads(p.read_text())
            if obj['train_indices']!=s['train_indices'] or obj['test_indices']!=s['test_indices'] or len(obj['rows'])!=8:raise ValueError('Invalid checkpoint: '+p.name)
            completed.append(p.name)
        else:pending.append(s)
    failures=[]
    with (out/'progress.jsonl').open('a',encoding='utf-8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(engine.run,(s,args.output_name)):s for s in pending}
            for future in as_completed(futures):
                s=futures[future]
                try:
                    report=future.result();report['seed']=s['seed']
                    completed.append(f"{s['dataset']}-s{s['seed']}-f{s['fold']}.json")
                except Exception as e:
                    report={'dataset':s['dataset'],'seed':s['seed'],'fold':s['fold'],'status':'failed','error':repr(e)}
                    failures.append(report)
                report.update({'finished_unix':time.time(),'completed':len(completed),'expected':len(splits)})
                log.write(json.dumps(report)+'\n');log.flush()
                atomic_json(out/'progress.json',report)
                print(json.dumps(report),flush=True)
    engine.verify_inputs()
    engine.dump_new(out/'completion.json',{'finished_unix':time.time(),'completed':len(completed),
                       'expected':len(splits),'failures':failures,'input_integrity_passed':True})
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
