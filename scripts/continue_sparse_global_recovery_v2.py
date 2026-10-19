"""Complete matched ALL15 recovery, preserving old failed sources.

Only the sparse-arm solver changed. Other arms are reused at the exact
original training splits and independently audited with their original
proof checks before being included. All three sparse starts are refit.
"""
import ctypes
import hashlib
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from fit_sparse_global_transition_frs_v2 import fit_sparse_global_transition_target
from sparse_global_frs_training_ablation import evaluation,STARTS
from audit_sparse_global_training_block_v2 import audit_block
from run_sparse_global_frs_training import prerequisites,DIAGNOSTIC

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/sparse-global-frs-training'
OUT=ROOT/'results/sparse-global-frs-training-recovery-v2'
FREEZE=ROOT/'research/sparse-global-frs-training-recovery-v2-source-freeze.json'
STATUS=ROOT/'results/sparse-global-frs-training-recovery-v2-chain-status.json'


class MemoryStatus(ctypes.Structure):
    _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),
        ('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),
        ('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),
        ('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),('ullAvailExtendedVirtual',ctypes.c_ulonglong)]


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def status(phase,**fields):STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**fields},indent=2),encoding='utf-8')


def wait_memory(name):
    # Parsing the largest legacy JSON records needs substantial headroom.
    required=max(3*1024**3,8*(OLD/(name+'.json')).stat().st_size)
    while True:
        m=MemoryStatus();m.dwLength=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        if m.ullAvailPhys>=required:return
        status('waiting_memory',dataset=name,required_bytes=required,available_bytes=m.ullAvailPhys)
        time.sleep(5)


def run(name,frozen):
    output=OUT/(name+'.json');audit_path=OUT/(name+'.audit.json')
    if output.exists() and audit_path.exists():
        a=json.loads(audit_path.read_text());assert a['passed'] and a['sha256']==digest(output)
        return a
    wait_memory(name);status('refitting_sparse_arm',dataset=name)
    source_path=OLD/(name+'.json');old=json.loads(source_path.read_text());source_hash=digest(source_path)
    source=json.loads((ROOT/'results/capped-frs-training'/(name+'.json')).read_text())
    diagnostic=json.loads((DIAGNOSTIC/(name+'.json')).read_text())
    block=old['block'];counts=Counter(datasets=1)
    if block is not None:
        X,y=load_dataset(name);ri,ca=np.asarray(old['reference_indices']),np.asarray(old['training_indices'])
        A,_=preprocess_fit(X[ri][ca],X[ri][ca]);labels=y[ri][ca]
        arm=block['arms']['global_signed_sparse']
        if arm['available']:
            fits={}
            for start in STARTS:
                timer=time.perf_counter()
                if name=='ecoli':
                    # Already refit/audited with identical source and settings.
                    recovery=ROOT/'results/sparse-global-v2-ecoli-recovery'
                    evidence=json.loads((recovery/'complete-audit.json').read_text())
                    path=recovery/(start+'.json');assert evidence['passed'] and evidence['source_sha256']==source_hash
                    assert next(r['sha256'] for r in evidence['rows'] if r['start']==start)==digest(path)
                    r=json.loads(path.read_text());fit,target=r['fit'],np.asarray(r['target']);timer=time.perf_counter()-r['fit_seconds']
                else:
                    target,fit=fit_sparse_global_transition_target(A,labels,block['features'],block['anchor'],
                        full_reference=np.ones(A.shape[1])/np.sqrt(A.shape[1]),reference_scope='original_full',
                        initialization=np.asarray(block['starts'][start]),initialization_scope=start)
                fits[start]={'features':block['features'],'anchor':block['anchor'],'target':target.tolist(),
                    'fit':fit,'fit_seconds':time.perf_counter()-timer}
            chosen=min(STARTS,key=lambda s:(fits[s]['fit']['training_objective_final'],STARTS.index(s)))
            block['arms']['global_signed_sparse']={**arm,'fits':fits,'selected_start':chosen,
                'target':fits[chosen]['target'],'optimizer_success':all(r['fit']['optimizer_success'] for r in fits.values())}
        block['evaluation']=evaluation(A,labels,block)
        status('independent_dataset_audit',dataset=name)
        audit_block(A,labels,block,diagnostic['diagnostic'],counts)
    packet={**old,'block':block,'legacy_fingerprints':old['fingerprints'],'fingerprints':frozen,
        'source_sha256':source_hash,'scope':'versioned_sparse_arm_dual_guard_recovery_other_arms_independently_audited',
        'classification_scored':False,'goal_achieved':False}
    output.write_text(json.dumps(packet,separators=(',',':')),encoding='utf-8')
    row={'dataset':name,'available':block is not None,'evaluation':None if block is None else block['evaluation']}
    report={'passed':True,'row':row,'counts':dict(counts),'sha256':digest(output),'source_sha256':source_hash}
    audit_path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(name,'recovery independent audit PASS',flush=True)
    return report


def main():
    threadpool_limits(1);OUT.mkdir(exist_ok=True);audits=prerequisites()
    names=sorted(audits['capped-frs-training']['artifact_hashes']);names=['ecoli']+[n for n in names if n!='ecoli']
    files=sorted((ROOT/'scripts').glob('*.py'))+sorted((ROOT/'src/certified_frfs').glob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    if not FREEZE.exists():FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':hashes,'datasets':names},indent=2),encoding='utf-8')
    frozen=json.loads(FREEZE.read_text())['fingerprints']
    for rel,sha in frozen.items():assert digest(ROOT/rel)==sha,rel
    reports=[run(name,frozen) for name in names]
    counts=Counter()
    for r in reports:counts.update(r['counts'])
    summary={'datasets':len(reports),'rows':[r['row'] for r in reports],'counts':dict(counts),
        'classification_scored':False,'goal_achieved':False,'independent_audit_passed':True,
        'source_failed_study_preserved':True,'scope':'matched_three_arm_three_start_training_recovery_not_test_performance'}
    path=OUT/'complete.json';path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    audit={'passed':True,'counts':dict(counts),'summary_sha256':digest(path),
        'artifact_hashes':{r['row']['dataset']:r['sha256'] for r in reports},'goal_achieved':False}
    (OUT/'complete-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    status('complete_audited_training_only',audit_sha256=digest(OUT/'complete-audit.json'))


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise
