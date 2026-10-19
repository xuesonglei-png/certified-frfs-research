"""Summarize frozen query audits and verify iteration25 baseline equality."""
import json
import numpy as np
from run_decision_iteration import ROOT


def main():
    grouped={};matched=0;split_checks=0
    for path in sorted((ROOT/'results/predictive-iteration25').glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());old=json.loads((ROOT/'results/predictive-iteration24'/path.name).read_text())
        assert obj['train_indices']==old['train_indices'] and obj['test_indices']==old['test_indices']
        assert obj['discovery_indices']==old['discovery_indices'] and obj['validation_indices']==old['validation_indices']
        split_checks+=1
        for row in obj['rows']:
            if row['method']=='protected':continue
            previous=next(r for r in old['rows'] if r['method']==row['method'])
            for key in ['chosen','trace','features','weights','restoration','metrics']:
                assert row[key]==previous[key]
            matched+=1
    for path in sorted((ROOT/'results/majority-query-audit25').glob('*-s101-f*.json')):
        obj=json.loads(path.read_text())
        for row in obj['rows']:
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append(row)
    keys=['balanced_certified_correct','balanced_certified_wrong','balanced_original_certified_correct','joint_mean_BA']
    summary={}
    for method,data in grouped.items():
        datasets={name:{key:float(np.mean([r[key] for r in rows])) for key in keys} for name,rows in data.items()}
        summary[method]={'datasets':datasets,'means':{key:float(np.mean([r[key] for r in datasets.values()])) for key in keys}}
    result={'matched_baseline_records':matched,'verified_splits':split_checks,
        'methods':summary,'status':'posthoc_sampled_diagnostic_not_confirmation'}
    out=ROOT/'results/majority-query-audit25/summary.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'matched_baselines':matched,'splits':split_checks,
        'means':{key:value['means'] for key,value in summary.items()}},indent=2))

if __name__=='__main__':main()
