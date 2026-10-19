"""Small-file live observation; never load or retain all large fold artifacts."""
import json
import time
from run_simplex_soft5_knn_pilot import ROOT,digest

STUDIES={
    'monotone-frs-inner':('monotone-frs-inner-chain-status.json','monotone-frs-inner-review.json','monotone-frs-inner-source-freeze.json'),
    'latent-frs-training':('latent-frs-training-chain-status.json','latent-FRS-training-review.json','latent-frs-training-source-freeze.json'),
    'pair-transition-frs-training':('pair-transition-frs-training-chain-status.json','pair-transition-FRS-training-review.json','pair-transition-frs-training-source-freeze.json'),
    'frs-coordinate-diagnostic-v2':('frs-coordinate-diagnostic-v2-chain-status.json','FRS-coordinate-minimum-review-v2.json','frs-coordinate-diagnostic-v2-source-freeze.json'),
    'frs-zero-face-repair':('frs-zero-face-repair-chain-status.json','FRS-zero-face-repair-review.json','frs-zero-face-repair-source-freeze.json'),
    'frs-joint-zero-face':('frs-joint-zero-face-chain-status.json','FRS-joint-zero-face-review.json','frs-joint-zero-face-source-freeze.json'),
    'frs-joint-zero-face-prediction':('frs-joint-zero-face-prediction-chain-status.json','FRS-joint-zero-face-prediction-review.json','frs-joint-zero-face-prediction-source-freeze.json'),
    'sparse-global-frs-training':('sparse-global-frs-training-chain-status.json','sparse-global-FRS-training-review.json','sparse-global-frs-training-source-freeze.json'),
    'pair-transition-frs-inner':('pair-transition-frs-inner-chain-status.json','pair-transition-FRS-inner-review.json','pair-transition-frs-inner-source-freeze.json'),
    'majority-hinge-canonical-benchmark':('majority-hinge-canonical-benchmark-chain-status.json',None,None),
}


def observe():
    report={'observed_unix':time.time(),'goal_achieved':False,'states':{},
        'limits':['Raw summaries remain provisional until full audit and review.',
                  'Repeatedly exposed development data cannot confirm superiority.',
                  'Source-only hash check is not a replay of large scientific result artifacts.']}
    cache={}
    for name,(status_name,review_name,freeze_name) in STUDIES.items():
        status_path=ROOT/'results'/status_name
        if not status_path.exists():
            prep=ROOT/'results'/(name+'-preparation-status.json')
            state=json.loads(prep.read_text()) if prep.exists() else {'phase':'not_started'}
        else:
            state=json.loads(status_path.read_text())
        folder=ROOT/'results'/name;summary=folder/'complete.json';audit=folder/'complete-audit.json'
        if summary.exists():
            raw=json.loads(summary.read_text())
            state['raw_summary_sha256']=digest(summary)
            state['raw_dataset_count']=raw.get('datasets')
            state['raw_summary_metrics']={k:raw[k] for k in ('mean_vs_previous_DC_pp',
                'mean_proposed_minus_controls_pp','dataset_balanced_metrics','optimizer_failures',
                'target_optimizer_failures','pilot_gate_passed','counts','means',
                'pair_clean_BA_differences_pp','mean_actual_feature_counts') if k in raw}
        if audit.exists():
            a=json.loads(audit.read_text());state['complete_audit_passed']=a['passed']
            state['summary_matches_full_audit']=summary.exists() and a['summary_sha256']==digest(summary)
            state['audit_sha256']=digest(audit)
        if review_name and (ROOT/'research'/review_name).exists():
            review=ROOT/'research'/review_name;r=json.loads(review.read_text())
            state['review_sha256']=digest(review);state['review_dataset_count']=len(r.get('all_dataset_rows',[]))
            state['review_matches_full_audit']=audit.exists() and r['audit_sha256']==digest(audit)
        if freeze_name and (ROOT/'research'/freeze_name).exists():
            freeze=ROOT/'research'/freeze_name;data=json.loads(freeze.read_text());changed=[];n=0
            for rel,sha in data['fingerprints'].items():
                if not rel.endswith('.py'):
                    continue
                n+=1
                if rel not in cache:
                    cache[rel]=digest(ROOT/rel)
                if cache[rel]!=sha:
                    changed.append(rel)
            state['frozen_python_sources_checked']=n;state['changed_frozen_python_sources']=changed
        report['states'][name]=state
    return report


if __name__=='__main__':
    data=observe()
    (ROOT/'research/latest-live-experiment-observation-v2.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data,indent=2),flush=True)
