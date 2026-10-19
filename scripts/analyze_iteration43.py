"""Complete45task, dataset-paired evaluation under frozen43protocol."""
import json,hashlib
import numpy as np
from scipy.stats import wilcoxon
from run_query_support_iteration43 import ROOT,DATASETS


def main():
    folder=ROOT/'results/predictive-iteration43';rows=[];audits=[]
    for name in DATASETS:
        folds=[]
        for f in range(3):
            path=folder/f'{name}-s101-f{f}.json';p=json.loads(path.read_text());a=json.loads((folder/'audits'/path.name).read_text())
            assert a['result_sha256']==hashlib.sha256(path.read_bytes()).hexdigest();audits.append(a)
            folds.append({'metrics':{r['method']:r['metrics'] for r in p['rows']+p['baseline_rows']},'sizes':{r['method']:len(r['fit']['features']) for r in p['rows']},'d':p['d'],'fallback':next(r['full_fallback'] for r in p['rows'] if r['method']=='query_certified')})
        methods=list(folds[0]['metrics']);rows.append({'dataset':name,'folds':folds,'metrics':{m:{k:float(np.mean([q['metrics'][m][k] for q in folds])) for k in ['clean','noise01']} for m in methods},'removed_fraction':{m:float(np.mean([1-q['sizes'][m]/q['d'] for q in folds])) for m in ['query_certified','query_unconstrained']}})
    tests=[];rng=np.random.default_rng(1020);boot=rng.integers(0,15,(10000,15))
    for baseline in ['fgs','all','query_unconstrained']:
        delta=np.array([r['metrics']['query_certified']['clean']-r['metrics'][baseline]['clean'] for r in rows]);rounded=np.round(delta,12)
        p1=1. if np.all(rounded==0) else float(wilcoxon(rounded,alternative='greater').pvalue)
        p2=1. if np.all(rounded==0) else float(wilcoxon(rounded,alternative='two-sided').pvalue)
        tests.append({'baseline':baseline,'mean_gain':float(delta.mean()),'raw_one_sided_p':p1,'raw_two_sided_p':p2,'interval95':np.quantile(delta[boot].mean(axis=1),[.025,.975]).tolist(),'wins':int(sum(delta>1e-12)),'ties':int(sum(abs(delta)<=1e-12)),'losses':int(sum(delta< -1e-12))})
    order=np.argsort([r['raw_one_sided_p'] for r in tests]);last=0.
    for rank,index in enumerate(order):last=max(last,min(1.,(3-rank)*tests[index]['raw_one_sided_p']));tests[index]['holm_p']=last
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in ['clean','noise01']} for m in methods}
    payload={'complete':True,'rows':rows,'means':means,'tests':tests,'gate_passed':all(t['mean_gain']>=.005 and t['holm_p']<.05 for t in tests[:2]),'audited_inner_scores':sum(a['inner_scores_checked'] for a in audits),'audited_outer_scores':sum(a['outer_scores_checked'] for a in audits),'sampled_loss_checks':sum(a['sampled_loss_checks'] for a in audits),'full_fallbacks':sum(f['fallback'] for r in rows for f in r['folds']),'mean_removed_fraction':{m:float(np.mean([r['removed_fraction'][m] for r in rows])) for m in ['query_certified','query_unconstrained']},'scope':'repeated_development_not_untouched_confirmation_or_journal_readiness'}
    out=folder/'complete-analysis.json';assert not out.exists();out.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(means),json.dumps(tests),flush=True)


if __name__=='__main__':main()
