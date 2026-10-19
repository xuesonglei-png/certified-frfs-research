"""Posthoc finite-query robust prediction lower bounds, not nominal accuracy."""
import hashlib,json
from run_decision_iteration import ROOT
from analyze_crossfit_majority_expansion import analyze


def main():
    directory=ROOT/'results/majority-query-audit33';grouped={};files=sorted(directory.glob('*-s101-f*.json'));assert len(files)==45
    for file in files:
        obj=json.loads(file.read_text());source=ROOT/'results/predictive-iteration33'/file.name
        assert obj['input_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        for r in obj['rows']:
            grouped.setdefault(r['method'],{}).setdefault(obj['dataset'],[]).append({
                'certified_correct':r['balanced_certified_correct'],
                'certified_wrong':r['balanced_certified_wrong'],'joint_mean_BA':r['joint_mean_BA']})
    result={'status':'posthoc_repeated_development_finite_query_lower_bound_not_confirmatory',
        'paired_metric':'certified_correct','query_radius':.01,'models':360,
        'all15':analyze(grouped,[],metric='certified_correct'),
        'without_ecoli':analyze(grouped,['ecoli'],metric='certified_correct')}
    (directory/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'means':{m:r['means'] for m,r in result['all15']['methods'].items()},'pairs':result['all15']['pairs']},indent=2))

if __name__=='__main__':main()
