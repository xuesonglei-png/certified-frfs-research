"""Full15 frozen development comparison, nine corrected primary comparisons."""
import json
from run_decision_iteration import ROOT
from analyze_crossfit_majority_expansion import analyze

EXPECTED={'banknote','ecoli','glass','haberman','heart_statlog','ionosphere','iris','seeds',
          'sonar','transfusion','vehicle','vertebral','wdbc','wine','yeast'}


def main():
    directory=ROOT/'results/predictive-iteration36';files=sorted(directory.glob('*-s101-f*.json'))
    assert len(files)==45,'full study incomplete'
    grouped={}
    for file in files:
        obj=json.loads(file.read_text())
        for r in obj['rows']:
            grouped.setdefault(r['method'],{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']})
        for iteration,alias in [(33,'majority_only33'),(31,'refreshed_witness31')]:
            old=json.loads((ROOT/f'results/predictive-iteration{iteration}'/file.name).read_text())
            r=next(r for r in old['rows'] if r['method']=='protected')
            grouped.setdefault(alias,{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']})
    assert set(grouped['protected'])==EXPECTED
    assert all(len(rows)==3 for data in grouped.values() for rows in data.values())
    result={'status':'frozen_full15_repeated_development_not_confirmation',
        'all15':analyze(grouped,[]),'without_ecoli':analyze(grouped,['ecoli']),
        'noisy_secondary':analyze(grouped,[],metric='noise01')}
    a=result['all15'];means={m:r['means'] for m,r in a['methods'].items()}
    pair=next(p for p in a['pairs'] if p['comparator']=='fgs')
    result['prespecified_primary_gate']={'fgs_mean_gain_at_least_half_point':pair['mean_delta']>=.005,
        'fgs_corrected_wilcoxon_significant':pair['holm_wilcoxon_p']<.05,
        'no_mean_deterioration_against_full':means['protected']['clean']>=means['all']['clean']}
    result['gate_passed']=all(result['prespecified_primary_gate'].values())
    (directory/'full-crossfit-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'means':means,'gates':result['prespecified_primary_gate'],'pairs':a['pairs']},indent=2))

if __name__=='__main__':main()
