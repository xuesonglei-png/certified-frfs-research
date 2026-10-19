"""Report realized size/coverage drift of selected fold-local recipes."""
import json
import numpy as np
from run_decision_iteration import ROOT


def main():
    directory=ROOT/'results/predictive-iteration26';rows=[]
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());row=next(r for r in obj['rows'] if r['method']=='protected')
        chosen=row['chosen'];folds=chosen['folds'];d=obj['d']
        sizes=[len(f['features']) for f in folds]
        final=len(row['features']);target=max(1,int(np.ceil(chosen['config']['fraction']*d)))
        counts=row['certificate']['protected_counts']
        rows.append({'dataset':obj['dataset'],'fold':obj['fold'],
            'target_size':target,'inner_sizes':sizes,'final_size':final,
            'final_minus_mean_inner_size':final-float(np.mean(sizes)),
            'final_target_size_gap':final-target,
            'zero_protected_classes':sum(c==0 for c in counts),
            'protected_counts':counts,'full_fallback':final==d,
            'mean_pairwise_inner_feature_jaccard':float(np.mean([
                len(set(a['features'])&set(b['features']))/len(set(a['features'])|set(b['features']))
                for i,a in enumerate(folds) for b in folds[i+1:]]))})
    summary={'models':len(rows),'full_fallbacks':sum(r['full_fallback'] for r in rows),
        'models_with_zero_protected_class':sum(r['zero_protected_classes']>0 for r in rows),
        'mean_final_minus_inner_size':float(np.mean([r['final_minus_mean_inner_size'] for r in rows])),
        'mean_abs_target_size_gap':float(np.mean([abs(r['final_target_size_gap']) for r in rows])),
        'mean_inner_feature_jaccard':float(np.mean([r['mean_pairwise_inner_feature_jaccard'] for r in rows]))}
    (directory/'recipe-transfer-diagnostic.json').write_text(json.dumps({'summary':summary,'rows':rows,
        'scope':'posthoc_development_mechanism_diagnostic_not_test_tuning'},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
