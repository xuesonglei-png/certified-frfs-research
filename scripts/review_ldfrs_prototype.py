"""Verify independent-variant pilot and matched historical baseline scores."""
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT,scaled_split,scores
from certified_frfs.datasets import load_dataset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--convention',default='legacy')
    args=parser.parse_args()
    suffix='' if args.convention=='legacy' else '-'+args.convention
    threadpool_limits(1);directory=ROOT/('results/ldfrs-independent-prototype'+suffix)
    pilot=json.loads((directory/'pilot.json').read_text());rows=[];comparators={}
    for row in pilot['rows']:
        X,y=load_dataset(row['dataset']);tr=np.array(row['train_indices']);te=np.array(row['test_indices'])
        A,B=scaled_split(X[tr],X[te])
        assert scores(A,y[tr],B,y[te],row['features'],104,k=5)==row['metrics']
        expected=max(row['inner_candidates'],key=lambda c:(np.mean(c['values']),-c['fraction']))
        assert expected['fraction']==row['chosen_fraction']
        old=json.loads((ROOT/'results/predictive-iteration15'/f"{row['dataset']}-s101-f0.json").read_text())
        assert old['train_indices']==row['train_indices'] and old['test_indices']==row['test_indices']
        for other in old['rows']:
            key=other['method'];comparators.setdefault(key,[]).append(other['metrics']['0.0']['balanced_accuracy'])
        rows.append({'dataset':row['dataset'],'clean':row['metrics']['0.0']['balanced_accuracy'],
                     'removed':1-len(row['features'])/row['d']})
    summary={'status':'independent_variant_not_official_reproduction_no_significance_claim',
        'verified_outer_models':len(rows),'pilot_clean':float(np.mean([r['clean'] for r in rows])),
        'pilot_removed':float(np.mean([r['removed'] for r in rows])),
        'historical_means':{key:float(np.mean(v)) for key,v in comparators.items()},'rows':rows,
        'limitations':'same folds/classifier; fixed density parameters vs tuned comparators; only independent kernel convention'}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
