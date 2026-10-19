"""Dataset-unit comparisons including recent graph and matched ablation."""
import argparse,json
from run_decision_iteration import ROOT
from analyze_crossfit_majority_expansion import analyze


def main(iteration=31):
    grouped={};directory=ROOT/f'results/predictive-iteration{iteration}'
    paths=sorted(directory.glob('*-s101-f*.json'));assert len(paths)==45
    for path in paths:
        obj=json.loads(path.read_text())
        for row in obj['rows']:
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append({**row['metrics'],'removed':1-len(row['features'])/obj['d']})
        fgs=json.loads((ROOT/'results/fgs-nested-comparison'/path.name).read_text())
        extra=[('fgs',fgs)]
        if iteration==31:
            ablation=json.loads((ROOT/'results/matched-uncertified-ablation'/path.name).read_text())
            extra.append(('uncertified',ablation))
        for method,row in extra:
            grouped.setdefault(method,{}).setdefault(obj['dataset'],[]).append({**row['metrics'],'removed':1-len(row['features'])/obj['d']})
    result={'status':'repeated_development_unequal_budgets','comparisons':len(grouped)-1,
        'all15':analyze(grouped,[]),'without_ecoli':analyze(grouped,['ecoli'])}
    (directory/'expanded-baseline-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for scope,analysis in result.items():
        if scope in ('status','comparisons'):continue
        print(json.dumps({'scope':scope,'means':{m:r['means'] for m,r in analysis['methods'].items()},'pairs':analysis['pairs']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[31,32],default=31)
    main(p.parse_args().iteration)
