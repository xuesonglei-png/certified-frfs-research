import json
import numpy as np
from diagnose_linear_retention import FOLDER

def main():
    p=json.loads((FOLDER/'complete.json').read_text());rows=[]
    for r in p['rows']:
        data=json.loads((FOLDER/f"{r['dataset']}.json").read_text());splits=data['splits']
        rows.append({'dataset':r['dataset'],'projected_clean':float(np.mean([s['metrics']['clean'] for s in splits])),
                     'refit_clean':float(np.mean([s['refit_metrics']['clean'] for s in splits])),
                     'full_supports':sum(len(s['selected']['features'])==data['d'] for s in splits),
                     'protected_counts':[next(z['linear_certificate']['protected_count'] for z in s['candidates'] if z['candidate']==s['selected']) for s in splits]})
    comparisons=p['comparisons'];result={'rows':rows,'mean_refit_clean':float(np.mean([r['refit_clean'] for r in rows])),
        'full_supports':sum(r['full_supports'] for r in rows),'gate_passed':p['mean_removed']>=.1 and comparisons['full']['gain']>=-.005 and all(comparisons[m]['gain']>=.005 and comparisons[m]['wins']>=10 for m in ('fgs2023','sfrip2025')),
        'scope':'fixed_teacher_projection_refit_ablation_not_refit_certification'}
    (FOLDER/'complete-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
