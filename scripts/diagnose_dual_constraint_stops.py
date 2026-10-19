"""Training-only decomposition of final-path stopping and feasible utility loss."""
import json
import numpy as np
from run_decision_iteration import ROOT


def main():
    directory=ROOT/'results/predictive-iteration36';rows=[]
    for file in sorted(directory.glob('*-s101-f*.json')):
        o=json.loads(file.read_text());r=next(r for r in o['rows'] if r['method']=='protected');p=r['path']
        attempts=[a for step in p['trace'] for a in step['certification_attempts']]
        stopped=p['trace'][-1] if p['trace'] and p['trace'][-1]['accepted'] is None else None
        terminal=stopped['certification_attempts'] if stopped else []
        rows.append({'dataset':o['dataset'],'fold':o['fold'],'d':o['d'],
            'selected_size':len(r['features']),'minimum_feasible_path_size':len(p['candidates'][-1]),
            'protected_counts':p['protected_counts'],'required_counts':p['required'],
            'no_initial_protection_classes':sum(c==0 for c in p['protected_counts']),
            'majority_rejections':sum(not a['majority_feasible'] for a in attempts),
            'dependency_rejections_after_majority_pass':sum(a['majority_feasible'] and not a['feasible'] for a in attempts),
            'terminal_majority_rejections':sum(not a['majority_feasible'] for a in terminal),
            'terminal_dependency_rejections':sum(a['majority_feasible'] and not a['feasible'] for a in terminal)})
    summary={'jobs':len(rows),'selected_full':sum(r['selected_size']==r['d'] for r in rows),
        'no_one_attribute_deletion_feasible':sum(r['minimum_feasible_path_size']==r['d'] for r in rows),
        **{k:sum(r[k] for r in rows) for k in ['majority_rejections','dependency_rejections_after_majority_pass',
            'terminal_majority_rejections','terminal_dependency_rejections','no_initial_protection_classes']}}
    result={'scope':'training_only_sequential_sufficient_bound_rejections_not_proved_infeasibility',
            'summary':summary,'rows':rows}
    (directory/'constraint-stop-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
