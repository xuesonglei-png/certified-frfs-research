"""Development diagnostic without treating validation tuning as test gains."""
import json
import numpy as np
from pathlib import Path
from run_density_calibrated_soft5_diagnostic import ROOT,FOLDER,METHODS,digest
from density_calibrated_soft5_stage import GRID
from continue_soft_topk_knn_pilot import verify


def main():
    verify(FOLDER);summary=json.loads((FOLDER/'complete.json').read_text());rows=summary['rows']
    gains={m:100*(summary['means'][m]['selected_inner']-summary['means'][m]['anchor']) for m in METHODS}
    per_dataset=[100*(r['selected_inner_BA']['proposed']-r['anchor_BA']['proposed']) for r in rows]
    fixed=100*(np.array(summary['means']['proposed']['fixed_grid'])-summary['means']['proposed']['anchor'])
    # A mechanism screen only. It does not choose a grid setting for deployment.
    screen=bool(gains['proposed']>=.3 and sum(x>1e-10 for x in per_dataset)>=5 and summary['target_optimizer_failures']==0)
    output={'shared_arm_inner_gain_pp':gains,'proposed_per_dataset_inner_gain_pp':dict(zip([r['dataset'] for r in rows],per_dataset)),
        'proposed_fixed_recipe_inner_gain_pp':[{'family':f,'interpolation':t,'gain_pp':float(g)} for (f,t),g in zip(GRID,fixed,strict=True)],
        'inner_mechanism_screen_passed':screen,'target_optimizer_failures':summary['target_optimizer_failures'],
        'new_outer_evaluation_performed':False,'goal_achieved':False,'audit_sha256':digest(FOLDER/'complete-audit.json'),
        'limitations':['Anchor included, selected inner nonworsening is tautological.',
            'Frozen anchors were selected by previous exposed inner validation; this is a diagnostic, not fresh nested learning.',
            'Full-fold minmax geometry is retained; no fully cross-fitted preprocessing claim.',
            'Crossgroup may benefit controls equally; no proposed superiority or population risk is established.'],
        'next_step':'If mechanism is promising, rebuild complete nested selector/learner/count controls for every arm; otherwise do not promote this branch.'}
    (ROOT/'research/density-calibrated-soft5-diagnostic-review.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
