"""Publish only matched-fuzzy comparisons as primary for this diagnostic."""
import json
from review_box_iterations import ROOT

if __name__=='__main__':
    folder=ROOT/'results/realized-fuzzy-budget'
    source=json.loads((folder/'complete.json').read_text())
    result={'means':{m:source['means'][m] for m in ('fgs2023','sfrip2025','proposed_logistic_diagnostic')},
        'comparisons':source['comparisons']['proposed_logistic_diagnostic'],
        'rows':[{'dataset':r['dataset'],'removed':r['removed'],'clean':{m:r['clean'][m] for m in ('fgs2023','sfrip2025','proposed_logistic_diagnostic')}} for r in source['rows']],
        'scope':'same_fuzzy_classifier_exact_raw_count_post_selection_diagnostic_not_native_or_certified_deployed_refit_or_confirmation'}
    (folder/'complete-fuzzy-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='rows'})
