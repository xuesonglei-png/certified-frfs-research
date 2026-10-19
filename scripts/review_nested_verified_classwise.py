"""Use the current same-fold recent baselines, not superseded archival scores."""
import json
from review_box_iterations import ROOT,paired

def main():
    folder=ROOT/'results/nested-verified-classwise'
    new=json.loads((folder/'complete.json').read_text());common=json.loads((ROOT/'results/common-fold-recent/complete.json').read_text())
    reference={r['dataset']:r for r in common['rows']};rows=[]
    for row in new['rows']:
        name=row['dataset'];controls=reference[name]['methods']
        rows.append({'dataset':name,'clean':row['clean'],'removed':row['removed'],'full90':row['full'],
            'samefold_controls':{m:controls[m]['clean'] for m in ('full','fgs2023_native','sfrip2025_native')},
            'selected_fallbacks':row['selected_fallbacks']})
    result={'rows':rows,'mean_clean':new['mean_clean'],'mean_removed':new['mean_removed'],
        'vs_samefold_native':{m:paired([r['clean']-r['samefold_controls'][m] for r in rows]) for m in rows[0]['samefold_controls']},
        'vs_matched_full90':paired([r['clean']-r['full90'] for r in rows]),
        'scope':'complete_inner90_pipeline_samefold_native_baselines_repeatedly_exposed_development_not_confirmation'}
    (folder/'complete-common-fold-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
