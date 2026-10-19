"""Same-fuzzy recent controls replace mixed historical raw-score summaries."""
import json
from review_box_iterations import ROOT,paired

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',default='nested-triangular')
    folder=ROOT/'results'/parser.parse_args().folder;source=json.loads((folder/'complete.json').read_text())
    control=json.loads((ROOT/'results/common-fuzzy-classifier/complete-fuzzy-review.json').read_text());byname={r['dataset']:r for r in control['rows']};rows=[]
    for row in source['rows']:
        name=row['dataset'];methods=byname[name]['methods']
        rows.append({'dataset':name,'clean':row['clean'],'full':row['full'],'removed':row['removed'],
            'controls':{m:methods[m]['clean'] for m in ('full','fgs2023_native','sfrip2025_native')},'selected_fallbacks':row['selected_fallbacks']})
    result={'rows':rows,'mean_clean':source['mean_clean'],'mean_removed':source['mean_removed'],
        'vs_matched_full':paired([r['clean']-r['full'] for r in rows]),
        'vs_same_fuzzy':{m:paired([r['clean']-r['controls'][m] for r in rows]) for m in rows[0]['controls']},
        'scope':'complete_inner_fuzzy_group_pipeline_same_fuzzy_recent_native_controls_exposed_development_not_confirmation',
        'source_folder':folder.name}
    (folder/'complete-fuzzy-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'})
