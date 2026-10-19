"""Separate actual witness identity changes from numerical record changes.

Read only independently audited v3 rows; never modify frozen flags/artifacts.
"""
import json
from run_simplex_soft5_knn_pilot import ROOT,digest

FOLDER=ROOT/'results/dynamic-certificate-development-v3'


def identities(cell):
    return [(r['object'],tuple(r['friends']),tuple(r['excluded_enemies'])) for r in cell['witnesses']]


def main():
    rows=[]
    for ap in sorted(FOLDER.glob('*.audit.json')):
        audit=json.loads(ap.read_text());name=audit['dataset'];path=FOLDER/(name+'.json')
        assert audit['passed'] and audit['sha256']==digest(path)
        saved=json.loads(path.read_text())
        for arm,sha in audit['fit_hashes'].items():
            fp=FOLDER/(name+'.'+arm+'.fit.json');assert digest(fp)==sha
            fit=json.loads(fp.read_text())['fit']
            if not fit.get('available'):continue
            previous=None;structure_changes=0;metadata_changes=0
            for step in fit['steps']:
                cell=step['witness_cell'];current=identities(cell)
                changed=previous is not None and current!=previous
                structure_changes+=changed;metadata_changes+=step['witness_refresh_changed']
                previous=current
            rows.append({'dataset':name,'arm':arm,'steps':len(fit['steps']),
                'actual_identity_changes':structure_changes,'stored_record_change_flags':metadata_changes,
                'source_fit_sha256':sha,'source_independent_dataset_audit_sha256':digest(ap)})
    out={'independently_audited_dataset_count':len({r['dataset'] for r in rows}),
        'rows':rows,'actual_identity_changes':sum(r['actual_identity_changes'] for r in rows),
        'stored_record_change_flags':sum(r['stored_record_change_flags'] for r in rows),
        'classification_superiority_established':False,'goal_achieved':False,
        'limit':'Partial audited cohort snapshot. Metadata flags include bound changes; actual structural changes are not predictive gains.'}
    (ROOT/'research/dynamic-witness-structure-diagnosis-v1.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
