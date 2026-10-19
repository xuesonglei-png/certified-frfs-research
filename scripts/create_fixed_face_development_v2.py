"""Correct the v1 control-region mismatch; retain every frozen v1 artifact."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def change(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def write(path,text):
    p=ROOT/path;assert not p.exists(),p;p.write_text(text,encoding='utf-8')


def main():
    code=(ROOT/'scripts/run_fixed_face_frs_development.py').read_text(encoding='utf-8')
    code=change(code,"    'capped-frs-training','frs-joint-zero-face-prediction')}","    'frs-joint-zero-face-prediction')}")
    code=code.replace('fixed-face-frs-development','fixed-face-frs-development-v2')
    code=change(code,"    old=json.loads((PARENTS['capped-frs-training']/(name+'.json')).read_text())\n",'')
    code=change(code,"    first=alignment(joint,native);block=old['inner_training'][0]\n    assert old['reference_indices']==joint['reference_indices']\n    assert block['reference_indices']==first['reference_indices']==joint['training_indices']",
        "    first=alignment(joint,native)\n    assert first['reference_indices']==joint['training_indices']")
    code=change(code,"    assert block['features']==fs and block['anchor']==anchor.tolist() and block['strict_cell_available']==joint['available']\n",'')
    code=change(code,"            if arm=='full_support_robust':\n                record=adapt_robust_control(block['fits']['anchor'])\n                reuse[arm]='audited_capped_anchor_fit_only'\n            elif arm.startswith('face_') and not zeros:",
        "            if arm.startswith('face_') and not zeros:")
    begin=code.index('def adapt_robust_control(');end=code.index('def compact_metric(')
    code=code[:begin]+code[end:]
    write('scripts/run_fixed_face_frs_development_v2.py',code)

    audit=(ROOT/'scripts/audit_fixed_face_frs_development.py').read_text(encoding='utf-8')
    audit=audit.replace('from run_fixed_face_frs_development import','from run_fixed_face_frs_development_v2 import')
    audit=change(audit,"    old=json.loads((PARENTS['capped-frs-training']/(name+'.json')).read_text())\n",'')
    audit=change(audit,"saved['available']==joint['available']==old['inner_training'][0]['strict_cell_available']", "saved['available']==joint['available']")
    begin=audit.index("            if arm=='full_support_robust':");end=audit.index("            elif facial and not zeros:",begin)
    audit=audit[:begin]+audit[end:].replace('            elif facial and not zeros:','            if facial and not zeros:',1)
    write('scripts/audit_fixed_face_frs_development_v2.py',audit)

    prep=(ROOT/'scripts/prepare_fixed_face_frs_development.py').read_text(encoding='utf-8')
    prep=prep.replace('from run_fixed_face_frs_development import','from run_fixed_face_frs_development_v2 import')
    prep=prep.replace('fixed-face-frs-development','fixed-face-frs-development-v2')
    prep=prep.replace('fixed-face-FRS-prototype-tests.json','fixed-face-FRS-development-v2-tests.json')
    prep=prep.replace('fixed-face-FRS-development-protocol.md','fixed-face-FRS-development-v2-protocol.md')
    prep=prep.replace('fixed-face-FRS-development-review.json','fixed-face-FRS-development-v2-review.json')
    prep=prep.replace("'run_fixed_face_frs_development.py'","'run_fixed_face_frs_development_v2.py'")
    prep=prep.replace("'audit_fixed_face_frs_development.py'","'audit_fixed_face_frs_development_v2.py'")
    prep=change(prep,"    # Fixed reverse lexical order: no selection based on current score.\n    for name in sorted(parents['frs-joint-zero-face']['artifact_hashes'],reverse=True):",
        "    # Predeclared schedule; every dataset is run, no result-based selection.\n    schedule=('iris','wine','seeds','transfusion','wdbc','vehicle','vertebral','heart_statlog',\n        'ionosphere','sonar','ecoli','glass','haberman','banknote','yeast')\n    assert set(schedule)==set(parents['frs-joint-zero-face']['artifact_hashes'])\n    for name in schedule:")
    write('scripts/prepare_fixed_face_frs_development_v2.py',prep)
    protocol=(ROOT/'research/fixed-face-FRS-development-protocol.md').read_text(encoding='utf-8')
    protocol=change(protocol,'Full-support robust arm reuses the completely audited old anchor-start fit,\nnot its better warm-start endpoint. New full-support nominal arm starts anchor.',
        'Both full-support controls are fitted freshly using EXACTLY the same pair\nregion as the facial methods. Both start the original anchor. The older\ncapped control used a one-sided region and must not be called a matched\npair-region control; v1 smoke audit detected this error and stopped it.')
    protocol+='\nV2 leaves v1 code/freeze/results/failure intact. Computational order is fixed\nIris, Wine, Seeds, Transfusion, WDBC, Vehicle, Vertebral, Heart Statlog,\nIonosphere, Sonar, Ecoli, Glass, Haberman, Banknote, Yeast. No stopping on a\nfavorable row. All four arms independently use the same optimizer and region.\n'
    write('research/fixed-face-FRS-development-v2-protocol.md',protocol)
    write('research/fixed-face-FRS-development-v1-failure.json',json.dumps({
        'retained_source':'research/fixed-face-frs-development-source-freeze.json',
        'retained_actual_iris':'results/fixed-face-frs-development/iris.json',
        'error':'Audited old capped control used a one-sided rather than pair-transition FRS region.',
        'stage':'independent smoke audit rejected mismatch before ALL15 execution',
        'v2_fix':'Freshly fit both full-support controls under the same pair region as facial arms.',
        'classification_superiority_established':False,'goal_achieved':False},indent=2))


if __name__=='__main__':main()
