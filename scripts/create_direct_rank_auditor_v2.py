"""Version rank identity audit and a resumable scientific recovery chain."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def write(path,text):
    p=ROOT/path;assert not p.exists(),p;p.write_text(text,encoding='utf-8')


def main():
    text=(ROOT/'scripts/audit_fixed_face_majority_milp.py').read_text(encoding='utf-8')
    text=text.replace('from audit_forest_compact_transfer import equivalent','from audit_forest_compact_transfer import equivalent\nfrom independent_direct_rank_ties_v2 import validate_report',1)
    old="    equivalent(seed_report,fit['seed_report']);equivalent(final_report,fit['final_report'])";assert text.count(old)==1
    text=text.replace(old,"    validate_report(A,y,fs,seed,margin,fit['seed_report']);validate_report(A,y,fs,point,margin,fit['final_report'])")
    old="        equivalent(candidate_report,fit['candidate_report'])";assert text.count(old)==1
    text=text.replace(old,"        validate_report(A,y,fs,candidate,margin,fit['candidate_report'])")
    write('scripts/audit_fixed_face_majority_milp_v2.py',text)
    audit=(ROOT/'scripts/audit_direct_majority_frs_development.py').read_text(encoding='utf-8')
    audit=audit.replace('from audit_fixed_face_majority_milp import check_fit','from audit_fixed_face_majority_milp_v2 import check_fit',1)
    audit=audit.replace("name+'.audit.json'","name+'.audit-v2.json'")
    write('scripts/audit_direct_majority_frs_development_v2.py',audit)
    warm=(ROOT/'scripts/audit_warm_majority_mip.py').read_text(encoding='utf-8')
    warm=warm.replace('from audit_fixed_face_majority_milp import check_fit as check_cold,rank_report',
        'from audit_fixed_face_majority_milp_v2 import check_fit as check_cold,rank_report',1)
    old="seed=fit['seed'];fs=fit['features'];report=rank_report(A,y,fs,np.asarray(seed),fit['margin'])";assert warm.count(old)==1
    warm=warm.replace(old,"seed=fit['seed'];fs=fit['features'];report=fit['seed_report']  # independently validated nearest-rank tie choices above")
    write('scripts/audit_warm_majority_mip_v2.py',warm)


if __name__=='__main__':main()
