"""Version solver-start extension without modifying the frozen original."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    text=(ROOT/'scripts/fit_fixed_face_majority_milp.py').read_text(encoding='utf-8')
    text=text.replace('def fit_majority_face_milp(', 'def fit_warm_majority_face_milp(',1)
    text=text.replace('from scipy.sparse import coo_matrix','from scipy.sparse import coo_matrix\nfrom warm_majority_mip_solver import solve_warm_mip',1)
    old="    solution=milp(objective,integrality=np.asarray(integrality),bounds=Bounds(lower,upper),\n        constraints=(LinearConstraint(matrix,-np.inf,b),LinearConstraint(eq,1.,1.)),\n        options={'time_limit':float(time_limit),'mip_rel_gap':0.01,'disp':False})\n    seed_report=strict_majority_report(A,y,fs,seed,margin=margin)"
    assert text.count(old)==1
    text=text.replace(old,"    seed_report=strict_majority_report(A,y,fs,seed,margin=margin)\n    solution,warm_start_record=solve_warm_mip(A,fs,seed,seed_report,declarations,objective,matrix,b,eq,lower,upper,integrality,time_limit)")
    old="'mass':mass,'margin':margin,'reg':reg,'time_limit':time_limit,'optimization_eps':0.,'certificate_eps':.01,"
    assert text.count(old)==1
    text=text.replace(old,old+"'warm_start':warm_start_record,")
    target=ROOT/'scripts/fit_warm_majority_face_milp.py';assert not target.exists();target.write_text(text,encoding='utf-8')


if __name__=='__main__':main()
