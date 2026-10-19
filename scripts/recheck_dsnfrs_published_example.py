"""Corrected published Table1 and explicit subset search for Table3 agreement.

Table1/page7 and Table3/page8 were visually verified. Table3's attribute
subset is not explicit in the example text. Searching it is a retrospective
formula diagnostic, not evidence identifying the authors' experiment settings.
The old erroneous/transcribed diagnostic is preserved, never overwritten.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import numpy as np
from dsnfrs_formula_baseline import DirectedSoftNeighborhood

ROOT=Path(__file__).resolve().parents[1]
TABLE1=[['4.4','2.9','1.4','.2'],['4.3','3','1.1','.1'],
        ['4.4','3','1.3','.2'],['4.5','2.3','1.3','.3'],['4.4','3.2','1.3','.2'],
        ['5.5','2.3','4','1.3'],['5.5','2.4','3.8','1.1'],['5.6','3','4.1','1.3'],
        ['5.5','2.5','4','1.3'],['5.5','2.6','4.4','1.2'],['7.7','3.8','6.7','2.2'],
        ['7.7','2.6','6.9','2.3'],['7.7','2.8','6.7','2'],['7.9','3.8','6.4','2'],
        ['6.3','2.8','5.1','1.5']]
PRINTED=['.9998','.9999','.9998','.9995','.9998','.7758','.7758','1.0000',
         '.7758','.7758','1.0000','1.0000','1.0000','1.0000','1.0000']


def rational_profile(features,*,beta=F(8,5),ddof=1,zero_lambda=F(1,1000000)):
    A=[[F(v) for v in row] for row in TABLE1]; y=[i//5 for i in range(15)]
    variances={}
    for label in range(3):
        group=[row for row,c in zip(A,y) if c==label]
        values=[]
        for a in range(4):
            mean=sum((row[a] for row in group),F(0))/len(group)
            variance=sum(((row[a]-mean)**2 for row in group),F(0))/(len(group)-ddof)
            values.append(variance if variance else zero_lambda)
        variances[label]=values
    output=[]
    for i,row in enumerate(A):
        distances=[sum(((row[a]-other[a])**2/variances[y[j]][a] for a in features),F(0))
                   for j,other in enumerate(A)]
        enemies=sorted((j for j in range(15) if y[j]!=y[i]),key=lambda j:(distances[j],j))[:5]
        supports=[sum(y[j]==y[i] and distances[j]<distances[e] for j in range(15)) for e in enemies]
        chosen=max(range(5),key=lambda t:(supports[t]-beta*t,-t)); enemy=enemies[chosen]
        with localcontext() as ctx:
            ctx.prec=80
            exponent=Decimal(distances[enemy].numerator)/Decimal(distances[enemy].denominator)
            lower=Decimal(1)-(-exponent).exp()
            rounded=format(lower,'.4f')
        output.append({'object':i+1,'enemy_object':enemy+1,'enemy_rank':chosen+1,
            'support_counts':supports,'exact_selected_distance':str(distances[enemy]),
            'lower_decimal80':str(lower),'rounded4':rounded})
    return output


def main():
    old_path=ROOT/'research/dsnfrs-paper-example-diagnostic.json'
    old=json.loads(old_path.read_text()); A=np.array(TABLE1,float); y=np.repeat([1,2,3],5)
    expected=np.array(PRINTED,float); corrections=[]
    for i,row in enumerate(old['A']):
        for j,value in enumerate(row):
            if value!=A[i,j]:
                corrections.append({'object':i+1,'attribute':j+1,'old':value,'published':float(A[i,j])})
    assert corrections==[{'object':3,'attribute':4,'old':.1,'published':.2},
                         {'object':4,'attribute':4,'old':.1,'published':.3},
                         {'object':5,'attribute':4,'old':.1,'published':.2}]
    configurations=sorted({(r['ddof'],r['beta'],r['zero_variance_lambda']) for r in old['rows']})
    assert len(configurations)==24
    rows=[]
    for size in range(1,5):
        for fs in combinations(range(4),size):
            for ddof,beta,lam in configurations:
                model=DirectedSoftNeighborhood(A,y,k=5,beta=beta,variance_ddof=ddof,
                    zero_variance_lambda=lam,singleton_variance='zero')
                actual=model.profile(model.distances(fs))['own_class_lower']
                rounded=[format(v,'.4f') for v in actual]
                match=rounded==[format(float(v),'.4f') for v in PRINTED]
                rows.append({'features_zero_based':list(fs),'ddof':ddof,'beta':beta,
                    'zero_variance_lambda':lam,'actual':actual,'rounded4':rounded,
                    'max_error':float(np.max(abs(np.array(actual)-expected))),
                    'matches_all_15_printed_own_class_values':match})
    exact=rational_profile([0]); assert [r['rounded4'] for r in exact]==[format(float(v),'.4f') for v in PRINTED]
    production=DirectedSoftNeighborhood(A,y,k=5,beta=1.6,variance_ddof=1,
        zero_variance_lambda=1e-6,singleton_variance='zero').profile(
            DirectedSoftNeighborhood(A,y,k=5,beta=1.6,variance_ddof=1,
                zero_variance_lambda=1e-6,singleton_variance='zero').distances([0]))
    assert np.allclose(production['own_class_lower'],[float(r['lower_decimal80']) for r in exact],rtol=0,atol=1e-14)
    matched=[r for r in rows if r['matches_all_15_printed_own_class_values']]
    sources=['research/literature/pdf/Directed2026.pdf',
             'research/literature/qa/dsnfrs-20261011/page-07.png',
             'research/literature/qa/dsnfrs-20261011/example-verified-08.png',
             'research/dsnfrs-paper-example-diagnostic.json','scripts/dsnfrs_formula_baseline.py',
             'scripts/recheck_dsnfrs_published_example.py']
    result={'passed':True,'table1':TABLE1,'labels':y.tolist(),'table3_own_class_printed':PRINTED,
        'old_record_corrections':corrections,'comparisons':len(rows),'matches':len(matched),
        'matched_subsets_zero_based':sorted({tuple(r['features_zero_based']) for r in matched}),
        'matches_rows':matched,'rational_distance_decimal80_a1':exact,'rows':rows,
        'fingerprints':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'limitations':['Attribute subset B not explicitly identified in the example text.',
            'Retrospective subset/convention search supports a compatible formula implementation, not author-setting identification.',
            'Only 15 own-class memberships checked; not a full off-class operator reconstruction.',
            'No dataset classifier, existing frozen protocol, benchmark artifacts or paper text changed.',
            'No prediction-superiority inference from this toy example.'],
        'scope':'corrected_published_example_formula_validation_not_author_experiment_reproduction'}
    out=ROOT/'research/dsnfrs-published-example-corrected-recheck.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','old_record_corrections','comparisons','matches','matched_subsets_zero_based','limitations')},indent=2))


if __name__=='__main__': main()
