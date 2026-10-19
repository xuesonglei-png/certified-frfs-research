"""Fixed-classifier posthoc transfer; never choose algorithm by outer scores."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from run_predictive_certified_iteration import scaled_split

ap=argparse.ArgumentParser()
ap.add_argument('--iteration',type=int,default=10)
args=ap.parse_args()
directory=ROOT/f'results/predictive-iteration{args.iteration}'
threadpool_limits(1)
records=[]
for path in sorted(directory.glob('*-s*-f*.json')):
    obj=json.loads(path.read_text())
    X,y=load_dataset(obj['dataset'])
    tr,te=obj['train_indices'],obj['test_indices']
    A,B=scaled_split(X[tr],X[te])
    for row in obj['rows']:
        fs=row['features']
        if not fs:
            continue
        models={'knn1_L1':KNeighborsClassifier(n_neighbors=1,metric='manhattan',algorithm='brute'),
                'knn5_L1':KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute'),
                'knn5_L2':KNeighborsClassifier(n_neighbors=5,algorithm='brute'),
                'svm_RBF_C1':SVC(C=1,gamma='scale'),
                'logistic_C1':LogisticRegression(C=1,max_iter=3000,random_state=101),
                'rf300':RandomForestClassifier(n_estimators=300,random_state=101,n_jobs=1)}
        for name,model in models.items():
            model.fit(A[:,fs],y[tr])
            pred=model.predict(B[:,fs])
            ba=float(balanced_accuracy_score(y[te],pred))
            if name=='knn1_L1':
                assert abs(ba-row['metrics']['0.0']['balanced_accuracy'])<1e-12
            records.append({'dataset':obj['dataset'],'fold':obj['fold'],'method':row['method'],
                            'classifier':name,'balanced_accuracy':ba,
                            'macro_f1':float(f1_score(y[te],pred,average='macro',zero_division=0))})
    print(json.dumps({'dataset':obj['dataset'],'fold':obj['fold'],'status':'transfer_audited'}),flush=True)
summary={}
for classifier in sorted(set(r['classifier'] for r in records)):
    summary[classifier]={}
    for method in sorted(set(r['method'] for r in records)):
        means=[]
        for dataset in sorted(set(r['dataset'] for r in records)):
            rows=[r for r in records if r['classifier']==classifier and r['method']==method and r['dataset']==dataset]
            means.append(np.mean([r['balanced_accuracy'] for r in rows]))
        summary[classifier][method]=float(np.mean(means))
(directory/'classifier-transfer.json').write_text(json.dumps({'status':'posthoc_development_no_outer_model_selection',
    'scope':'only_L1_1NN_has_decision_certificate_other_classifiers_empirical_only',
    'records':records,'dataset_balanced_means':summary},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
