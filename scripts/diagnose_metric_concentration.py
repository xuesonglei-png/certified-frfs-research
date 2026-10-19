"""Training-only class vote and weight concentration on saved query trajectories."""
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.query_probability import QueryProbabilityObjective


def main():
    threadpool_limits(1); out=ROOT/'results/metric-concentration-diagnostic.json'
    if out.exists():raise FileExistsError(out)
    sources=[ROOT/'results/feasible-query-ablation/complete.json',ROOT/'results/vote-objective-diagnostic/complete.json'];rows=[]
    for source in sources:
        p=json.loads(source.read_text())
        for row in p['rows']:
            X,y=load_dataset(row['dataset']);splits=[]
            for s in row['splits']:
                ri=np.array(s['reference_indices']);labels=y[ri];fit=s['fit'];objective=QueryProbabilityObjective(X[ri],labels)
                assert objective.splits==fit['training_splits'];frames=[]
                for split in objective.splits:
                    ii,jj=np.array(split['reference']),np.array(split['query']);A,B=preprocess_fit(X[ri][ii],X[ri][jj]);frames.append((np.abs(B[:,None,:]-A[None,:,:]),labels[ii],labels[jj]))
                nodes=[]
                for node in fit['trace']:
                    weights=np.array(node['weights']);normalized=weights/weights.sum();correct={int(q):[] for q in np.unique(labels)}
                    for component,yt,yv in frames:
                        D=component@weights;nn=np.argsort(D,axis=1,kind='stable')[:,:min(5,len(yt))];classes=np.unique(yt)
                        pred=classes[np.stack([(yt[nn]==q).sum(axis=1) for q in classes],axis=1).argmax(axis=1)]
                        for q in np.unique(yv):correct[int(q)].extend((pred[yv==q]==q).tolist())
                    recall={str(q):float(np.mean(v)) for q,v in correct.items() if v}
                    nodes.append({'iteration':node['iteration'],'objective':node['objective'],
                        'effective_attributes':float(1/np.sum(normalized**2)),
                        'max_weight_share':float(normalized.max()),'active_attributes':int((weights>0).sum()),
                        'training_crossfit_class_recall':recall,'training_crossfit_BA':float(np.mean(list(recall.values())))})
                splits.append({'nodes':nodes})
            rows.append({'method_source':str(source.relative_to(ROOT)),'dataset':row['dataset'],'d':row['d'],'splits':splits})
    summaries=[]
    for source in sources:
        selected=[r for r in rows if r['method_source']==str(source.relative_to(ROOT))]
        effective=float(np.mean([np.mean([s['nodes'][-1]['effective_attributes']/r['d'] for s in r['splits']]) for r in selected]))
        delta=float(np.mean([np.mean([s['nodes'][-1]['training_crossfit_BA']-s['nodes'][0]['training_crossfit_BA'] for s in r['splits']]) for r in selected]))
        summaries.append({'source':str(source.relative_to(ROOT)),'mean_effective_attribute_fraction':effective,'mean_training_crossfit_BA_change':delta})
    result={'rows':rows,'summaries':summaries,'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'scope':'adaptive_training_only_class_recall_concentration_not_external_selection_or_inference'}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
