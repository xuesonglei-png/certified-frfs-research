"""Independent numerical relations/CD/graph, selections and classifier votes."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from run_expanded_fgs_comparison import ROOT,SOURCE,FOLDER,METHODS,CONFIGS,LEGACY_INDICES,digest,fingerprints,summarize
from expanded_fgs_grid import BETAS,XIS
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_certified_metric_knn_pool import independent_evaluate
from audit_atomic_pruning_classifier import replay_final
from audit_forest_compact_transfer import equivalent


def graph(cd,edges):
    components=[{i} for i in range(len(cd))];tree=[]
    for weight,i,j in sorted(edges):
        if weight==0:continue
        left=next(k for k,c in enumerate(components) if i in c)
        right=next(k for k,c in enumerate(components) if j in c)
        if left==right:continue
        tree.append((float(weight),i,j));components[left]|=components[right];components.pop(right)
    neighbors=[set() for _ in cd];interactive=set()
    for weight,i,j in tree:
        if weight<0 and abs(weight)>=max(cd[i],cd[j]):interactive.update((i,j))
        if abs(weight)>=min(cd[i],cd[j]):neighbors[i].add(j);neighbors[j].add(i)
    groups=[];seen=set()
    for i in range(len(cd)):
        if i in seen:continue
        stack=[i];found=set()
        while stack:
            v=stack.pop()
            if v in found:continue
            found.add(v);stack.extend(neighbors[v]-found)
        seen|=found;groups.append(sorted(found))
    selected=set(interactive)
    for group in groups:selected.add(max(group,key=lambda i:(cd[i],-i)))
    return {'features':sorted(selected),'groups':groups,'tree':tree,'interactive_features':sorted(interactive)}


def paths(A,y):
    n,d=A.shape;classes=np.unique(y);decision=(y[:,None]==y[None,:]).astype(float)
    def entropy(r):return float(np.mean(-np.log2(np.sum(r,axis=1)/n)))
    hd=entropy(decision);lookup={}
    for xi in XIS:
        relations=[]
        for a in range(d):
            difference=np.abs(A[:,a,None]-A[None,:,a]);radius=np.std(A[:,a],ddof=1)/xi
            relations.append(np.where(difference<=radius,1-difference,0.))
        h=[entropy(r) for r in relations];joint=[entropy(np.minimum(r,decision)) for r in relations];edges=[]
        for i in range(d):
            for j in range(i+1,d):
                pair=np.minimum(relations[i],relations[j])
                edges.append((h[i]+h[j]+hd-entropy(pair)-joint[i]-joint[j]+entropy(np.minimum(pair,decision)),i,j))
        arrays=[[np.sort(1-r[:,y!=c],axis=1) for c in classes] for r in relations]
        totals=[[(1-r[:,y!=c]).sum(axis=1) for c in classes] for r in relations]
        for beta in BETAS:
            cd=[]
            for a in range(d):
                lower=[]
                for values,sums in zip(arrays[a],totals[a],strict=True):
                    if not values.shape[1]:lower.append(np.ones(n));continue
                    index=np.minimum(np.maximum(1,np.floor(sums*(1-beta)).astype(int)),values.shape[1])-1
                    lower.append(values[np.arange(n),index])
                cd.append(float(np.mean(np.max(lower,axis=0))))
            native={**graph(cd,edges),'cd':cd,'beta':beta,'xi':xi,
                'implementation':'independent_FGS_RFRAS_numeric_documented_boundary_convention'}
            chosen=set(native['features']);ranking=sorted(range(d),key=lambda a:(a not in chosen,-cd[a],a))
            lookup[(beta,xi)]={'native':native,'ranking':ranking,'adaptation':'native_members_first_then_descending_cd_not_original_budget_algorithm'}
    return [{'index':i,'config':dict(c),'path':lookup[(c['beta'],c['xi'])]} for i,c in enumerate(CONFIGS)]


def check_block(A,y,B,truth,block,count,counters):
    assert block['source_proposed_count']==count;fresh=paths(A,y);equivalent(fresh,block['paths'])
    assert set(block['options'])==set(METHODS);used=set();models=set()
    for m in METHODS:
        assert len(block['options'][m])==120
        for i,option in enumerate(block['options'][m]):
            assert option['config_index']==i;eid=option['evaluation_id'];used.add(eid)
            p=fresh[i]['path'];fs=p['native']['features'] if m=='fgs_native120' else sorted(p['ranking'][:count])
            assert block['records'][eid]['metric']=={'features':fs,'weights':[1.]*len(fs)}
    assert used==set(range(len(block['records'])))
    for record in block['records']:
        fs,w=record['metric']['features'],record['metric']['weights'];assert tuple(fs) not in models;models.add(tuple(fs))
        equivalent(independent_evaluate(A,y,B,truth,fs,w),record['evaluation']);counters['inner_predictions']+=1
    counters['selectors']+=120;counters['inner_folds']+=1


def choices(blocks,m,indices):
    rows=[]
    for i in indices:
        records=[b['records'][b['options'][m][i]['evaluation_id']] for b in blocks]
        rows.append({'config_index':i,'mean_BA':float(np.mean([r['evaluation']['metrics']['clean_BA'] for r in records])),
            'mean_logloss':float(np.mean([r['evaluation']['metrics']['balanced_logloss'] for r in records])),
            'mean_count':float(np.mean([len(r['metric']['features']) for r in records]))})
    chosen=min(rows,key=lambda r:(-round(r['mean_BA'],12),round(r['mean_logloss'],12),r['mean_count'],r['config_index']))
    return {'rows':rows,'selected':chosen['config_index']}


def audit_one(name,source_audit,counters):
    saved=json.loads((FOLDER/(name+'.json')).read_text());source=json.loads((SOURCE/(name+'.json')).read_text())
    assert saved['complete'] and not saved['query_labels_used_for_selection'] and saved['fingerprints']==fingerprints(name)
    assert digest(SOURCE/(name+'.json'))==source_audit['artifact_hashes'][name]
    assert saved['reference_indices']==source['reference_indices'] and saved['validation_indices']==source['validation_indices']
    X,y=load_dataset(name);ri,vi=np.array(saved['reference_indices']),np.array(saved['validation_indices']);assert not set(ri)&set(vi)
    raw,labels=X[ri],y[ri];assert len(saved['inner'])==len(source['inner'])
    for fi,(block,prior) in enumerate(zip(saved['inner'],source['inner'],strict=True)):
        ca,cb=np.array(block['reference_indices']),np.array(block['validation_indices'])
        assert block['reference_indices']==prior['reference_indices'] and block['validation_indices']==prior['validation_indices'] and not set(ca)&set(cb)
        A,B=preprocess_fit(raw[ca],raw[cb]);check_block(A,labels[ca],B,labels[cb],block,prior['exact_proposed_count'],counters)
        print(name,'inner',fi+1,'FGS120 graphs and classifier votes independently replayed',flush=True)
    for m in METHODS:
        equivalent(saved['selection'][m],choices(saved['inner'],m,range(120)))
        equivalent(saved['restricted_legacy_selection'][m],choices(saved['inner'],m,LEGACY_INDICES))
    native=saved['restricted_legacy_selection']['fgs_native120']['selected']
    for block,prior in zip(saved['inner'],source['inner'],strict=True):
        record=block['records'][block['options']['fgs_native120'][native]['evaluation_id']]
        equivalent(record['evaluation'],prior['options']['fgs_native'][0]['evaluation'])
    A,B=preprocess_fit(raw,X[vi]);outer=saved['outer'];fresh=paths(A,labels);equivalent(fresh,outer['paths']);counters['selectors']+=120
    count=source['outer']['exact_proposed_count'];assert outer['source_proposed_count']==count
    for m in METHODS:
        result=outer['results'][m];i=saved['selection'][m]['selected'];assert result['config_index']==i and result['config']==CONFIGS[i]
        fs=fresh[i]['path']['native']['features'] if m=='fgs_native120' else sorted(fresh[i]['path']['ranking'][:count])
        assert result['metric']=={'features':fs,'weights':[1.]*len(fs)}
        replay_final(A,labels,B,y[vi],result,counters)
    counters['datasets']+=1;print(name,'FGS120 outer nominal and16noise draws independently replayed',flush=True)
    return saved


def main():
    threadpool_limits(1);source_audit=verify(SOURCE);names=sorted(source_audit['artifact_hashes']);reports=[];counters=Counter()
    assert list(BETAS)==[i/10 for i in range(1,11)]
    assert list(XIS)==sorted({i/10 for i in range(1,20,2)}|{1.,2.})
    assert len(CONFIGS)==120 and len(LEGACY_INDICES)==9
    reports=[audit_one(name,source_audit,counters) for name in names]
    equivalent(json.loads((FOLDER/'complete.json').read_text()),summarize(reports))
    assert counters['datasets']==15 and counters['inner_folds']==44 and counters['selectors']==7080
    output={'passed':True,'counts':dict(counters),'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},
        'summary_sha256':digest(FOLDER/'complete.json'),'source_audit_sha256':digest(SOURCE/'complete-audit.json'),
        'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,
        'scope':'independent_relations_entropy_CD_set_graph_all120_selection_inner_outer_noise;declared_conventions_not_author_defaults'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(output,flush=True)


if __name__=='__main__':main()
