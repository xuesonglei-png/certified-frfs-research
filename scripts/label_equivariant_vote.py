"""Resolve 5NN plurality ties geometrically instead of by numeric class code.

All strict vote winners, hence every certified three-friend 5NN majority,
are unchanged. Equal counts use mean neighbor distance then reference ID.
"""
import numpy as np


def predict(distance,y):
    distance,y=np.asarray(distance,float),np.asarray(y)
    if distance.ndim!=2 or distance.shape[1]!=len(y) or len(y)<5 or not np.isfinite(distance).all():
        raise ValueError('finite query/reference distances with at least five references required')
    classes=np.unique(y);nn=np.argsort(distance,axis=1,kind='stable')[:,:5]
    votes=np.column_stack([(y[nn]==c).sum(axis=1) for c in classes])
    predictions=classes[votes.argmax(axis=1)].copy();tied=np.zeros(len(distance),bool)
    for i in range(len(distance)):
        contenders=np.flatnonzero(votes[i]==votes[i].max())
        if len(contenders)==1:continue
        tied[i]=True
        def key(ci):
            ids=nn[i][y[nn[i]]==classes[ci]]
            return (float(np.mean(distance[i,ids])),int(ids.min()))
        winner=min(contenders,key=key);predictions[i]=classes[winner]
    return {'predictions':predictions.tolist(),'probabilities':((votes+1)/(5+len(classes))).tolist(),
        'neighbor_indices':nn.tolist(),'tied_queries':tied.tolist(),
        'rule':'strict_plurality_then_mean_neighbor_distance_then_reference_id'}


def independent_predict(A,y,B,features,weights):
    """Scalar coordinate accumulation and direct class counter, no vote helper."""
    A,y,B=np.asarray(A,float),np.asarray(y),np.asarray(B,float);predictions=[]
    for query in B:
        distance=[sum(abs(query[a]*w-A[j,a]*w) for a,w in zip(features,weights)) for j in range(len(A))]
        ids=sorted(range(len(A)),key=lambda j:(distance[j],j))[:5]
        counts={label:sum(y[j]==label for j in ids) for label in np.unique(y)}
        maximum=max(counts.values());tied=[label for label in counts if counts[label]==maximum]
        scores={label:(sum(distance[j] for j in ids if y[j]==label)/counts[label],
                       min(j for j in ids if y[j]==label)) for label in tied}
        predictions.append(min(tied,key=lambda label:scores[label]))
    return np.array(predictions).tolist()


def independent_predict_vectorized(A,y,B,features,weights):
    """Explicit coordinate distances and direct counters for full cohort replay."""
    A,y,B=np.asarray(A,float),np.asarray(y),np.asarray(B,float)
    distance=np.zeros((len(B),len(A)))
    for a,w in zip(features,weights):distance+=abs(B[:,a,None]*w-A[None,:,a]*w)
    predictions=[]
    for row in distance:
        ids=np.argsort(row,kind='stable')[:5].tolist()
        counter={label:sum(y[j]==label for j in ids) for label in np.unique(y)}
        maximum=max(counter.values());contenders=[label for label in counter if counter[label]==maximum]
        scores={label:(sum(float(row[j]) for j in ids if y[j]==label)/counter[label],
                       min(j for j in ids if y[j]==label)) for label in contenders}
        predictions.append(min(contenders,key=lambda label:scores[label]))
    return np.array(predictions).tolist()
