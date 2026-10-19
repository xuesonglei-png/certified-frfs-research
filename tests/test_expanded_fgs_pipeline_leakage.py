import json
from pathlib import Path
import numpy as np
import run_expanded_fgs_comparison as pipeline
from continue_soft_topk_knn_pilot import verify


def test_complete120grid_pipeline_choices_ignore_outer_truth(tmp_path,monkeypatch):
    X,y=pipeline.load_dataset('iris');audit=verify(pipeline.SOURCE)
    source=json.loads((pipeline.SOURCE/'iris.json').read_text());vi=np.array(source['validation_indices'])
    first_folder=tmp_path/'first';first_folder.mkdir();monkeypatch.setattr(pipeline,'FOLDER',first_folder)
    first=pipeline.run('iris',audit)
    changed=y.copy();classes=np.unique(y);changed[vi]=classes[(np.searchsorted(classes,y[vi])+1)%len(classes)]
    second_folder=tmp_path/'second';second_folder.mkdir();monkeypatch.setattr(pipeline,'FOLDER',second_folder)
    monkeypatch.setattr(pipeline,'load_dataset',lambda name:(X,changed));second=pipeline.run('iris',audit)
    assert first['inner']==second['inner'] and first['selection']==second['selection']
    assert first['restricted_legacy_selection']==second['restricted_legacy_selection']
    assert first['outer']['paths']==second['outer']['paths']
    differences=0
    for m in pipeline.METHODS:
        a=first['outer']['results'][m];b=second['outer']['results'][m]
        assert a['config_index']==b['config_index'] and a['metric']==b['metric'] and a['query_certificate']==b['query_certificate']
        assert a['evaluation']['predictions']==b['evaluation']['predictions']
        assert [r['predictions'] for r in a['perturbation_evaluations']]==[r['predictions'] for r in b['perturbation_evaluations']]
        differences+=a['evaluation']['metrics']!=b['evaluation']['metrics']
    assert differences==2
