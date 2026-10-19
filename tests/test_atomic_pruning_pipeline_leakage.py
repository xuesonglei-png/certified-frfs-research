"""Outer truth changes scores, never full-pipeline model selection."""
from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_atomic_pruning_classifier as pipeline


def test_entire_iris_pipeline_ignores_outer_labels_for_choices(tmp_path,monkeypatch):
    # Existing Iris is exposed development; outputs here are QA only and
    # never replace or join the scientific cohort's JSON artifacts.
    X,y=pipeline.load_dataset('iris')
    source=json.loads((pipeline.SOURCE/'iris.json').read_text()); vi=np.array(source['validation_indices'])
    original_folder=tmp_path/'original'; changed_folder=tmp_path/'changed'
    monkeypatch.setattr(pipeline,'FOLDER',original_folder)
    pipeline.run('iris'); first=json.loads((original_folder/'iris.json').read_text())
    altered=y.copy(); classes=np.unique(y); encoded=np.searchsorted(classes,altered[vi])
    altered[vi]=classes[(encoded+1)%len(classes)]
    monkeypatch.setattr(pipeline,'load_dataset',lambda name:(X,altered))
    monkeypatch.setattr(pipeline,'FOLDER',changed_folder)
    pipeline.run('iris'); second=json.loads((changed_folder/'iris.json').read_text())
    assert first['selection']==second['selection'] and first['inner']==second['inner']
    assert first['outer']['pruning_records']==second['outer']['pruning_records']
    assert first['outer']['exact_stage']==second['outer']['exact_stage']
    differences=0
    for method in pipeline.METHODS:
        a=first['outer']['results'][method]; b=second['outer']['results'][method]
        assert a['metric']==b['metric'] and a['option_index']==b['option_index']
        assert a['evaluation']['predictions']==b['evaluation']['predictions']
        differences+=a['evaluation']['metrics']!=b['evaluation']['metrics']
    assert differences>0
