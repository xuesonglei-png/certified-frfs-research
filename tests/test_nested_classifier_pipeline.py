import importlib.util
from pathlib import Path
import sys
import numpy as np

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('nested_pipeline',scripts/'compare_nested_classifier_pipelines.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_representation_scale_and_uniform_ablation():
    c={'features':[0,2],'weights':[.2,0.,.8]}
    reps=module.representations(c,c,3)
    np.testing.assert_allclose(reps['proposed_weighted'][1],[.4,1.6])
    np.testing.assert_array_equal(reps['proposed_uniform'][1],[1.,1.])
    assert reps['full'][0]==[0,1,2]


def test_empty_support_is_training_prior_for_both_models():
    A=np.arange(12).reshape(6,2);B=np.array([[1,2],[3,4]])
    y=np.array([0,0,0,0,1,1])
    for kind in module.KINDS:
        np.testing.assert_array_equal(module.predict(kind,1.,A,y,B,[],np.array([])),[0,0])
