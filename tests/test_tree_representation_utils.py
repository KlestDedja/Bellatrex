import pytest
import numpy as np
import pandas as pd
from bellatrex import tree_representation_utils as tru

class DummyTree:
    def __init__(self, n_features=3):
        self.feature = np.array([0, 1, -1])
        self.n_node_samples = np.array([10, 5, 5])
        self.decision_path_called = False
        self.n_features_in_ = n_features
        self.tree_ = self
    def decision_path(self, X):
        self.decision_path_called = True
        # Return a fake sparse matrix with 1s on the diagonal
        class DummyCSR:
            def toarray(self_inner):
                return np.eye(X.shape[1])
        return DummyCSR()

class DummyClf:
    def __init__(self):
        self.estimators_ = [DummyTree()]
        self.n_features_in_ = 3
        self.n_estimators = 1
    def __getitem__(self, idx):
        return self.estimators_[idx]
    @property
    def tree_(self):
        return self.estimators_[0]

@pytest.mark.filterwarnings('ignore:MDS matrix has rank 0')
def test_add_emergency_noise():
    mat = np.zeros((3, 3))
    noisy = tru.add_emergency_noise(mat, noise_level=1e-2)
    assert noisy.shape == mat.shape
    assert not np.allclose(noisy, mat)

def test_count_rule_length_ensemble():
    clf = DummyClf()
    sample = pd.DataFrame([[1, 2, 3]])
    length = tru.count_rule_length(clf, 0, sample)
    assert isinstance(length, (int, float, np.integer, np.floating))

def test_count_rule_length_single():
    tree = DummyTree()
    tree.n_features_in_ = 3
    class SingleClf:
        def __init__(self):
            self.tree_ = tree
    clf = SingleClf()
    sample = pd.DataFrame([[1, 2, 3]])
    length = tru.count_rule_length(clf, 0, sample)
    assert isinstance(length, (int, float, np.integer, np.floating))

def test_tree_splits_to_vector_simple():
    clf = DummyClf()
    vec = tru.tree_splits_to_vector(clf, 0, split_weight="simple")
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == clf.n_features_in_

def test_tree_splits_to_vector_by_samples():
    clf = DummyClf()
    vec = tru.tree_splits_to_vector(clf, 0, split_weight="by_samples")
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == clf.n_features_in_

def test_tree_splits_to_vector_invalid():
    clf = DummyClf()
    with pytest.raises(KeyError):
        tru.tree_splits_to_vector(clf, 0, split_weight="invalid")

def test_rule_splits_to_vector_simple():
    clf = DummyClf()
    sample = np.array([1, 2, 3])
    vec = tru.rule_splits_to_vector(clf, 0, feature_represent="simple", sample=sample)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == clf.n_features_in_

def test_rule_splits_to_vector_weighted():
    clf = DummyClf()
    sample = np.array([1, 2, 3])
    vec = tru.rule_splits_to_vector(clf, 0, feature_represent="weighted", sample=sample)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == clf.n_features_in_

def test_rule_splits_to_vector_invalid():
    clf = DummyClf()
    sample = np.array([1, 2, 3])
    with pytest.raises(KeyError):
        tru.rule_splits_to_vector(clf, 0, feature_represent="invalid", sample=sample)
