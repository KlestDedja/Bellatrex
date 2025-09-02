# Minimal DummyCSR for .toarray() compatibility
class DummyCSR:
    def toarray(self):
        return np.ones(3)
    def __array__(self, dtype=None):
        return np.ones(3)


# DummyTree for single learner and ensemble
class DummyTree:
    def __init__(self, n_features=3):
        self.feature = np.array([0, 1, -1])
        self.n_node_samples = np.array([10, 5, 5])
        self.decision_path_called = False
        self.n_features_in_ = n_features
        self.tree_ = self  # For compatibility with code expecting .tree_
    def decision_path(self, X):
        self.decision_path_called = True
        return DummyCSR()

# DummyClf for ensemble case
class DummyClf:
    def __init__(self, n_features=3):
        self.estimators_ = [DummyTree(n_features)]
        self.n_features_in_ = n_features
        self.tree_ = self.estimators_[0]  # For single-tree compatibility
    def __getitem__(self, idx):
        return self.estimators_[idx]


import pytest
import numpy as np
import pandas as pd
from bellatrex import tree_representation_utils as tr_utils

@pytest.mark.filterwarnings('ignore:MDS matrix has rank 0')
def test_add_emergency_noise():
    mat = np.zeros((3, 3))
    noisy = tr_utils.add_emergency_noise(mat, noise_level=1e-2)
    assert noisy.shape == mat.shape
    assert not np.allclose(noisy, mat)

def test_count_rule_length_ensemble():
    clf = DummyClf()
    sample = pd.DataFrame([[1, 2, 3]])
    length = tr_utils.count_rule_length(clf, 0, sample)
    assert isinstance(length, (int, float, np.integer, np.floating))

def test_count_rule_length_single():
    class SingleClf:
        def __init__(self, n_features=3):
            self.tree_ = DummyTree(n_features)
            self.n_features_in_ = n_features
    clf = SingleClf()
    # Patch tree.decision_path for single learner case
    clf.tree_.decision_path = lambda X: np.ones(3)
    sample = pd.DataFrame([[1, 2, 3]])
    length = tr_utils.count_rule_length(clf, 0, sample)
    assert isinstance(length, (int, float, np.integer, np.floating))

def test_tree_splits_to_vector_simple():
    clf = DummyClf()
    vec = tr_utils.tree_splits_to_vector(clf, 0, split_weight="simple")
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == clf.n_features_in_

def test_tree_splits_to_vector_by_samples():
    clf = DummyClf()
    vec = tr_utils.tree_splits_to_vector(clf, 0, split_weight="by_samples")
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == clf.n_features_in_

def test_tree_splits_to_vector_invalid():
    clf = DummyClf()
    with pytest.raises(KeyError):
        tr_utils.tree_splits_to_vector(clf, 0, split_weight="invalid")

def test_rule_splits_to_vector_simple():
    clf = DummyClf()
    sample = np.array([1, 2, 3])
    vec = tr_utils.rule_splits_to_vector(clf, 0, feature_represent="simple", sample=sample)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == clf.n_features_in_

def test_rule_splits_to_vector_weighted():
    clf = DummyClf()
    sample = np.array([1, 2, 3])
    vec = tr_utils.rule_splits_to_vector(clf, 0, feature_represent="weighted", sample=sample)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == clf.n_features_in_

def test_rule_splits_to_vector_invalid():
    clf = DummyClf()
    sample = np.array([1, 2, 3])
    with pytest.raises(KeyError):
        tr_utils.rule_splits_to_vector(clf, 0, feature_represent="invalid", sample=sample)
