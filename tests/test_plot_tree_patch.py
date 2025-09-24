import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

try:
    from bellatrex.plot_tree_patch import _color_brew, plot_tree_patched, _MPLTreeExporter
except ImportError:
    from app.bellatrex.plot_tree_patch import _color_brew, plot_tree_patched, _MPLTreeExporter


def test_color_brew_length_and_type():
    n = 5
    colors = _color_brew(n)
    assert isinstance(colors, list)
    assert len(colors) == n
    for color in colors:
        assert isinstance(color, list)
        assert len(color) == 3
        assert all(isinstance(c, int) for c in color)
        assert all(0 <= c <= 255 for c in color)


def test_plot_tree_patched_runs():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    fig, ax = plt.subplots()
    anns = plot_tree_patched(clf, ax=ax)
    assert isinstance(anns, list)
    assert all(hasattr(a, "get_window_extent") for a in anns)
    plt.close(fig)


def test_plot_tree_patched_with_feature_names():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y)
    fig, ax = plt.subplots()
    anns = plot_tree_patched(clf, feature_names=iris.feature_names, ax=ax)
    assert isinstance(anns, list)
    plt.close(fig)


def test_exporter_invalid_precision():
    with pytest.raises(ValueError):
        _MPLTreeExporter(precision=-1)
    with pytest.raises(ValueError):
        _MPLTreeExporter(precision="bad")


@pytest.mark.skipif(
    pytest.importorskip("sksurv", reason="sksurv not installed") is None,
    reason="sksurv not available",
)
def test_plot_tree_patched_survivaltree():
    from sksurv.tree import SurvivalTree

    X = np.random.rand(20, 2)
    y = np.array([(True, 1.0 + i) for i in range(20)], dtype=[("event", "?"), ("time", "f8")])
    clf = SurvivalTree(max_depth=1, random_state=0)
    clf.fit(X, y)
    fig, ax = plt.subplots()
    anns = plot_tree_patched(clf, ax=ax)
    assert isinstance(anns, list)
    plt.close(fig)
