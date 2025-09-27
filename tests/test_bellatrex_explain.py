import os
import pytest
import numpy as np
import pandas as pd

try:  # TODO: Paths to be updated: this workaround makes tests work across different setups.
    from app.bellatrex.bellatrex_explain import BellatrexExplain
except ImportError:
    from bellatrex.bellatrex_explain import BellatrexExplain
from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest


@pytest.fixture
def mock_clf():
    return RandomForestClassifier(n_estimators=10, random_state=0)


@pytest.fixture
def mock_survival_clf():
    return RandomSurvivalForest(n_estimators=10, random_state=0)


@pytest.fixture
def mock_data():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = np.random.randint(0, 2, size=100)  # random binary target
    return X, y


@pytest.fixture
def mock_survival_data():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])

    times = np.random.exponential(scale=5, size=100).astype(float)
    status = np.random.binomial(1, 0.7, size=100).astype(bool)
    # order of columns is: status, times
    y = np.rec.fromarrays([status, times], dtype=[("event", "?"), ("time", "f8")])

    return X, y


def test_bellatrex_explain_fit(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    assert explainer.is_fitted() is True


# def test_bellatrex_explain_explain(mock_clf, mock_data):
#     X, y = mock_data
#     explainer = BellatrexExplain(mock_clf, verbose=0)
#     explainer.fit(X, y)
#     explanation = explainer.explain(X, idx=0)
#     assert explanation is not None


def test_bellatrex_explain_plot_overview(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    fig, axes = explainer.explain(X, idx=0).plot_overview(show=False)
    # fig, axes = explainer.plot_overview(show=False)
    assert fig is not None
    assert axes is not None


@pytest.mark.xfail(raises=NotImplementedError, reason="Not implemented yet")
def test_predict_survival_curve(mock_survival_clf, mock_survival_data):
    X, y = mock_survival_data
    explainer = BellatrexExplain(mock_survival_clf, verbose=1, set_up="survival")
    explainer.fit(X, y)
    survival_curve = explainer.predict_survival_curve(X, 0)  # pylint: disable=E1111

    assert survival_curve is not None
    assert isinstance(survival_curve, pd.DataFrame)


@pytest.mark.xfail(raises=NotImplementedError, reason="Not implemented yet")
def test_predict_median_surv_time(mock_survival_clf, mock_survival_data):
    X, y = mock_survival_data
    explainer = BellatrexExplain(mock_survival_clf, verbose=1, set_up="survival")
    explainer.fit(X, y)
    survival_curve = explainer.predict_median_surv_time(X, 0)  # pylint: disable=E1111
    assert survival_curve is not None
    assert isinstance(survival_curve, pd.DataFrame)


# BIG DUMMY TREE TO MAKE EVERYTHING WORK.
# Better refactgor code than go through this pain.
# class DummyTree(dict):
#     def __init__(self):
#         super().__init__()
#         self.value = [[0]]
#         self.n_outputs_ = 1
#         self["feature_names_in_"] = ["f1", "f2"]
#         self["n_features_in_"] = 2

#     def get(self, *a, **k):
#         return self.value


# def test_is_fitted_dict():
#     # Should wrap dict in EnsembleWrapper and return True

#     dict_trees = {"trees": [DummyTree(), DummyTree(), DummyTree()]}
#     explainer = BellatrexExplain(dict_trees)
#     assert explainer.is_fitted() is True


# def test_is_fitted_ensemblewrapper():
#     from bellatrex.wrapper_class import EnsembleWrapper

#     dict_trees = {"trees": [DummyTree(), DummyTree(), DummyTree()]}
#     ew = EnsembleWrapper(dict_trees)
#     explainer = BellatrexExplain(ew)
#     assert explainer.is_fitted() is True


# def test_fit_force_refit_verbose(monkeypatch, mock_clf, mock_data):
#     X, y = mock_data
#     explainer = BellatrexExplain(mock_clf, force_refit=True, verbose=2)
#     called = {}

#     def fake_fit(X_, y_, n_jobs):
#         called["fit"] = True

#     monkeypatch.setattr(mock_clf, "fit", fake_fit)
#     explainer.fit(X, y)
#     assert called["fit"]


# def test_explain_verbose(monkeypatch, mock_clf, mock_data):
#     X, y = mock_data
#     explainer = BellatrexExplain(mock_clf, verbose=5)
#     explainer.fit(X, y)

#     # Patch TreeExtraction to avoid heavy computation
#     class DummyTE:
#         def __init__(self, *a, **k):
#             pass

#         def set_params(self, **params):
#             return self

#         def main_fit(self):
#             class Dummy:
#                 final_trees_idx = [0]
#                 cluster_sizes = [1]
#                 score = lambda self, *a, **k: 1.0

#             return Dummy()

#     monkeypatch.setattr("app.bellatrex.bellatrex_explain.TreeExtraction", DummyTE)
#     explainer.explain(X, idx=0)
#     assert hasattr(explainer, "tuned_method")


# def test_plot_overview_verbose(monkeypatch, mock_clf, mock_data):
#     X, y = mock_data
#     explainer = BellatrexExplain(mock_clf, verbose=2)
#     explainer.fit(X, y)

#     class Dummy:
#         final_trees_idx = [0]
#         cluster_sizes = [1]
#         preselect_represent_cluster_trees = lambda self: (None, None)

#     explainer.tuned_method = Dummy()
#     explainer.sample = X.iloc[[0]]
#     explainer.surrogate_pred_str = "0.0"
#     explainer.clf = mock_clf
#     fig, axes = explainer.plot_overview(show=False)
#     assert fig is not None


def test_create_rules_txt_file(monkeypatch, mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf)
    explainer.fit(X, y)

    class DummyModel:
        final_trees_idx = [0]
        cluster_sizes = [1]
        sample = X.iloc[[0]]

    explainer.tuned_method = DummyModel()
    explainer.sample = X.iloc[[0]]
    explainer.sample_iloc = 0
    explainer.surrogate_pred_str = "0.0"
    explainer.clf = mock_clf
    # Monkeypatch rule_to_file and read_rules to avoid file IO
    # should paths include app.bellatrex or just bellatrex? TBD and TB fixed
    monkeypatch.setattr("bellatrex.utilities.rule_to_file", lambda *a, **k: None)
    monkeypatch.setattr("bellatrex.visualization.read_rules", lambda **k: ([1], [1], [1], [1], [1]))
    monkeypatch.setattr("bellatrex.visualization_extra._input_validation", lambda *a, **k: None)
    out_file, file_extra = explainer.create_rules_txt(
        out_dir="temp_files", out_file="testing_rules.txt"
    )
    assert os.path.exists(out_file)
    assert os.path.exists(file_extra)

    os.remove(out_file)
    os.remove(file_extra)
