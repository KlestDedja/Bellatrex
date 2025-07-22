'''
Author: Klest Dedja
Here we manually test most of the features
'''

import os

IS_CI = os.environ.get("CI") == "true"
if IS_CI:
    import matplotlib
    matplotlib.use("Agg")  # Must be before importing pyplot

import pytest
import matplotlib.pyplot as plt  # Safe after backend is set
import bellatrex as btrex
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest

from bellatrex import BellatrexExplain
from bellatrex.wrapper_class import pack_trained_ensemble
from bellatrex.datasets import (
    load_mlc_data,
    load_regression_data,
    load_survival_data,
    load_binary_data,
    load_mtr_data
)
from bellatrex.utilities import get_auto_setup

MAX_TEST_SAMPLES = 2

DATA_LOADERS = [
    load_binary_data,
    load_regression_data,
    load_survival_data,
    load_mlc_data,
    load_mtr_data,
]

# --- Common test setup logic shared by both tests ---
def prepare_fitted_bellatrex(loader):
    X, y = loader(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    SETUP = get_auto_setup(y)

    if SETUP.lower() in "survival":
        clf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, n_jobs=-2, random_state=0)
    elif SETUP.lower() in ["binary", "multi-label"]:
        clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0)
    elif SETUP.lower() in ["regression", "multi-target"]:
        clf = RandomForestRegressor(n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0)
    else:
        raise ValueError(f"Detection task {SETUP} not compatible with Bellatrex (yet)")

    clf.fit(X_train, y_train)
    test_grid = {"n_trees": [0.6, 1.0], "n_dims": [2, None], "n_clusters": [1, 2, 3]}
    btrex_fitted = BellatrexExplain(clf, set_up="auto", p_grid=test_grid, verbose=3).fit(X_train, y_train)

    return btrex_fitted, X_test


# --- Core (non-GUI) test ---
def test_core_workflow():
    for loader in DATA_LOADERS:
        btrex_fitted, X_test = prepare_fitted_bellatrex(loader)
        for i in range(MAX_TEST_SAMPLES):
            tuned_method = btrex_fitted.explain(X_test, i)
            tuned_method.plot_overview(show=not IS_CI, plot_gui=False)
            # TODO: test create_rules_txt() method (do not store files)

# --- GUI test with plot_gui=True ---
@pytest.mark.gui
def test_gui_workflow():
    if IS_CI:
        matplotlib.use("Agg")  # Non-blocking backend when running in CI

    for loader in DATA_LOADERS:
        btrex_fitted, X_test = prepare_fitted_bellatrex(loader)

        for i in range(MAX_TEST_SAMPLES):
            tuned_method = btrex_fitted.explain(X_test, i)
            tuned_method.plot_overview(show=not IS_CI, plot_gui=True, auto_close=True)
            plt.close("all")
