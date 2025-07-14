'''
Author: Klest Dedja
Here we manually test most of the features
'''

import os
import pytest
import matplotlib.pyplot as plt

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
from bellatrex.utilities import get_auto_setup, predict_helper

@pytest.mark.gui  # tag this test as GUI-dependent
def test_gui_workflow():
    print("Bellatrex version:", btrex.__version__)

    MAX_TEST_SAMPLES = 2
    PLOT_GUI = True
    root_folder = os.getcwd()

    # Load dataset
    X, y = load_mlc_data(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    SETUP = get_auto_setup(y)
    print("Detected prediction task 'SETUP':", SETUP)

    if SETUP.lower() in "survival":
        clf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, n_jobs=-2, random_state=0)
    elif SETUP.lower() in ["binary", "multi-label"]:
        clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0)
    elif SETUP.lower() in ["regression", "multi-target"]:
        clf = RandomForestRegressor(n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0)
    else:
        raise ValueError(f"Detection task {SETUP} not compatible with Bellatrex (yet)")

    clf.fit(X_train, y_train)
    print("Model fitting complete.")
    clf_packed = pack_trained_ensemble(clf)

    Btrex_fitted = BellatrexExplain(clf, set_up="auto", p_grid={"n_clusters": [1, 2, 3]}, verbose=3).fit(X_train, y_train)

    for i in range(MAX_TEST_SAMPLES):
        print(f"Explaining sample i={i}")
        explan_dir = os.path.join(root_folder, "explanations-out")
        os.makedirs(explan_dir, exist_ok=True)
        FILE_OUT = os.path.join(explan_dir, f"Rules_{SETUP}_id{i}.txt")

        y_train_pred = predict_helper(clf, X_train)

        tuned_method = Btrex_fitted.explain(X_test, i)
        tuned_method.plot_overview(show=True, plot_gui=PLOT_GUI)

        # You can add assertions here for specific outputs, files created, etc.
        plt.show()
