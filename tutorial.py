import numpy as np
import pandas as pd
import os
import bellatrex as btrex

print("Bellatrex version:", btrex.__version__)

PLOT_GUI = False

##########################################################################
root_folder = os.getcwd()
print(root_folder)

from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from bellatrex import BellatrexExplain
from bellatrex.utilities import predict_helper
from bellatrex.wrapper_class import pack_trained_ensemble

import matplotlib.pyplot as plt
from bellatrex.datasets import load_mtr_data, load_mlc_data
from bellatrex.datasets import load_survival_data, load_binary_data, load_regression_data
from bellatrex.utilities import get_auto_setup

X, y = load_binary_data(return_X_y=True)
# X, y = load_regression_data(return_X_y=True)
# X, y = load_survival_data(return_X_y=True)
# X, y = load_mlc_data(return_X_y=True)
# X, y = load_mtr_data(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


SETUP = get_auto_setup(
    y
)  # not necessary, but comfortable while swithcing between mnay prediction tasks
print("Detected prediction task 'SETUP':", SETUP)


### instantiate original R(S)F estimator, works best with some pruning.
if SETUP.lower() in "survival":
    clf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, n_jobs=-2, random_state=0)

elif SETUP.lower() in ["binary", "multi-label"]:
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0)

elif SETUP.lower() in ["regression", "multi-target"]:
    clf = RandomForestRegressor(n_estimators=100, min_samples_split=5, n_jobs=-2, random_state=0)
else:
    clf = None
    raise ValueError(f"Unknown prediction task SETUP={SETUP}.")

clf.fit(X_train, y_train)
print("Model fitting complete.")


# Pretrained RF model should be packed as a list of dicts with the function below.
clf_packed = pack_trained_ensemble(clf)


# fit RF here. The hyperparameters for fitting the explanation are given
# compatible with trained ensemble model clf, and with packed dictionary as in clf_packed
Btrex_fitted = BellatrexExplain(
    clf_packed, set_up="auto", p_grid={"n_clusters": [1, 2, 3]}, verbose=3
).fit(X_train, y_train)

N_TEST_SAMPLES = 3
for i in range(N_TEST_SAMPLES):

    print(f"Explaining sample i={i}")

    y_train_pred = predict_helper(
        clf, X_train
    )  # calls, predict or predict_proba, depending on the underlying model

    tuned_method = Btrex_fitted.explain(X_test, i)

    fig1, axs1 = tuned_method.plot_overview(plot_gui=False, show=False)
    plt.show(block=True)

    if SETUP.lower() in ["binary", "survival", "regression"]:

        fig2, axs2 = tuned_method.plot_visuals(
            plot_max_depth=5, preds_distr=y_train_pred, conf_level=0.9, tot_digits=4, show=False
        )
        plt.show(block=True)
