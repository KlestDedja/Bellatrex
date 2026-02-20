import os
import numpy as np
import pandas as pd
import bellatrex
import matplotlib.pyplot as plt
import joblib

from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from bellatrex import BellatrexExplain, pack_trained_ensemble, predict_helper
from bellatrex.datasets import load_mtr_data, load_mlc_data
from bellatrex.datasets import load_survival_data, load_binary_data, load_regression_data
from bellatrex.utilities import get_auto_setup

print("Bellatrex version:", bellatrex.__version__)
print("Working directory:", os.getcwd())

PLOT_GUI = True

# Uncomment the dataset that matches the prediction task you want to explore:
X, y = load_binary_data(return_X_y=True)  # binary classification
# X, y = load_regression_data(return_X_y=True)  # regression
# X, y = load_survival_data(return_X_y=True)    # survival analysis
# X, y = load_mlc_data(return_X_y=True)         # multi-label classification
# X, y = load_mtr_data(return_X_y=True)         # multi-target regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# --- Step 1: Train a Random Forest ------------------------------------------

SETUP = get_auto_setup(y)
print("Detected prediction task 'SETUP':", SETUP)

if SETUP.lower() == "survival":
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


# --- Step 2: Pack or load the trained model (optional) ----------------------

# The pre-trained model may be stored under app/bellatrex/datasets/model_example.pkl
# To save your own model, uncomment and adjust the line below:
# joblib.dump(clf, os.path.join('app', 'bellatrex', 'datasets', 'model_example.pkl'))

model_path = os.path.join("app", "bellatrex", "datasets", "model_example.pkl")
if os.path.exists(model_path):
    clf = joblib.load(model_path)
    print(f"Loaded pre-trained model from {model_path}")
else:
    # No pre-trained model found: use the fitted clf from Step 1
    print("No pre-trained model found; using the freshly fitted model.")

# pack_trained_ensemble converts the fitted forest into a memory-efficient dictionary.
# Pass clf_packed (or the original clf) to BellatrexExplain – both are supported.
clf_packed = pack_trained_ensemble(clf)
print(f"Packed {clf_packed['ensemble_class']} with {len(clf_packed['trees'])} trees.")


# --- Step 3: Fit Bellatrex and explain predictions --------------------------

Btrex_fitted = BellatrexExplain(
    clf_packed, set_up="auto", p_grid={"n_clusters": [1, 2, 3]}, verbose=1
).fit(X_train, y_train)

# Pre-compute training predictions once, used as background distribution in plot_visuals
y_train_pred = predict_helper(clf, X_train)

N_TEST_SAMPLES = 3
for i in range(N_TEST_SAMPLES):
    print(f"\n--- Explaining sample i={i} ---")

    tuned_method = Btrex_fitted.explain(X_test, i)

    # Plot 1: cluster overview (shows pre-selected trees and selected rules)
    fig1, axs1 = tuned_method.plot_overview(plot_gui=False, show=False)
    plt.show(block=True)

    # Plot 2: rule-level detail (single-output tasks only)
    if SETUP.lower() in ["binary", "survival", "regression"]:
        fig2, axs2 = tuned_method.plot_visuals(
            plot_max_depth=5, preds_distr=y_train_pred, conf_level=0.9, tot_digits=4, show=False
        )
        plt.show(block=True)
