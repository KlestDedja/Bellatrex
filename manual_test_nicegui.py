import os
import sys

# Ensure package import from local `app` directory
sys.path.insert(0, os.path.join(os.getcwd(), "app"))

from bellatrex import BellatrexExplain, pack_trained_ensemble
from bellatrex.datasets import load_binary_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Starting NiceGUI smoke test (non-blocking)...")

# Small dataset and quick model to keep runtime short
X, y = load_binary_data(return_X_y=True)
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RandomForestClassifier(n_estimators=10, random_state=0)
clf.fit(X_train, y_train)

clf_packed = pack_trained_ensemble(clf)

# Use a tiny p_grid to speed up the explainer
b = BellatrexExplain(clf_packed, set_up="auto", p_grid={"n_clusters": [1, 2]}, verbose=0).fit(
    X_train, y_train
)

tuned = b.explain(X_test, 0)

print("Prepared tuned explainer — building plot objects...")

from bellatrex import gui_plots_nicegui as gui

# Stub ui.run to avoid starting a blocking NiceGUI server during the smoke test
try:
    gui.ui.run = lambda *a, **k: print(
        "ui.run() stubbed; NiceGUI server not started for smoke test"
    )
except Exception:
    # If ui not available yet, the import will fail earlier
    pass

# Build the kmeans + data objects used by the GUI
# Note: preselect_represent_cluster_trees() is on the internal TreeExtraction object
plot_kmeans, plot_data_bunch = tuned.tuned_method.preselect_represent_cluster_trees()

# Build the interact plots (no server required)
interact_plots = gui._build_interact_plots(plot_data_bunch, plot_kmeans, tuned.tuned_method)

# Convert one interact plot to a Plotly figure to verify rendering code
fig = gui._build_plotly_scatter(interact_plots[0])

print("Built Plotly figure: type=", type(fig))

# Call the public plot_with_interface (ui.run is stubbed, so it will return immediately)
gui.plot_with_interface(plot_data_bunch, plot_kmeans, tuned.tuned_method, native=False)

print("NiceGUI smoke test completed — UI call stubbed.")
