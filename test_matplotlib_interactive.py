"""
Test the Matplotlib interactive click handler (non-GUI fallback).

Automatically simulates a click on a scatter point and verifies that a
tree figure opens. Run with:  python test_matplotlib_interactive.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "app"))

import matplotlib
matplotlib.use("Agg")   # headless – no GUI window needed

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from bellatrex import BellatrexExplain, pack_trained_ensemble
from bellatrex.datasets import load_binary_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Loading data and training a small Random Forest...")
X, y = load_binary_data(return_X_y=True)
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RandomForestClassifier(n_estimators=20, random_state=0)
clf.fit(X_train, y_train)

clf_packed = pack_trained_ensemble(clf)

print("Fitting Bellatrex explainer...")
b = BellatrexExplain(
    clf_packed,
    set_up="auto",
    p_grid={"n_clusters": [1, 2]},
    verbose=1,
).fit(X_train, y_train)

print("Explaining test sample 0...")
tuned = b.explain(X_test, 0)

# ── Build the overview plot with click-handler attached ──────────────────────
fig, axes = tuned.plot_overview(plot_gui=False, show=False)

print("\n" + "=" * 60)
print("Simulating a programmatic click on the nearest scatter point...")

# Extract the first plotted scatter point's data coordinates from axes[0]
scatter_col = axes[0].collections[0]
offsets = scatter_col.get_offsets()   # (N, 2) data coords
assert len(offsets) > 0, "No scatter points found in axes[0]"

target_x, target_y = float(offsets[0, 0]), float(offsets[0, 1])
print(f"  Clicking at data coords ({target_x:.4f}, {target_y:.4f})")

# Convert to display coordinates so the MouseEvent is well-formed
x_disp, y_disp = axes[0].transData.transform((target_x, target_y))

n_figs_before = len(plt.get_fignums())

event = MouseEvent("button_press_event", fig.canvas, x=x_disp, y=y_disp, button=1)
event.inaxes = axes[0]
event.xdata  = target_x
event.ydata  = target_y

fig.canvas.callbacks.process("button_press_event", event)

n_figs_after = len(plt.get_fignums())

if n_figs_after > n_figs_before:
    print(f"  ✓  Tree plot opened! (figures before={n_figs_before}, after={n_figs_after})")
    print("Click simulation PASSED.")
else:
    print(f"  ✗  No new figure opened (before={n_figs_before}, after={n_figs_after})")
    sys.exit(1)
