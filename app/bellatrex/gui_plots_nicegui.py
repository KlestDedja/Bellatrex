"""
NiceGUI-based interactive scatter + tree viewer for Bellatrex.

Replaces the legacy DearPyGui implementation (gui_plots_code.py).

Key improvements over the DearPyGui backend:
- No temporary PNG files on disk; Matplotlib figures are passed directly.
- Works both as a native desktop window (native=True, via pywebview)
  and in any web browser (native=False) — same code, zero changes.
- Click interaction is handled by Plotly's native plotly_click event,
  bound to a Python handler via NiceGUI's WebSocket bridge.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from nicegui import ui

from .utilities import colormap_from_str
from .plot_tree_patch import plot_tree_patched
from .utilities import rule_print_inline, custom_axes_limit


# ---------------------------------------------------------------------------
# Internal data models (mirror the old InteractPoint / InteractPlot)
# ---------------------------------------------------------------------------

class InteractPoint:
    """Holds per-point metadata used while building the scatter and the tree viewer."""

    def __init__(self, name, pos, color, size, shape, cluster_memb=None, value=None):
        self.name = str(name)          # original tree index (string)
        self.pos = pos                 # (x, y) in PCA space
        self.color = color             # "rgba(r,g,b,a)" string for Plotly
        self.size = size               # marker size in pixels
        self.shape = shape             # "circle" | "star"
        self.cluster_memb = cluster_memb  # "1", "2", … or None
        self.value = value             # formatted prediction/loss string or None


class InteractPlot:
    """Holds all points for one scatter panel."""

    def __init__(self, name, points, clustered=False, xlabel="PC1", ylabel="PC2"):
        self.name = str(name)
        self.points = points
        self.clustered = clustered
        self.xlabel = xlabel
        self.ylabel = ylabel


# ---------------------------------------------------------------------------
# Plotly discrete-colour helpers
# ---------------------------------------------------------------------------

# Modest palette that works for up to ~12 clusters.
_CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


def _mpl_rgba_to_plotly(mpl_rgba):
    """Convert a Matplotlib 4-tuple (0-1 floats) to a Plotly rgba() CSS string."""
    r, g, b, a = [int(v * 255) for v in mpl_rgba]
    return f"rgba({r},{g},{b},{a})"


def _cluster_colour_for_index(cluster_idx: int) -> str:
    """Return a hex colour string for a 0-based cluster index."""
    return _CLUSTER_PALETTE[cluster_idx % len(_CLUSTER_PALETTE)]


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------

def _build_plotly_scatter(interact_plot: InteractPlot) -> go.Figure:
    """
    Convert an InteractPlot to a Plotly Figure.

    Each tree is a single Scatter point.  customdata[0] stores the original
    tree index so the click handler can retrieve it without additional look-ups.
    """
    points = interact_plot.points

    # Plotly marker symbol mapping
    plotly_shape_map = {"circle": "circle", "star": "star"}

    xs = [p.pos[0] for p in points]
    ys = [p.pos[1] for p in points]
    colors = [p.color for p in points]
    sizes = [p.size for p in points]
    symbols = [plotly_shape_map.get(p.shape, "circle") for p in points]

    # Build hover text
    hover_texts = []
    for p in points:
        tt = f"Tree: {p.name}"
        if p.cluster_memb is not None:
            tt += f"<br>Cluster: {p.cluster_memb}"
        if p.value is not None:
            lbl = "Pred" if not interact_plot.clustered else "Value"
            tt += f"<br>{lbl}: {p.value}"
        hover_texts.append(tt)

    # customdata carries the tree index (as int) so the click handler can use it
    custom_data = [int(p.name) for p in points]

    scatter = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(
            color=colors,
            size=sizes,
            symbol=symbols,
            line=dict(color="black", width=0.5),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        customdata=custom_data,
    )

    title = "Cluster membership" if interact_plot.clustered else "Prediction / loss"
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title=title,
        xaxis_title=interact_plot.xlabel,
        yaxis_title=interact_plot.ylabel,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="closest",
        width=600,
        height=600,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Data preparation — returns list[InteractPlot] without side-effects
# ---------------------------------------------------------------------------

def _build_interact_plots(
    plot_data_bunch,
    kmeans,
    input_method,
    colormap=None,
    clusterplots=(True, False),
):
    """
    Replicate the colour/normalisation logic from the old plot_with_interface()
    but return only the list of InteractPlot objects with no GUI side-effects.
    """

    def _shaper(is_final):
        return "star" if is_final else "circle"

    def _sizer(is_final):
        return 16.0 if is_final else 9.0

    # 2-D PCA projection (same logic as the old DearPyGui file)
    pca = PCA(n_components=2).fit(plot_data_bunch.proj_data)
    plottable_data = pca.transform(plot_data_bunch.proj_data)  # (n_trees, 2)

    cluster_memb = kmeans.labels_
    final_ts_idx = input_method.final_trees_idx

    is_final_candidate = [
        plot_data_bunch.index[i] in final_ts_idx for i in range(len(plot_data_bunch.index))
    ]
    custom_sizes = list(map(_sizer, is_final_candidate))
    custom_shapes = list(map(_shaper, is_final_candidate))

    color_map_right = colormap_from_str(colormap)

    interact_plots = []

    for _plotindex, clustered in enumerate(clusterplots):

        if clustered:
            # Discrete colour per cluster using a fixed palette (more readable in Plotly than viridis)
            freqs = np.bincount(cluster_memb)
            if np.min(freqs) == 0:
                raise KeyError(
                    "Empty clusters detected; scatter and colorbar may differ in shade."
                )
            colours = [_cluster_colour_for_index(int(cluster_memb[i]))
                       for i in range(len(plot_data_bunch.index))]
        else:
            # Continuous colour by prediction or loss value
            plot_data_bunch.rf_pred = np.array(plot_data_bunch.rf_pred).squeeze()

            if isinstance(plot_data_bunch.rf_pred, float) or plot_data_bunch.rf_pred.size == 1:
                is_binary = plot_data_bunch.set_up == "binary"
                v_min, v_max = custom_axes_limit(
                    np.array(plot_data_bunch.pred).min(),
                    np.array(plot_data_bunch.pred).max(),
                    plot_data_bunch.rf_pred,
                    is_binary,
                )
                norm_preds = mpl.colors.BoundaryNorm(
                    np.linspace(v_min, v_max, 256), color_map_right.N
                )
                plot_data_bunch.pred = np.array(plot_data_bunch.pred).squeeze()
                norms = [norm_preds(float(plot_data_bunch.pred[i]))
                         for i in range(len(cluster_memb))]
            else:
                v_min, v_max = custom_axes_limit(
                    np.array(plot_data_bunch.loss).min(),
                    np.array(plot_data_bunch.loss).max(),
                    force_in=None,
                    is_binary=False,
                )
                norm_preds = mpl.colors.BoundaryNorm(
                    np.linspace(v_min, v_max, 256), color_map_right.N
                )
                norms = [norm_preds(plot_data_bunch.loss[i]) for i in range(len(cluster_memb))]

            colours = [_mpl_rgba_to_plotly(color_map_right(norms[i]))
                       for i in range(len(plot_data_bunch.index))]

        points = []
        for j, index in enumerate(plot_data_bunch.index):
            pt = InteractPoint(
                name=index,
                pos=tuple(plottable_data[j]),
                color=colours[j],
                size=custom_sizes[j],
                shape=custom_shapes[j],
            )
            if clustered:
                pt.cluster_memb = str(int(cluster_memb[j]) + 1)  # 1-indexed label
            else:
                try:
                    pred_val = plot_data_bunch.pred[j]
                    if isinstance(pred_val, float):
                        pt.value = f"{pred_val:.3f}"
                    else:
                        pt.value = f"{float(plot_data_bunch.loss[j]):.3f}"
                except (TypeError, AttributeError):
                    pt.value = None
            points.append(pt)

        interact_plots.append(InteractPlot(_plotindex, points, clustered=clustered))

    return interact_plots


# ---------------------------------------------------------------------------
# NiceGUI layout
# ---------------------------------------------------------------------------

def _build_ui(interact_plots, input_method, max_depth, sample_index):
    """
    Declare the NiceGUI page layout.

    Called inside the @ui.page("/") handler so that all ui.* calls happen in
    the right async context.  No blocking here — ui.run() is called by the caller.
    """

    # Build Plotly figures once (static per session; only the tree dialog is dynamic)
    plotly_figs = [_build_plotly_scatter(ip) for ip in interact_plots]

    # Shared dialog for tree display
    tree_dialog = ui.dialog().props("maximized=false persistent=false")

    def _show_tree(tree_index: int):
        """Render the selected tree directly into a ui.matplotlib element (no temp files).

        plot_tree_patched() accepts an ``ax`` keyword, so we pass NiceGUI's own
        axes directly — no intermediate figure creation or axis-transfer needed.
        """
        my_clf = input_method.clf
        feature_names = (
            my_clf.feature_names_in_
            if hasattr(my_clf, "feature_names_in_")
            else [f"X{i}" for i in range(my_clf.n_features_in_)]
        )
        the_tree = my_clf[tree_index]
        rule_print_inline(the_tree, input_method.sample)

        if max_depth is not None:
            real_plot_leaves = max(the_tree.tree_.n_leaves, 2 ** (max_depth - 1))
            real_plot_depth = min(the_tree.tree_.max_depth, max_depth)
        else:
            real_plot_leaves = the_tree.tree_.n_leaves
            real_plot_depth = the_tree.tree_.max_depth

        smart_width = max(8, int(1 + 0.4 * real_plot_leaves))
        smart_height = max(4, int(real_plot_depth + 1))
        if my_clf.n_outputs_ > 3:
            smart_height = int(smart_height * (0.92 + 0.08 * my_clf.n_outputs_))

        with tree_dialog:
            tree_dialog.clear()
            with ui.card().style("max-width: 95vw; max-height: 90vh; overflow: auto;"):
                ui.label(f"Tree {tree_index} — sample {sample_index}").classes("text-h6")
                # ui.matplotlib creates a NiceGUI-managed figure; we draw into it
                # using the ax= parameter so no figure is created externally.
                mpl_elem = ui.matplotlib(figsize=(smart_width, smart_height))
                with mpl_elem.figure as fig:
                    ax = fig.add_subplot(111)
                    plot_tree_patched(
                        the_tree,
                        max_depth=max_depth,
                        feature_names=feature_names,
                        fontsize=8,
                        ax=ax,
                    )
                    ax.set_title(
                        f"Tree {tree_index} for sample index {sample_index}",
                        fontsize=10 + int(1.3 * real_plot_depth),
                    )
                    fig.tight_layout()
                ui.button("Close", on_click=tree_dialog.close).classes("mt-2")
        tree_dialog.open()

    def _on_plotly_click(event_args, _plot_index: int):
        """Handle Plotly click events — extract tree index from customdata and open dialog."""
        try:
            points = event_args.args.get("points", [])
            if not points:
                return
            pt = points[0]
            # customdata holds the original tree index stored as an int at build time
            tree_index = int(pt.get("customdata", pt.get("pointIndex", 0)))
            _show_tree(tree_index)
        except Exception as exc:
            ui.notify(f"Click handler error: {exc}", color="negative")

    # Page heading
    ui.label(f"Bellatrex — explaining sample {sample_index}").classes("text-h5 q-pa-md")
    ui.label(
        "Click a point on either scatter plot to view the corresponding decision tree."
    ).classes("text-caption q-px-md")

    # Side-by-side scatter panels
    with ui.row().classes("q-pa-md items-start"):
        for idx, (ip, pfig) in enumerate(zip(interact_plots, plotly_figs)):
            with ui.card().classes("q-ma-sm"):
                label_text = "Cluster colouring" if ip.clustered else "Prediction / loss colouring"
                ui.label(label_text).classes("text-subtitle1 text-center")
                scatter_elem = ui.plotly(pfig).style("width:580px; height:580px;")
                # Capture idx in the closure with a default-argument idiom
                scatter_elem.on(
                    "plotly_click",
                    lambda e, i=idx: _on_plotly_click(e, i),
                )


# ---------------------------------------------------------------------------
# Public API — same signature as the old plot_with_interface() in gui_plots_code.py
# ---------------------------------------------------------------------------

def plot_with_interface(
    plot_data_bunch,
    kmeans,
    input_method,  # fitted TreeExtraction / BellatrexExplain instance
    temp_files_dir=None,  # kept for API compatibility; not used (no temp PNG files)
    max_depth=None,
    colormap=None,
    clusterplots=(True, False),
    native=True,
):
    """
    Launch the NiceGUI interactive interface.

    Replaces the DearPyGui make_interactive_plot() call and the DPG event loop.

    Parameters
    ----------
    plot_data_bunch : sklearn.utils.Bunch
        Pre-processed tree projections and predictions from
        TreeExtraction.preselect_represent_cluster_trees().
    kmeans : fitted KMeans
        Cluster assignments for the projected trees.
    input_method : TreeExtraction
        Fitted extractor carrying .clf, .sample, .final_trees_idx, etc.
    temp_files_dir : str or None
        Ignored — kept only for backwards-compatible call sites.
    max_depth : int or None
        Maximum depth shown when rendering individual decision trees.
    colormap : str or Colormap or None
        Colormap used for the prediction / loss scatter panel.
    clusterplots : tuple of bool
        Which panels to show: (cluster-coloured, prediction-coloured).
    native : bool
        If True (default) open a native desktop window (requires pywebview/CEF).
        Set to False to open in the system browser instead.

    Returns
    -------
    list of InteractPlot
        The data objects backing the two scatter panels — mirrors the old
        DearPyGui return value for any callers that inspect the result.
    """
    interact_plots = _build_interact_plots(
        plot_data_bunch, kmeans, input_method, colormap=colormap, clusterplots=clusterplots
    )

    sample_index = input_method.sample.index[0]

    # Register the page builder.  All ui.* calls must happen inside the page
    # handler (or a startup task) so they run in the NiceGUI event loop.
    @ui.page("/")
    def _page():
        _build_ui(interact_plots, input_method, max_depth, sample_index)

    # ui.run() is blocking — it starts the web server and the window, and only
    # returns when the window is closed (native=True) or the process is killed.
    # This mirrors the behaviour of dpg.start_dearpygui() in the old code.
    ui.run(
        native=native,
        title=f"Bellatrex — sample {sample_index}",
        reload=False,
        show=True,
    )

    return interact_plots
