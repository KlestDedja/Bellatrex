import argparse
import io
import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
from uuid import uuid4

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA

from .plot_tree_patch import plot_tree_patched
from .utilities import colormap_from_str
from .utilities import custom_axes_limit, custom_formatter
from .utilities import rule_print_inline


class InteractPoint:  # Object containg all information of a point
    def __init__(self, name, pos, color, size, shape, cluster_memb=None, value=None):
        self.name = str(name)
        self.pos = pos
        self.color = color
        self.size = size
        self.shape = shape
        self.cluster_memb = cluster_memb
        self.value = value


class InteractPlot:
    def __init__(self, name, points, clustered=False, xlabel="PC1", ylabel="PC2"):
        self.name = str(name)
        self.points = points
        self.clustered = clustered
        self.xlabel = xlabel
        self.ylabel = ylabel


def _feature_names_for_clf(clf) -> list[str]:
    if hasattr(clf, "feature_names_in_") and clf.feature_names_in_ is not None:
        return list(clf.feature_names_in_)
    return [f"X{i}" for i in range(clf.n_features_in_)]


def _render_tree_image(tree_index, render_context: dict) -> tuple[bytes, str]:
    clf = render_context["clf"]
    sample = render_context["sample"]
    sample_index = render_context.get("sample_index", sample.index[0])
    max_depth = render_context.get("max_depth")
    feature_names = render_context.get("feature_names") or _feature_names_for_clf(clf)

    tree_index = int(tree_index)
    the_tree = clf[tree_index]
    rule_print_inline(the_tree, sample)

    if max_depth is not None:
        real_plot_leaves = max(the_tree.tree_.n_leaves, 2 ** (max_depth - 1))
        real_plot_depth = min(the_tree.tree_.max_depth, max_depth)
    else:
        real_plot_leaves = the_tree.tree_.n_leaves
        real_plot_depth = the_tree.tree_.max_depth

    # Width driven by leaves (at least 10 in), height driven by depth (at least 4 in).
    # Enforce width >= 2 * height so shallow trees stay wide rather than tall.
    smart_width = max(10, 1.0 * real_plot_leaves)
    smart_height = max(4, 1.5 * (real_plot_depth + 1))
    if clf.n_outputs_ > 3:
        smart_height = smart_height * (0.92 + 0.08 * clf.n_outputs_)
    smart_width = max(smart_width, 2 * smart_height)

    figure, axis = plt.subplots(figsize=(smart_width, smart_height), dpi=90)
    plot_tree_patched(
        the_tree,
        max_depth=max_depth,
        feature_names=feature_names,
        fontsize=8,
        ax=axis,
    )
    title = f"Tree {tree_index} for sample index {sample_index}"
    axis.set_title(title, fontsize=10 + int(1.3 * real_plot_depth))
    figure.tight_layout()

    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, bbox_inches="tight", format="png")
    plt.close(figure)
    return image_buffer.getvalue(), title


def _detect_native_window_support() -> bool:
    return importlib.util.find_spec("webview") is not None


def _build_colorbar_paths(plots, temp_files_dir: str) -> list[str]:
    return [
        os.path.abspath(os.path.join(temp_files_dir, f"temp_colourbar{i}.png"))
        for i in range(len(plots))
    ]


def _build_subprocess_payload(
    plots, colorbar_paths, render_context: dict | None, native: bool, port: int, temp_files_dir: str
) -> str:
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".bellatrex-gui.pkl", delete=False, dir=temp_files_dir
    ) as handle:
        pickle.dump(
            {
                "plots": plots,
                "colorbar_paths": colorbar_paths,
                "render_context": render_context,
                "native": native,
                "port": port,
                "temp_files_dir": temp_files_dir,
            },
            handle,
        )
        return handle.name


def _run_subprocess_app(payload_path: str, blocking: bool) -> None:
    command = [sys.executable, "-m", "bellatrex.nicegui_plots_code", "--payload", payload_path]
    process = subprocess.Popen(command)

    if blocking:
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError(f"NiceGUI window process exited with code {exit_code}.")


def _serve_payload_file(payload_path: str) -> None:
    with open(payload_path, "rb") as handle:
        payload = pickle.load(handle)

    _run_nicegui_app(
        payload["plots"],
        payload["colorbar_paths"],
        payload.get("render_context"),
        native=payload["native"],
        port=payload["port"],
        temp_files_dir=payload["temp_files_dir"],
        payload_path=payload_path,
    )


def plot_with_interface(
    plot_data_bunch,
    kmeans,
    input_method,  # a fitted bellatrex instance
    temp_files_dir,
    max_depth=None,
    colormap=None,
    clusterplots=(True, False),
):
    """Prepare plot data for the NiceGUI interactive frontend.

    Builds the same ``plots`` structure used internally for interactive display
    but does not create any GUI context. Call this before launching the NiceGUI
    server.
    """
    _ = max_depth

    def shaper(in_shape):
        return "star" if in_shape is True else "circle"

    def sizer(in_size):
        return 16.0 if in_size is True else 9.0

    def rgbaconv(mpl_rgba):
        return [i * 255 for i in mpl_rgba]

    PCA_fitted = PCA(n_components=2).fit(plot_data_bunch.proj_data)
    plottable_data = PCA_fitted.transform(plot_data_bunch.proj_data)

    cluster_memb = kmeans.labels_

    final_ts_idx = input_method.final_trees_idx

    is_final_candidate = [
        plot_data_bunch.index[i] in final_ts_idx for i in range(len(plot_data_bunch.index))
    ]
    custom_sizes = list(map(sizer, is_final_candidate))
    custom_shapes = list(map(shaper, is_final_candidate))

    plots = []

    color_map_left = colormaps["viridis"]
    color_map_left = LinearSegmentedColormap.from_list(
        "Custom cmap", [color_map_left(i) for i in range(color_map_left.N)], color_map_left.N
    )

    color_map_right = colormap_from_str(colormap)

    for plotindex, clustered in enumerate(clusterplots):

        if clustered:
            fig, ax = plt.subplots(1, 1, figsize=(1, 4.5), dpi=120)

            freqs = np.bincount(cluster_memb)
            if np.min(freqs) == 0:
                raise KeyError(
                    "There are empty clusters, the scatter and colorbar could differ in color shade"
                )
            norm_bins = list(np.cumsum(freqs))
            norm_bins.insert(0, 0)

            norm_bins = np.array(norm_bins)

            labels = []
            for i in np.unique(cluster_memb):
                labels.append("cl.{:d}".format(i + 1))

            norm = BoundaryNorm(norm_bins, color_map_left.N)
            tickz = (norm_bins[:-1] + (norm_bins[1:] - norm_bins[:-1]) / 2).tolist()

            cb1 = Colorbar(
                ax,
                cmap=color_map_left,
                norm=norm,
                spacing="proportional",
                ticks=tickz,
                boundaries=norm_bins.tolist(),
                format="%1i",
            )
            cb1.ax.set_yticklabels(labels)

            ax.yaxis.set_ticks_position("left")
            norms = [norm(norm_bins[cluster_memb[i]]) for i in range(len(cluster_memb))]

        else:
            fig, ax = plt.subplots(1, 1, figsize=(1.2, 4.5), dpi=110)

            if isinstance(plot_data_bunch.rf_pred, float) or plot_data_bunch.rf_pred.size == 1:

                is_binary = plot_data_bunch.set_up == "binary"

                plot_data_bunch.rf_pred = np.array(plot_data_bunch.rf_pred).squeeze()

                v_min, v_max = custom_axes_limit(
                    np.array(plot_data_bunch.pred).min(),
                    np.array(plot_data_bunch.pred).max(),
                    plot_data_bunch.rf_pred,
                    is_binary,
                )

                norm_preds = BoundaryNorm(np.linspace(v_min, v_max, 256), color_map_right.N)

                pred_tick = np.round(plot_data_bunch.rf_pred, 3)

                cb2 = Colorbar(
                    ax, cmap=color_map_right, norm=norm_preds, label="RF pred: " + str(pred_tick)
                )

                plot_data_bunch.pred = np.array(plot_data_bunch.pred).squeeze()

                cb2.ax.plot([0, 1], [plot_data_bunch.pred] * 2, color="grey", linewidth=1)
                cb2.ax.plot([0.02, 0.98], [pred_tick] * 2, color="black", linewidth=2.5, marker="P")

            else:
                color_map_right = colormap_from_str(colormap)

                v_min, v_max = custom_axes_limit(
                    np.array(plot_data_bunch.loss).min(),
                    np.array(plot_data_bunch.loss).max(),
                    force_in=None,
                    is_binary=False,
                )

                norm_preds = BoundaryNorm(np.linspace(v_min, v_max, 256), color_map_right.N)

                cb2 = Colorbar(
                    ax,
                    cmap=color_map_right,
                    norm=norm_preds,
                    label=str(input_method.fidelity_measure) + " loss",
                )
                cb2.ax.plot([0, 1], [plot_data_bunch.loss] * 2, color="grey", linewidth=1)

            ticks_to_plot = ax.get_yticks()

            if np.abs(np.min(ticks_to_plot)) < 1e-3 and np.abs(np.max(ticks_to_plot)) > 1e-2:
                min_index = np.argmin(ticks_to_plot)
                ticks_to_plot[min_index] = 0
                ax.set_yticks(ticks_to_plot)

            ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
            ax.minorticks_off()

            if isinstance(plot_data_bunch.rf_pred, float) or plot_data_bunch.rf_pred.size == 1:
                norms = [
                    norm_preds(float(plot_data_bunch.pred[i])) for i in range(len(cluster_memb))
                ]
            else:
                norms = [norm_preds(plot_data_bunch.loss[i]) for i in range(len(cluster_memb))]

        fig.tight_layout()
        fig.savefig(os.path.join(temp_files_dir, f"temp_colourbar{plotindex}.png"))
        plt.close(fig)

        cmap_gui = color_map_left if clustered else color_map_right

        colours = [rgbaconv(cmap_gui(norms[i])) for i in range(len(plot_data_bunch.index))]

        points = []
        for j, index in enumerate(plot_data_bunch.index):
            points.append(
                InteractPoint(
                    index,
                    plottable_data[j],
                    colours[j],
                    custom_sizes[j],
                    custom_shapes[j],
                )
            )

            if clustered:
                points[j].cluster_memb = str(cluster_memb[j] + 1)
            else:
                if isinstance(plot_data_bunch.pred[j], float):
                    points[j].value = f"{plot_data_bunch.pred[j]:.3f}"
                elif isinstance(plot_data_bunch.loss[j], float):
                    points[j].value = f"{plot_data_bunch.loss[j]:.3f}"
                else:
                    raise ValueError("expecting float, got {type(plot_data_bunch.loss)} instead")

        plots.append(InteractPlot(plotindex, points, clustered=clustered))

    # NOTE: colorbar PNGs are kept here so the NiceGUI window can display them.
    # launch_nicegui_window() is responsible for deleting them on shutdown.
    return plots


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cleanup_temp_artifacts(
    temp_files_dir: str,
    colorbar_paths: list[str] | None = None,
    dialog_image_paths: list[str] | None = None,
    payload_path: str | None = None,
) -> None:
    paths_to_remove = set(colorbar_paths or [])
    paths_to_remove.update(dialog_image_paths or [])
    if payload_path:
        paths_to_remove.add(payload_path)

    if os.path.isdir(temp_files_dir):
        for entry in os.listdir(temp_files_dir):
            if (
                entry.startswith("temp_colourbar")
                and entry.endswith(".png")
                or entry.startswith("tree_")
                and entry.endswith(".png")
                or entry.endswith(".bellatrex-gui.pkl")
            ):
                paths_to_remove.add(os.path.join(temp_files_dir, entry))

    for path in paths_to_remove:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _run_nicegui_app(
    plots,
    colorbar_paths,
    render_context: dict | None,
    native: bool,
    port: int,
    temp_files_dir: str,
    payload_path: str | None = None,
) -> None:
    import plotly.graph_objects as go
    from nicegui import app as ng_app
    from nicegui import ui

    # Serve temporary render assets (tree PNGs and colorbars) via HTTP paths.
    ng_app.add_static_files("/bellatrex_tmp", temp_files_dir)
    dialog_image_paths: list[str] = []

    @ui.page("/")
    def main_page() -> None:
        ui.label("Bellatrex Explorer").classes("text-2xl font-bold p-4")
        info_label = ui.label("Click on a point to open the corresponding tree").classes(
            "text-sm italic text-gray-500 px-4 pb-2"
        )

        tree_dialog = ui.dialog()

        def _normalize_tree_name(raw_tree_name) -> str | None:
            """Return a scalar tree id from Plotly click payloads.

            Depending on Plotly/NiceGUI versions, ``customdata`` can be a scalar,
            a single-item list/tuple, or a dict-like value.
            """
            if raw_tree_name is None:
                return None

            tree_name = raw_tree_name
            if isinstance(tree_name, (list, tuple)):
                if not tree_name:
                    return None
                tree_name = tree_name[0]
            elif isinstance(tree_name, dict):
                # Pick the first non-None value deterministically.
                for value in tree_name.values():
                    if value is not None:
                        tree_name = value
                        break

            if tree_name is None:
                return None
            return str(tree_name)

        def _open_tree_dialog(tree_name: str, subtitle: str | None = None) -> None:
            if render_context is None:
                ui.notify("Tree rendering context is unavailable.", color="warning")
                return

            try:
                tree_png, title = _render_tree_image(tree_name, render_context)
            except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
                ui.notify(f"Could not render tree {tree_name}: {exc}", color="negative")
                return

            image_name = f"tree_{tree_name}_{uuid4().hex}.png"
            image_path = os.path.abspath(os.path.join(temp_files_dir, image_name))
            with open(image_path, "wb") as image_file:
                image_file.write(tree_png)
            if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                ui.notify(f"Could not write tree image for {tree_name}.", color="negative")
                return
            dialog_image_paths.append(image_path)
            tree_source = f"/bellatrex_tmp/{image_name}"

            tree_dialog.clear()
            with (
                tree_dialog,
                ui.card().style(
                    "width:95vw; max-width:none; height:92vh;"
                    "display:flex; flex-direction:column; gap:0.5rem; padding:1rem;"
                ),
            ):
                with ui.row().style(
                    "width:100%; display:flex; align-items:center;"
                    "justify-content:space-between; flex-shrink:0"
                ):
                    with ui.column().style("gap:0"):
                        ui.label(title).classes("text-lg font-semibold")
                        if subtitle:
                            ui.label(subtitle).classes("text-sm text-gray-500")
                        ui.link("Open image directly", tree_source, new_tab=True).classes("text-xs")
                    ui.button("Close", on_click=lambda: tree_dialog.close())
                # width:100% pins the div to the card boundary so the vertical
                # scrollbar always appears at the right edge of the visible card,
                # not at the far right of the (potentially very wide) image.
                # min-width:0 lets the flex child shrink below its content size.
                with ui.element("div").style(
                    "flex:1; min-height:0; min-width:0; width:100%; "
                    "overflow-x:auto; overflow-y:auto;"
                    "border:1px solid #e5e7eb; border-radius:4px"
                ):
                    ui.html(
                        f'<img src="{tree_source}" style="max-width:none; height:auto; display:block;" />'
                    )
            tree_dialog.open()

        # Outer page-level scroll: only kicks in when both pairs exceed the window width.
        with ui.element("div").style(
            "display:flex; flex-direction:row; align-items:flex-start; "
            "gap:2rem; overflow-x:auto; padding:0 2rem 1rem 1rem;"
        ):
            for idx, interactplot in enumerate(plots):
                # Each pair is a flat flex-row: [plot] [colorbar].
                # Both children are flex-shrink:0 so the colorbar is ALWAYS
                # visible right next to its plot without any nested scrolling.
                with ui.element("div").style(
                    "display:flex; flex-direction:row; align-items:flex-start; "
                    "gap:6px; flex-shrink:0;"
                ):
                    with ui.element("div").style("flex-shrink:0;"):
                        fig = go.Figure()

                        circle_pts = [pt for pt in interactplot.points if pt.shape != "star"]
                        star_pts = [pt for pt in interactplot.points if pt.shape == "star"]

                        def _add_trace(
                            pts: list,
                            symbol: str,
                            size: float,
                            trace_name: str,
                            figure,
                            clustered: bool,
                        ) -> None:
                            if not pts:
                                return
                            xs = [float(pt.pos[0]) for pt in pts]
                            ys = [float(pt.pos[1]) for pt in pts]
                            colors = [
                                "rgba({},{},{},{:.2f})".format(
                                    int(pt.color[0]),
                                    int(pt.color[1]),
                                    int(pt.color[2]),
                                    float(pt.color[3]) / 255.0,
                                )
                                for pt in pts
                            ]
                            if clustered:
                                texts = [
                                    f"Tree {pt.name} | Cluster {pt.cluster_memb}" for pt in pts
                                ]
                            else:
                                texts = [f"Tree {pt.name} | Pred {pt.value}" for pt in pts]
                            tree_ids = [pt.name for pt in pts]
                            figure.add_trace(
                                go.Scatter(
                                    x=xs,
                                    y=ys,
                                    mode="markers",
                                    marker=dict(
                                        symbol=symbol,
                                        size=size,
                                        color=colors,
                                        line=dict(width=1, color="DarkSlateGrey"),
                                    ),
                                    text=texts,
                                    customdata=tree_ids,
                                    hoverinfo="text",
                                    name=trace_name,
                                    showlegend=False,
                                )
                            )

                        def _add_legend_trace(
                            symbol: str,
                            size: float,
                            trace_name: str,
                            figure,
                            fill_color: str = "rgba(220,220,220,1.0)",
                        ) -> None:
                            figure.add_trace(
                                go.Scatter(
                                    x=[None],
                                    y=[None],
                                    mode="markers",
                                    marker=dict(
                                        symbol=symbol,
                                        size=size,
                                        color=fill_color,
                                        line=dict(width=1, color="black"),
                                    ),
                                    hoverinfo="skip",
                                    name=trace_name,
                                )
                            )

                        _add_trace(
                            circle_pts, "circle", 9, "Candidate trees", fig, interactplot.clustered
                        )

                        _add_trace(
                            star_pts, "star", 16, "Selected trees", fig, interactplot.clustered
                        )

                        _add_legend_trace("circle", 9, "Candidate trees", fig)
                        _add_legend_trace("star", 16, "Selected trees", fig)

                        plot_title = (
                            "Cluster colors" if interactplot.clustered else "Prediction colors"
                        )
                        fig.update_layout(
                            title=plot_title,
                            xaxis_title=interactplot.xlabel,
                            yaxis_title=interactplot.ylabel,
                            clickmode="event+select",
                            width=520,
                            height=430,
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                            margin=dict(l=40, r=20, t=50, b=40),
                        )

                        plot_elem = ui.plotly(fig)

                        def _on_plotly_click(event, lbl=info_label) -> None:
                            args = getattr(event, "args", None)
                            if isinstance(args, dict):
                                text = args.get("text") or "Point selected"
                                lbl.set_text(text)
                                tree_name = _normalize_tree_name(args.get("tree_name"))
                                if tree_name is not None:
                                    _open_tree_dialog(tree_name, text)

                        plot_elem.on(
                            "plotly_click",
                            _on_plotly_click,
                            js_handler="(event) => { const cd = event?.points?.[0]?.customdata; emit({tree_name: Array.isArray(cd) ? cd[0] : cd, text: event?.points?.[0]?.text}); }",
                        )

                    # Colorbar: direct sibling of the plot div, never inside a scroll container.
                    cb_path = colorbar_paths[idx] if idx < len(colorbar_paths) else None
                    if cb_path and os.path.exists(cb_path):
                        cb_source = (
                            f"/bellatrex_tmp/{os.path.basename(cb_path)}"
                            f"?v={int(os.path.getmtime(cb_path))}"
                        )
                        # Display the colorbar at 75% of the previous size; flex-shrink:0 keeps it visible.
                        ui.html(
                            f'<img src="{cb_source}" '
                            'style="height:240px; width:auto; max-width:82px; '
                            'margin-top:32px; display:block; flex-shrink:0;" />'
                        )
                    else:
                        ui.label("Colorbar missing").classes("text-xs text-red-600")

    def _cleanup() -> None:
        _cleanup_temp_artifacts(
            temp_files_dir,
            colorbar_paths=colorbar_paths,
            dialog_image_paths=dialog_image_paths,
            payload_path=payload_path,
        )

    ng_app.on_shutdown(_cleanup)

    # Initial native window size
    initial_window_size = (1360, 680)

    ui.run(
        native=native,
        port=port,
        reload=False,
        title="Bellatrex Explorer",
        show=not native,
        window_size=initial_window_size,
    )


def launch_nicegui_window(
    plots, temp_files_dir: str, blocking=None, render_context: dict | None = None
) -> None:
    """Open a NiceGUI window with two interactive scatter plots.

    Each point in the scatter can be clicked to see its label (cluster or
    prediction value). Star-shaped points are the final selected trees;
    circle-shaped points are candidates.

    Parameters
    ----------
    plots:
        List of :class:`InteractPlot` objects as returned by
        :func:`plot_with_interface`.
    temp_files_dir:
        Directory that holds the ``temp_colourbar*.png`` images produced by
        :func:`plot_with_interface`.
    blocking:
        ``True``  – block until the window is closed.
        ``False`` – return immediately (background process).
        ``None``  – defaults to blocking mode.
    """
    from multiprocessing import get_context

    if blocking is None:
        blocking = True

    native = _detect_native_window_support()
    colorbar_paths = _build_colorbar_paths(plots, temp_files_dir)
    port = _find_free_port()

    if sys.platform.startswith("win"):
        payload_path = _build_subprocess_payload(
            plots, colorbar_paths, render_context, native, port, temp_files_dir
        )
        _run_subprocess_app(payload_path, blocking)
        if blocking:
            _cleanup_temp_artifacts(
                temp_files_dir,
                colorbar_paths=colorbar_paths,
                payload_path=payload_path,
            )
        return

    ctx = get_context("spawn")
    proc = ctx.Process(
        target=_run_nicegui_app,
        args=(plots, colorbar_paths, render_context, native, port, temp_files_dir),
        daemon=False,
    )
    proc.start()

    if blocking:
        proc.join()
        _cleanup_temp_artifacts(temp_files_dir, colorbar_paths=colorbar_paths)
        if proc.exitcode not in (0, None):
            raise RuntimeError(f"NiceGUI window process exited with code {proc.exitcode}.")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bellatrex NiceGUI window.")
    parser.add_argument("--payload", required=True, help="Pickle payload generated by Bellatrex.")
    return parser.parse_args()


if __name__ == "__main__":
    _serve_payload_file(_parse_cli_args().payload)
