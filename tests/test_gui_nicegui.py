"""
Unit tests for the NiceGUI-based GUI backend (gui_plots_nicegui.py).

Strategy
--------
- _build_interact_plots and _build_plotly_scatter are pure functions with no
  GUI side-effects and are tested directly with synthetic data.
- plot_with_interface launches a web server (ui.run); this is patched out in
  tests that exercise the full call path without opening any window.
- All tests run headless (no display required).
"""

import re
import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from bellatrex.gui_plots_nicegui import (
    InteractPoint,
    InteractPlot,
    _build_interact_plots,
    _build_plotly_scatter,
    _mpl_rgba_to_plotly,
    _cluster_colour_for_index,
    plot_with_interface,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_TREES = 20
N_DIMS = 4
N_CLUSTERS = 3
RNG = np.random.default_rng(42)


def _make_plot_data_bunch(set_up="regression", multi_output=False):
    """Return a minimal Bunch-like object mimicking preselect_represent_cluster_trees output."""
    pred = RNG.random(N_TREES).tolist() if not multi_output else RNG.random((N_TREES, 2)).tolist()
    loss = RNG.random(N_TREES).tolist()
    rf_pred = RNG.random() if not multi_output else RNG.random(2)

    return SimpleNamespace(
        proj_data=RNG.random((N_TREES, N_DIMS)),
        index=np.arange(N_TREES),
        loss=np.array(loss),
        pred=np.array(pred) if not multi_output else np.array(pred),
        rf_pred=rf_pred,
        set_up=set_up,
    )


def _make_kmeans(n_trees=N_TREES, n_clusters=N_CLUSTERS):
    """Return a SimpleNamespace mimicking a fitted KMeans object."""
    # Deterministic balanced cluster assignment
    labels = np.array([i % n_clusters for i in range(n_trees)], dtype=np.int32)
    centers = RNG.random((n_clusters, N_DIMS))
    return SimpleNamespace(labels_=labels, cluster_centers_=centers)


def _make_input_method(final_count=3):
    """Return a minimal mock of TreeExtraction for _build_interact_plots."""
    final_trees_idx = list(range(final_count))  # first `final_count` trees are "final"
    return SimpleNamespace(final_trees_idx=final_trees_idx)


# ---------------------------------------------------------------------------
# Helper colour utilities
# ---------------------------------------------------------------------------


class TestColourHelpers:
    def test_mpl_rgba_to_plotly_black(self):
        result = _mpl_rgba_to_plotly((0.0, 0.0, 0.0, 1.0))
        assert result == "rgba(0,0,0,255)"

    def test_mpl_rgba_to_plotly_white(self):
        result = _mpl_rgba_to_plotly((1.0, 1.0, 1.0, 1.0))
        assert result == "rgba(255,255,255,255)"

    def test_mpl_rgba_format(self):
        result = _mpl_rgba_to_plotly((0.5, 0.25, 0.75, 0.8))
        assert result.startswith("rgba(")
        assert result.endswith(")")
        # Four comma-separated integers inside
        inner = result[len("rgba(") : -1]
        parts = inner.split(",")
        assert len(parts) == 4
        assert all(p.strip().isdigit() for p in parts)

    def test_cluster_colour_cycles(self):
        # Index beyond palette length should wrap without raising
        _cluster_colour_for_index(0)
        _cluster_colour_for_index(1000)
        _cluster_colour_for_index(11)
        _cluster_colour_for_index(12)  # wraps to index 0
        # Palette has 12 entries; index 12 == index 0
        assert _cluster_colour_for_index(12) == _cluster_colour_for_index(0)

    def test_cluster_colours_are_hex(self):
        hex_pat = re.compile(r"^#[0-9a-fA-F]{6}$")
        for idx in range(15):
            assert hex_pat.match(
                _cluster_colour_for_index(idx)
            ), f"Cluster colour {idx!r} is not a 6-digit hex string"


# ---------------------------------------------------------------------------
# InteractPoint and InteractPlot data models
# ---------------------------------------------------------------------------


class TestInteractPoint:
    def test_name_is_string(self):
        pt = InteractPoint(name=7, pos=(0.1, 0.2), color="#abc", size=9.0, shape="circle")
        assert pt.name == "7"

    def test_defaults_none(self):
        pt = InteractPoint(name=0, pos=(0.0, 0.0), color="#000", size=9.0, shape="circle")
        assert pt.cluster_memb is None
        assert pt.value is None

    def test_custom_attrs(self):
        pt = InteractPoint(
            name=3,
            pos=(1.0, -1.0),
            color="rgba(1,2,3,255)",
            size=16.0,
            shape="star",
            cluster_memb="2",
            value="0.453",
        )
        assert pt.shape == "star"
        assert pt.cluster_memb == "2"
        assert pt.value == "0.453"


class TestInteractPlot:
    def test_name_is_string(self):
        ip = InteractPlot(name=0, points=[])
        assert ip.name == "0"

    def test_default_axis_labels(self):
        ip = InteractPlot(name=0, points=[])
        assert ip.xlabel == "PC1"
        assert ip.ylabel == "PC2"


# ---------------------------------------------------------------------------
# _build_interact_plots — regression set-up (single output)
# ---------------------------------------------------------------------------


class TestBuildInteractPlotsRegression:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bunch = _make_plot_data_bunch(set_up="regression")
        self.kmeans = _make_kmeans()
        self.method = _make_input_method(final_count=3)

    def test_returns_two_plots(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        assert len(plots) == 2

    def test_first_plot_is_clustered(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        assert plots[0].clustered is True

    def test_second_plot_is_not_clustered(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        assert plots[1].clustered is False

    def test_correct_point_count(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        for ip in plots:
            assert len(ip.points) == N_TREES

    def test_final_trees_get_star(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        for ip in plots:
            for i, pt in enumerate(ip.points):
                expected = "star" if i < 3 else "circle"
                assert pt.shape == expected, f"Point {i}: expected {expected}, got {pt.shape}"

    def test_final_trees_larger(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        for ip in plots:
            for i, pt in enumerate(ip.points):
                if i < 3:
                    assert pt.size > 10
                else:
                    assert pt.size <= 10

    def test_clustered_points_have_cluster_memb(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        clustered_plot = plots[0]
        for pt in clustered_plot.points:
            assert pt.cluster_memb is not None
            # Labels are 1-indexed strings
            assert pt.cluster_memb.isdigit()
            assert int(pt.cluster_memb) >= 1

    def test_prediction_points_have_value(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        pred_plot = plots[1]
        for pt in pred_plot.points:
            assert pt.value is not None
            float(pt.value)  # should be parseable as float

    def test_clustered_colours_are_hex(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        hex_pat = re.compile(r"^#[0-9a-fA-F]{6}$")
        for pt in plots[0].points:
            assert hex_pat.match(pt.color), f"Not a hex colour: {pt.color!r}"

    def test_prediction_colours_are_rgba(self):
        plots = _build_interact_plots(self.bunch, self.kmeans, self.method)
        for pt in plots[1].points:
            assert pt.color.startswith("rgba("), f"Not rgba: {pt.color!r}"


# ---------------------------------------------------------------------------
# _build_interact_plots — binary classification set-up
# ---------------------------------------------------------------------------


class TestBuildInteractPlotsBinary:
    def test_binary_setup(self):
        bunch = _make_plot_data_bunch(set_up="binary")
        # For binary, rf_pred must be a float in [0,1]
        bunch.rf_pred = float(RNG.random())
        bunch.pred = RNG.random(N_TREES)
        kmeans = _make_kmeans()
        method = _make_input_method()
        plots = _build_interact_plots(bunch, kmeans, method)
        assert len(plots) == 2
        for pt in plots[1].points:
            assert pt.value is not None


# ---------------------------------------------------------------------------
# _build_interact_plots — single-plot mode
# ---------------------------------------------------------------------------


class TestBuildInteractPlotsSinglePanel:
    def test_single_clustered_panel(self):
        bunch = _make_plot_data_bunch()
        kmeans = _make_kmeans()
        method = _make_input_method()
        plots = _build_interact_plots(bunch, kmeans, method, clusterplots=(True,))
        assert len(plots) == 1
        assert plots[0].clustered is True

    def test_single_prediction_panel(self):
        bunch = _make_plot_data_bunch()
        kmeans = _make_kmeans()
        method = _make_input_method()
        plots = _build_interact_plots(bunch, kmeans, method, clusterplots=(False,))
        assert len(plots) == 1
        assert plots[0].clustered is False


# ---------------------------------------------------------------------------
# _build_interact_plots — empty-cluster guard
# ---------------------------------------------------------------------------


class TestBuildInteractPlotsEdgeCases:
    def test_empty_cluster_raises(self):
        bunch = _make_plot_data_bunch()
        bad_kmeans = _make_kmeans()
        # Labels skip cluster index 1: np.bincount gives [18, 0, 2] → min == 0
        bad_kmeans.labels_ = np.array([0] * 18 + [2] * 2, dtype=np.int32)
        method = _make_input_method()
        with pytest.raises(KeyError, match="Empty clusters"):
            _build_interact_plots(bunch, bad_kmeans, method)

    def test_no_final_trees(self):
        """If final_trees_idx is empty, all points should be circles."""
        bunch = _make_plot_data_bunch()
        kmeans = _make_kmeans()
        method = _make_input_method(final_count=0)
        plots = _build_interact_plots(bunch, kmeans, method)
        for ip in plots:
            for pt in ip.points:
                assert pt.shape == "circle"


# ---------------------------------------------------------------------------
# _build_plotly_scatter
# ---------------------------------------------------------------------------


class TestBuildPlotlyScatter:
    @pytest.fixture(autouse=True)
    def setup(self):
        bunch = _make_plot_data_bunch()
        kmeans = _make_kmeans()
        method = _make_input_method()
        self.plots = _build_interact_plots(bunch, kmeans, method)

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go

        for ip in self.plots:
            fig = _build_plotly_scatter(ip)
            assert isinstance(fig, go.Figure)

    def test_figure_has_one_trace(self):
        for ip in self.plots:
            fig = _build_plotly_scatter(ip)
            assert len(fig.data) == 1

    def test_trace_point_count(self):
        for ip in self.plots:
            fig = _build_plotly_scatter(ip)
            assert len(fig.data[0].x) == N_TREES
            assert len(fig.data[0].y) == N_TREES

    def test_customdata_contains_tree_indices(self):
        for ip in self.plots:
            fig = _build_plotly_scatter(ip)
            cd = list(fig.data[0].customdata)
            # Each customdata entry must be an int matching the tree index
            for i, val in enumerate(cd):
                assert isinstance(val, int), f"customdata[{i}] = {val!r} is not int"

    def test_hover_template_set(self):
        for ip in self.plots:
            fig = _build_plotly_scatter(ip)
            assert fig.data[0].hovertemplate is not None

    def test_layout_has_title(self):
        for ip in self.plots:
            fig = _build_plotly_scatter(ip)
            assert fig.layout.title.text is not None
            assert len(fig.layout.title.text) > 0

    def test_clustered_scatter_title(self):
        clustered_ip = self.plots[0]
        fig = _build_plotly_scatter(clustered_ip)
        assert "Cluster" in fig.layout.title.text

    def test_prediction_scatter_title(self):
        pred_ip = self.plots[1]
        fig = _build_plotly_scatter(pred_ip)
        assert "loss" in fig.layout.title.text or "Prediction" in fig.layout.title.text

    def test_axis_labels(self):
        for ip in self.plots:
            fig = _build_plotly_scatter(ip)
            assert fig.layout.xaxis.title.text == "PC1"
            assert fig.layout.yaxis.title.text == "PC2"


# ---------------------------------------------------------------------------
# plot_with_interface — API compatibility (gui mocked out)
# ---------------------------------------------------------------------------


class TestPlotWithInterfaceAPI:
    """
    Verify the public API contract without launching a real GUI window.
    ui.run and ui.page are patched so the test runs headless.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.bunch = _make_plot_data_bunch()
        self.kmeans = _make_kmeans()
        # input_method needs .sample.index[0] (used for the window title)
        sample_df = pd.DataFrame(RNG.random((1, 3)), columns=["f0", "f1", "f2"], index=[42])
        self.method = SimpleNamespace(
            final_trees_idx=[0, 1],
            sample=sample_df,
        )

    def test_returns_list_of_interact_plots(self):
        with patch("bellatrex.gui_plots_nicegui.ui") as mock_ui:
            mock_ui.run = MagicMock()
            mock_ui.page = lambda path: (lambda f: f)  # no-op decorator
            mock_ui.dialog = MagicMock(return_value=MagicMock())
            mock_ui.label = MagicMock()
            mock_ui.row = MagicMock(
                return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
            )
            mock_ui.card = MagicMock(
                return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
            )
            mock_ui.plotly = MagicMock(return_value=MagicMock())

            result = plot_with_interface(
                self.bunch,
                self.kmeans,
                self.method,
                temp_files_dir=None,  # must be accepted (backwards compat)
            )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_temp_files_dir_ignored(self):
        """Passing a temp_files_dir path must not cause any file I/O."""
        with patch("bellatrex.gui_plots_nicegui.ui") as mock_ui:
            mock_ui.run = MagicMock()
            mock_ui.page = lambda path: (lambda f: f)
            mock_ui.dialog = MagicMock(return_value=MagicMock())
            mock_ui.label = MagicMock()
            mock_ui.row = MagicMock(
                return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
            )
            mock_ui.card = MagicMock(
                return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
            )
            mock_ui.plotly = MagicMock(return_value=MagicMock())

            # Should not raise even though "/tmp/nonexistent" doesn't exist
            result = plot_with_interface(
                self.bunch,
                self.kmeans,
                self.method,
                temp_files_dir="/tmp/nonexistent",
            )
        assert result is not None

    def test_ui_run_called_once(self):
        with patch("bellatrex.gui_plots_nicegui.ui") as mock_ui:
            mock_ui.run = MagicMock()
            mock_ui.page = lambda path: (lambda f: f)
            mock_ui.dialog = MagicMock(return_value=MagicMock())
            mock_ui.label = MagicMock()
            mock_ui.row = MagicMock(
                return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
            )
            mock_ui.card = MagicMock(
                return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
            )
            mock_ui.plotly = MagicMock(return_value=MagicMock())

            plot_with_interface(self.bunch, self.kmeans, self.method)
            mock_ui.run.assert_called_once()

    def test_native_false_forwarded_to_ui_run(self):
        with patch("bellatrex.gui_plots_nicegui.ui") as mock_ui:
            mock_ui.run = MagicMock()
            mock_ui.page = lambda path: (lambda f: f)
            mock_ui.dialog = MagicMock(return_value=MagicMock())
            mock_ui.label = MagicMock()
            mock_ui.row = MagicMock(
                return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
            )
            mock_ui.card = MagicMock(
                return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
            )
            mock_ui.plotly = MagicMock(return_value=MagicMock())

            plot_with_interface(self.bunch, self.kmeans, self.method, native=False)
            call_kwargs = mock_ui.run.call_args.kwargs
            assert call_kwargs.get("native") is False
