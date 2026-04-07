"""Tests for the Core Visualization Library (CVL)."""

import io

import matplotlib
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coremaker.example import example_core
from coremaker.visualization import (
    TransitionPlan,
    plot_categorical,
    plot_heatmap,
    plot_transition,
)
from coremaker.visualization.coregeometry import (
    all_site_geometries,
    cell_dimensions_for_site,
    occupied_sites,
    site_geometry,
    site_labels_from_core,
)
from coremaker.visualization.geometry import make_patch
from coremaker.visualization.types import CellGeometry, CellShape


@pytest.fixture()
def core():
    return example_core


@pytest.fixture()
def site_values(core):
    """Deterministic per-site scalar values based on radial distance."""
    geoms = all_site_geometries(core)
    occ = occupied_sites(core)
    return {
        site: round(1.5 - np.hypot(geoms[site].center_x, geoms[site].center_y) / 60.0, 4)
        for site in occ
    }


@pytest.fixture()
def site_categories(core):
    """Assign a category to each occupied site in a checkerboard pattern."""
    occ = sorted(occupied_sites(core))
    return {site: ("TypeA" if i % 2 == 0 else "TypeB") for i, site in enumerate(occ)}


def _fig_to_png_bytes(fig) -> bytes:
    """Render a figure to PNG bytes for image regression checks."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


class TestCellGeometry:
    def test_default_width_y_is_none(self):
        geom = CellGeometry(1.0, 2.0, CellShape.HEXAGON, 5.0)
        assert geom.width_y is None


class TestCoreGeometry:
    def test_cell_dimensions_match_lattice(self, core):
        site = next(iter(core.grid.keys()))
        wx, wy = cell_dimensions_for_site(core.grid, site)
        dims = core.grid.lattice.dimensions
        assert wx == pytest.approx(float(dims[0]))
        assert wy == pytest.approx(float(dims[1]))

    def test_site_geometry_returns_cellgeometry(self, core):
        site = next(iter(core.grid.keys()))
        geom = site_geometry(core, site)
        assert isinstance(geom, CellGeometry)
        assert geom.cell_shape == CellShape.SQUARE

    def test_all_site_geometries_covers_all_sites(self, core):
        geoms = all_site_geometries(core)
        all_sites = set(core.grid.sites())
        assert set(geoms.keys()) == all_sites

    def test_occupied_sites_subset_of_all(self, core):
        occ = occupied_sites(core)
        all_sites = set(core.grid.sites())
        assert occ.issubset(all_sites)

    def test_site_labels_uses_element_name(self, core):
        labels = site_labels_from_core(core)
        for site, label in labels.items():
            element = core.grid[site]
            assert label == element.name


class TestMakePatch:
    @given(
        cx=st.floats(-1e3, 1e3),
        cy=st.floats(-1e3, 1e3),
        wx=st.floats(0.1, 1e3),
        wy=st.floats(0.1, 1e3),
    )
    def test_square_patch_position_and_size(self, cx, cy, wx, wy):
        geom = CellGeometry(cx, cy, CellShape.SQUARE, wx, wy)
        patch = make_patch(geom)
        assert patch.get_x() == pytest.approx(cx - wx / 2)
        assert patch.get_y() == pytest.approx(cy - wy / 2)
        assert patch.get_width() == pytest.approx(wx)
        assert patch.get_height() == pytest.approx(wy)

    @given(
        cx=st.floats(-1e3, 1e3),
        cy=st.floats(-1e3, 1e3),
        wx=st.floats(0.1, 1e3),
    )
    def test_square_patch_default_height_equals_width(self, cx, cy, wx):
        geom = CellGeometry(cx, cy, CellShape.SQUARE, wx)
        patch = make_patch(geom)
        assert patch.get_width() == pytest.approx(wx)
        assert patch.get_height() == pytest.approx(wx)

    def test_hexagon_patch_has_six_vertices(self):
        geom = CellGeometry(0.0, 0.0, CellShape.HEXAGON, 10.0)
        patch = make_patch(geom)
        verts = patch.get_xy()
        # Polygon.get_xy() repeats the first vertex to close the path
        assert len(verts) == 7
        np.testing.assert_allclose(verts[0], verts[-1])

    def test_hexagon_patch_centered(self):
        cx, cy = 3.0, 7.0
        geom = CellGeometry(cx, cy, CellShape.HEXAGON, 10.0)
        patch = make_patch(geom)
        verts = patch.get_xy()[:-1]  # exclude closing vertex
        centroid = verts.mean(axis=0)
        assert centroid[0] == pytest.approx(cx, abs=1e-10)
        assert centroid[1] == pytest.approx(cy, abs=1e-10)


class TestPlotHeatmap:
    def test_returns_figure_and_axes(self, core, site_values):
        fig, ax = plot_heatmap(core, site_values)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_title_is_set(self, core, site_values):
        fig, ax = plot_heatmap(core, site_values, title="Power Map")
        assert ax.get_title() == "Power Map"
        plt.close(fig)

    def test_axis_labels(self, core, site_values):
        fig, ax = plot_heatmap(core, site_values)
        assert "x" in ax.get_xlabel().lower()
        assert "y" in ax.get_ylabel().lower()
        plt.close(fig)

    def test_colorbar_present(self, core, site_values):
        fig, ax = plot_heatmap(core, site_values, units="MW")
        colorbars = [c for c in fig.get_axes() if c is not ax]
        assert len(colorbars) >= 1
        plt.close(fig)

    def test_accepts_existing_axes(self, core, site_values):
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_heatmap(core, site_values, ax=ax_ext)
        assert ax is ax_ext
        assert fig is fig_ext
        plt.close(fig)

    def test_labels_hidden_when_disabled(self, core, site_values):
        fig, ax = plot_heatmap(core, site_values, show_labels=False)
        annotations = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
        assert len(annotations) == 0
        plt.close(fig)

    def test_image_regression(self, core, site_values, image_regression):
        fig, _ = plot_heatmap(
            core, site_values, cmap="viridis", title="Regression Test", show_labels=False
        )
        image_regression.check(
            _fig_to_png_bytes(fig), diff_threshold=0.1, basename="heatmap_regression"
        )
        plt.close(fig)


class TestPlotCategorical:
    def test_returns_figure_and_axes(self, core, site_categories):
        fig, ax = plot_categorical(core, site_categories)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_legend_contains_all_categories(self, core, site_categories):
        fig, ax = plot_categorical(core, site_categories)
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = {t.get_text() for t in legend.get_texts()}
        assert legend_labels == set(site_categories.values())
        plt.close(fig)

    def test_image_regression(self, core, site_categories, image_regression):
        colors = {"TypeA": "#4a90d9", "TypeB": "#d97b4a"}
        fig, _ = plot_categorical(
            core, site_categories, color_dict=colors,
            title="Categorical Regression", show_labels=False,
        )
        image_regression.check(
            _fig_to_png_bytes(fig), diff_threshold=0.1, basename="categorical_regression"
        )
        plt.close(fig)


class TestTransitionPlan:
    def test_empty_plan(self):
        plan = TransitionPlan()
        assert plan.movements == []
        assert plan.loads == []
        assert plan.discharges == []

    def test_plan_with_data(self):
        plan = TransitionPlan(
            movements=[("A1", "B2")],
            loads=[("C3", "fresh")],
            discharges=["D4"],
        )
        assert len(plan.movements) == 1
        assert len(plan.loads) == 1
        assert len(plan.discharges) == 1


class TestPlotTransition:
    def test_returns_figure_and_axes(self, core):
        plan = TransitionPlan(movements=[("A1", "B2")])
        fig, ax = plot_transition(core, plan)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_arrows_drawn_for_movements(self, core):
        plan = TransitionPlan(movements=[("A1", "B2"), ("C3", "D4")])
        fig, ax = plot_transition(core, plan)
        from matplotlib.patches import FancyArrowPatch
        arrows = [c for c in ax.get_children() if isinstance(c, FancyArrowPatch)]
        assert len(arrows) >= 2
        plt.close(fig)

    def test_discharge_markers_drawn(self, core):
        plan = TransitionPlan(discharges=["E5", "F6"])
        fig, ax = plot_transition(core, plan)
        lines = ax.get_lines()
        discharge_markers = [ln for ln in lines if ln.get_marker() == "X"]
        assert len(discharge_markers) == 2
        plt.close(fig)

    def test_load_markers_drawn(self, core):
        plan = TransitionPlan(loads=[("A1", "fresh"), ("B2", "MOX")])
        fig, ax = plot_transition(core, plan)
        lines = ax.get_lines()
        load_markers = [ln for ln in lines if ln.get_marker() == "s"]
        assert len(load_markers) == 2
        plt.close(fig)

    def test_image_regression(self, core, image_regression):
        plan = TransitionPlan(
            movements=[("A1", "B2"), ("C3", "D4")],
            loads=[("E5", "fresh")],
            discharges=["F6"],
        )
        fig, _ = plot_transition(
            core, plan, title="Transition Regression",
            load_color_dict={"fresh": "#2ecc71"},
            rod_color_dict={"fuel_rod": "#4a90d9"},
        )
        image_regression.check(
            _fig_to_png_bytes(fig), diff_threshold=0.1, basename="transition_regression"
        )
        plt.close(fig)
