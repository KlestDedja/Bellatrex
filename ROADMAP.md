# Roadmap

This file sketches the direction of the project. It’s meant as a _guidepost_ and priorities can shift over time.


## Near-term goals (v0.x series)

- Expand integration tests
- Expand compatibility up to Python 3.14 (bottleneck is _dearpygui_)
- Track coverage % in CI and target >80%. To reach this:
    - Refactor the GUI code completely, possibly get rid of _dearpygui_

### Code quality

- Align the supported Python version classifiers in `pyproject.toml` with what the
  CI matrix actually tests — blocked on `dearpygui <2.0`, which has no Python 3.13
  wheels. Will be resolved when the GUI dependency constraint is updated.


## Mid-term ideas (future versions)

### New features

- Enhance vector representation by including leaf predictions in the representation of the trees. The resulting representation could have $d+q$ dimensions, where $q$ is the output dimensionality and $d$ is the input dimensionality. Currently only feature splits are used to create the tree representation.
The weight of the extra $q$ dimensions should be controlled by a new parameter.

- Enhance Bellatrex explanations for multi-output tasks, enable users to:
    - select a subset of targets to run explanations for;
    - select a (single) target to run ``plot_visuals()``


### Type safety

- Add type hints throughout the codebase, starting with the public API in
  `bellatrex_explain.py` and `utilities.py`. Current coverage is below 1%.
- Use `from __future__ import annotations` to keep annotations compatible with
  older Python versions while writing modern syntax.

### Refactoring

- Break up functions that have grown too long. Primary targets:
    - `plot_preselected_trees()` in `utilities.py` (~268 lines): extract coloring,
      legend, and annotation logic into focused helpers.
    - `explain()` in `bellatrex_explain.py` (~186 lines): move grid search into its
      own internal method.
    - `plot_rules()` in `visualization.py` (~237 lines): separate colormap and
      subplot construction.
- Consider splitting `BellatrexExplain` responsibilities: a lean core class,
  a `RulesExporter` for file I/O, and visualization helpers. The current class
  handles fitting, tuning, plotting, and text export simultaneously.
- Replace the 12-parameter constructor with a config/dataclass object for
  infrequently-used options (`ys_oracle`, `force_refit`, `verbose`, …), keeping the
  common-case API simple.

### Documentation

- Add docstrings to all internal functions that currently have none (e.g.,
  `frmt_pretty_print`, `rule_to_file`, `_validate_p_grid`).
- Set up auto-generated API docs (Sphinx / ReadTheDocs).

### Testing

- Increase unit test coverage for `utilities.py` (currently the largest file with
  the fewest dedicated tests).
- Add edge-case tests: single-feature datasets, all-identical predictions, NaN
  inputs, feature mismatch between fit and explain.

### Drop dependency on pandas
- Replace internal `pandas.DataFrame` usage with:
  - NumPy arrays where appropriate.
  - A lightweight `Batch` object for structured tabular data.
- Ensure public API doesn’t break for users who rely on pandas-like inputs
- If pandas dependency cannot be avoided, add compatibility with `polars`


### Wider compatibility and support
- Add explainability features for more model types (Extra Tress, Gradient Boosting Trees, etc.).
- Provide a simple web demo (Streamlit or FastAPI).
- Improve documentation:
    - interactive notebooks
    - fully fledged website `readthedocs` style


## How to contribute

Open a PR directly, either for small fixes or for suggesting new features and roadmap items.

---
_Last updated: 2026-02-20_
