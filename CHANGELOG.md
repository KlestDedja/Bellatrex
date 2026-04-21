# Changelog

## [0.4.0] - 2026-04-12

This release supersedes the previously planned `0.3.2`. Its scope grew enough to warrant a
minor-version bump, driven mainly by the new browser-based GUI and broader API and packaging
cleanup.

### Added
- A browser-based interactive GUI powered by NiceGUI, with clickable rule exploration,
  dedicated tree views, and a new internal rendering/runtime layer.
- Top-level imports for `pack_trained_ensemble` and `predict_helper`, making it easier to work
  with packed forests directly from `bellatrex`.
- Bundled pretrained example models for all supported task types, together with normalized
  tutorial dataset names.

### Changed
- The optional GUI stack has moved from DearPyGUI to a browser-based NiceGUI workflow;
  `bellatrex[gui]` now installs `nicegui`, `pywebview`, and `plotly`.
- The README and tutorials were overhauled with clearer installation steps, a proper
  quickstart, and an API overview for the main public entry points.
- `BellatrexExplain` now accepts normalized `set_up` aliases, exposes a more helpful
  `__repr__`, uses `p_grid=None` instead of a mutable default, and produces cleaner text
  explanations.
- Declared Python support now covers 3.10 through 3.14, and the CI pipeline now separates a
  quick branch workflow from the full cross-platform matrix.

### Fixed
- `explain()` now fails fast with clearer errors for non-DataFrame inputs, invalid sample
  indices, NaN or infinite values, and feature-name mismatches.
- `ys_oracle` is now stored correctly and no longer gets mutated during single-sample
  explanations.
- Packed-tree conversion now handles the single-class binary-tree edge case more gracefully
  when exporting wrapped ensembles.
- `verbose=0` is now truly silent, and `n_jobs > 1` no longer emits a misleading warning.
- `create_rules_txt()` and `print_rules_txt()` now share consistent path resolution and
  correctly honor `BELLATREX_EXPLAIN_DIR`.
- NiceGUI tree windows, scrollbars, colorbars, and temporary-artifact cleanup were stabilized
  across the GUI refactor.
- Tree-related tests are now deterministic, improving reproducibility in local runs and CI.

### Removed
- DearPyGUI as the supported optional GUI backend; the legacy `gui_plots_code` module is now
  deprecated in favor of `nicegui_plots_code`.
- Legacy Docker-based CI artifacts that were no longer part of the active development flow.

## [0.3.1] - 2025-10-25

### Enhanced
- Test coverage has been extended, and `codecov` platform is being used for reporting. Its reports are synchronized with local pytest runs.
- Improved GitHub workflow actions, they now include: `cross-platform`,  `cross-version` checks, `coverage` checks, `CodeQL`, and automatic `release` to PyPi.


## [0.3.0] - 2025-07-24

### Enhanced
Revamped README.md, which now includes project badges (build status, version, license, etc.), clearer project description and usage instructions

A CI pipeline is now running with the first workflows (e.g., install checks, linting, test execution), which are now functioning correctly. A first version of dependency checks has been set in place.

An initial test coverage is introduced, with first set of pytest running successfully

### Fixed

Many `DeprecationWarning` warnings have been resolved (mainly within numpy and matplotlib), making the package more future-proof


## [0.2.3] - 2024-11-15

### Fixed
- Fixed bug when running `set_up=multi-label` with `EnsembleWrapper`
- Updated dependency constraints with `dearpygui`, newest versions are not supported (`lighttheme = create_theme_imgui_light()` raises an error).

### Refactored
- Cleaned some code and dropped unused files and functions (still ongoing)
### Enhanced
- Further streamlined loading of pre-trained models with `EnsembleWrapper` class.
- Improved compatibility between `EnsembleWrapper`and GUI interface



## [0.2.2] - 2024-08-12
### Fixed
- Updated README.md with absolute paths instead of relative ones.
- Updated Python requirements, show explicitly under ```setup.classifiers`` that the package is compatible with any `python >= 3.9`.
### Enhanced
- Improved plotting format in plot_visuals() and inline printing in general, with custom function `frmt_pretty_print`.
- Streamlined storing of trained (Random Forest) models and loading of pre-trained models (Random Forest)

## [0.2.1] - 2024-09-02
### Fixed
- Fixed bug while calling plot_visuals() with `preds_distr = None` and `conf_level != None`, sometimes the wrong axis was being modified.

## [0.2.0] - 2024-09-01
### Refactored
- Refactor the explain() method to accomodate for plot_overview() and plot_visuals(). Separating tasks and making method chaining possible.
- Refactor code to improve performance (e.g. not storing entire test set under `self.X` if not necessary).
### Enhanced
- Improved plotting format in plot_visuals() and inline printing in general, with custom function `frmt_pretty_print`.

## [0.1.4] - 2024-07-28
### Added
- Added regression and multi-label classification datasets to bellatrex.datasets, added feature names to binary dataset.
- Added enhanced visualization output, compatible with single-output predictions.

### Refactored
- Refactored code in the BellatrexExplain class, and gui_plots_code script.
- Refactored code in the visualization script for compatibiility with BellatrexExplain.

## [0.1.3] - 2024-07-24
### Fixed
- Fixed version file (it was still problematic), made it a .txt file. Updated MANIFEST.in accordingly.

## [0.1.2] - 2024-07-24
### Tested
- Test PyPi released successfully.

## [0.1.1] - 2024-07-24
### Fixed
- Fixed a bug with the version file that prevented the package from being installed properly. Moved the version file and updated the related path imports.

## [0.1.0] - 2024-07-23
### Added
- Initial release, including a draft version of the Graphical User Interface.
