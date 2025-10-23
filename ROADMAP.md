# Roadmap

This file sketches the direction of the project. It’s meant as a _guidepost_ and priorities can shift over time.
No temporal line is suggested, but ratheer a list of things to do and ideas to pursue.

## Near-term goals (v0.x series)

- Expand integration tests
- Expand compatibility up to Python 3.14 (bottleneck is _dearpygui_)
- Track coverage % in CI and target >80%. To reach this:
    - Refactor the GUI code completely, possibly get rid of _dearpygui_

### New features

- Enhance vector representation by including leaf predictions in the representation of the trees. The resulting representation could have $d+q$ dimensions, where $q$ is the output dimensionalityt and $d$ is the input dimensionality. Currently only feature splits are used to create the tree representation.
The weight of the extra $q$ dimensions should be controlled by a new parameter.

- Enhance Bellatrex explanations for multi-output tasks, enable users to:
    - select a subset of targets to run explanations for;
    - select a (single) target to run ``plot_visuals()``




## Mid-term ideas (future versions)

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
_Last updated: 2025-10-23_
