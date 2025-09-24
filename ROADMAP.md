# Roadmap

This file sketches the direction of the project. It’s meant as a _guidepost, not a contract_: priorities can shift as we learn more.

## Near-term goals (v0.x series)

- Expand integration tests
- Track coverage % in CI and target >80%.

### New features

- Enhance vector representation by using leaf prediction (currently only feature nodes splits are used))


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
_Last updated: 2025-09-24_
