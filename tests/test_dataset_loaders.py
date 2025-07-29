import pytest
import pandas as pd
import numpy as np

from bellatrex.datasets import (
    load_binary_data,
    load_regression_data,
    load_survival_data,
    load_mlc_data,
    load_mtr_data,
)

DATA_LOADERS = {
    "binary": load_binary_data,
    "regression": load_regression_data,
    "survival": load_survival_data,
    "multi-label": load_mlc_data,
    "multi-target": load_mtr_data,
}


@pytest.mark.parametrize(
    "loader",
    [
        load_binary_data,
        load_regression_data,
        load_survival_data,
        load_mlc_data,
        load_mtr_data,
    ],
)
def test_dataset_returns_dataframe(loader):
    df = loader(return_X_y=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Dataset should not be empty"


@pytest.mark.parametrize(
    "loader",
    [
        load_binary_data,
        load_regression_data,
        load_survival_data,
        load_mlc_data,
        load_mtr_data,
    ],
)
def test_dataset_returns_X_y(loader):
    X, y = loader(return_X_y=True)

    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"

    # Survival task returns structured numpy array
    if isinstance(y, np.ndarray) and y.dtype.names:
        assert "event" in y.dtype.names or len(y.dtype.names) == 2
    else:
        assert isinstance(y, (pd.Series, pd.DataFrame)), "y should be Series or DataFrame"

    assert len(X) == len(y), "X and y should have the same number of samples"
