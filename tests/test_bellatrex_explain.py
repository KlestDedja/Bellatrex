import pytest
import numpy as np
import pandas as pd
from app.bellatrex.bellatrex_explain import BellatrexExplain
# from app.bellatrex.wrapper_class import EnsembleWrapper
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest

@pytest.fixture
def mock_clf():
    return RandomForestClassifier(n_estimators=10, random_state=0)

@pytest.fixture
def mock_data():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = np.random.randint(0, 2, size=100)
    return X, y

def test_bellatrex_explain_fit(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    assert explainer.is_fitted() is True

def test_bellatrex_explain_explain(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    explanation = explainer.explain(X, idx=0)
    assert explanation is not None

def test_bellatrex_explain_plot_overview(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    fig, axes = explainer.explain(X, idx=0).plot_overview(show=False)
    # fig, axes = explainer.plot_overview(show=False)
    assert fig is not None
    assert axes is not None


@pytest.mark.skip(reason="Not implemented yet")
def test_predict_survival_curve(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    survival_curve = explainer.predict_survival_curve(X, 0)
    assert survival_curve is not None
    assert isinstance(survival_curve, pd.DataFrame)

@pytest.mark.skip(reason="Not implemented yet")
def test_predict_median_surv_time(mock_clf, mock_data):
    X, y = mock_data
    explainer = BellatrexExplain(mock_clf, verbose=1)
    explainer.fit(X, y)
    survival_curve = explainer.predict_median_surv_time(X, 0)
    assert survival_curve is not None
    assert isinstance(survival_curve, pd.DataFrame)

