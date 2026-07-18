"""Stacking model smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold

from pred_irr_comp.modeling.train import build_stacking_model


def test_build_stacking_model_type():
    m = build_stacking_model()
    assert isinstance(m, StackingClassifier)


@pytest.mark.slow
def test_stacking_fit_predict_proba_smoke():
    """Fast hyperparameters for CI; checks shapes and label range."""
    rng = np.random.default_rng(42)
    n, p = 200, 71
    x = rng.random((n, p)).astype(np.float64)
    y = np.repeat([0, 1, 2], [70, 70, 60])

    model = build_stacking_model()
    model.set_params(
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        lgb__n_estimators=40,
        lgb__min_child_samples=5,
        lgb__n_jobs=1,
        cat__iterations=40,
        cat__depth=4,
        cat__thread_count=1,
        xgb__n_estimators=40,
        xgb__max_depth=3,
        xgb__n_jobs=1,
        final_estimator__max_iter=100,
        final_estimator__n_jobs=1,
    )
    model.fit(x, y)
    pred = model.predict(x)
    assert set(np.unique(pred)).issubset({0, 1, 2})
    proba = model.predict_proba(x)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
