"""Feature engineering and preprocessing (v1.3)."""

from __future__ import annotations

import numpy as np

from pred_irr_comp.features import (
    NEW_NUM_FEATURES_V13,
    build_preprocessor,
    build_processed_dataset,
    engineer_features,
)
from pred_irr_comp.schema import TARGET_COL, split_columns


def test_engineer_features_adds_columns_and_finite(tiny_train_df):
    out = engineer_features(tiny_train_df)
    for name in NEW_NUM_FEATURES_V13:
        assert name in out.columns, f'missing engineered column {name}'
    for name in NEW_NUM_FEATURES_V13:
        s = out[name]
        assert not s.is_null().any(), f'nulls in {name}'
        arr = s.to_numpy()
        assert np.isfinite(arr).all(), f'non-finite in {name}'


def test_preprocessor_shape_matches_expectation(tiny_train_df):
    num_base, _ = split_columns(tiny_train_df)
    train_fe = engineer_features(tiny_train_df)
    num_cols_v13 = num_base + NEW_NUM_FEATURES_V13
    pre = build_preprocessor(num_cols_v13)
    x = pre.fit_transform(train_fe.drop(TARGET_COL).to_pandas())
    assert x.ndim == 2
    assert x.shape[0] == len(tiny_train_df)
    expected_cols = len(pre.get_feature_names_out())
    assert x.shape[1] == expected_cols


def test_build_processed_dataset_aligns_train_test(tiny_train_df, tiny_test_df):
    bundle = build_processed_dataset(tiny_train_df, tiny_test_df)
    x_tr = bundle['X_train']
    x_te = bundle['X_test']
    y = bundle['y_train']
    names = bundle['feature_names']
    assert x_tr.shape[1] == x_te.shape[1]
    assert len(names) == x_tr.shape[1]
    assert y.shape[0] == x_tr.shape[0]
    assert set(np.unique(y)).issubset({0, 1, 2})
