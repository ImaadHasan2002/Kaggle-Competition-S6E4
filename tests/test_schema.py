"""Schema and column-splitting invariants."""

from __future__ import annotations

import polars as pl

from pred_irr_comp.schema import LABEL_MAP, TARGET_COL, split_columns


def test_label_map():
    assert LABEL_MAP == {'Low': 0, 'Medium': 1, 'High': 2}


def test_target_col():
    assert TARGET_COL == 'Irrigation_Need'


def test_split_columns_excludes_target_and_id(tiny_train_df: pl.DataFrame):
    num, cat = split_columns(tiny_train_df)
    assert TARGET_COL not in num
    assert TARGET_COL not in cat
    assert 'id' not in num
    assert 'id' not in cat
    for name, dtype in tiny_train_df.schema.items():
        if name in ('id', TARGET_COL):
            continue
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64):
            assert name in num
        elif dtype in (pl.Utf8, pl.Categorical, pl.String):
            assert name in cat
