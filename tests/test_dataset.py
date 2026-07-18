"""Dataset I/O."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pred_irr_comp.config import RAW_DATA_DIR
from pred_irr_comp.dataset import load_train_test, read_dataframe
from pred_irr_comp.schema import TARGET_COL


def test_read_dataframe_roundtrip(tmp_path: Path, tiny_train_df: pl.DataFrame):
    path = tmp_path / 't.csv'
    tiny_train_df.write_csv(path)
    out = read_dataframe(path)
    assert out.shape == tiny_train_df.shape
    assert set(out.columns) == set(tiny_train_df.columns)


def test_load_train_test_schema():
    train_p = RAW_DATA_DIR / 'train.csv'
    if not train_p.is_file():
        pytest.skip('data/raw/train.csv not present')
    tr, te = load_train_test()
    assert isinstance(tr, pl.DataFrame)
    assert isinstance(te, pl.DataFrame)
    assert TARGET_COL in tr.columns
    assert TARGET_COL not in te.columns
    # Feature columns (excluding id and target) should align
    tr_feats = [c for c in tr.columns if c not in ('id', TARGET_COL)]
    te_feats = [c for c in te.columns if c != 'id']
    assert tr_feats == te_feats
