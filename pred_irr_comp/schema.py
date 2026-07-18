import polars as pl

ID_COLS: tuple[str, ...] = ('id',)
TARGET_COL: str = 'Irrigation_Need'
LABEL_MAP: dict[str, int] = {'Low': 0, 'Medium': 1, 'High': 2}

_NUM_DTYPES = {pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64}
_STR_DTYPES = {pl.Utf8, pl.Categorical}

def split_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numerical_cols, categorical_cols) excluding IDs and target."""
    num = [n for n, d in df.schema.items()
           if d in _NUM_DTYPES and n.lower() not in ID_COLS]
    cat = [n for n, d in df.schema.items()
           if d in _STR_DTYPES and n.lower() not in ID_COLS and n != TARGET_COL]
    return num, cat