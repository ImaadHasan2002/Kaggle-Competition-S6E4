"""Feature engineering (preprocessed v1.3) and encoding for irrigation prediction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from pred_irr_comp.config import PROCESSED_DATA_DIR
from pred_irr_comp.dataset import load_train_test
from pred_irr_comp.schema import ID_COLS, TARGET_COL, LABEL_MAP, split_columns

app = typer.Typer()

# --- Crop coefficient lookup (same as preprocessing.ipynb v1.3) ---
KC_TABLE: dict[tuple[str, str], float] = {
    ('Rice', 'Sowing'): 1.05,
    ('Rice', 'Vegetative'): 1.10,
    ('Rice', 'Flowering'): 1.20,
    ('Rice', 'Harvest'): 0.90,
    ('Sugarcane', 'Sowing'): 0.40,
    ('Sugarcane', 'Vegetative'): 0.85,
    ('Sugarcane', 'Flowering'): 1.25,
    ('Sugarcane', 'Harvest'): 0.75,
    ('Wheat', 'Sowing'): 0.30,
    ('Wheat', 'Vegetative'): 0.70,
    ('Wheat', 'Flowering'): 1.15,
    ('Wheat', 'Harvest'): 0.40,
    ('Maize', 'Sowing'): 0.30,
    ('Maize', 'Vegetative'): 0.70,
    ('Maize', 'Flowering'): 1.20,
    ('Maize', 'Harvest'): 0.60,
    ('Potato', 'Sowing'): 0.50,
    ('Potato', 'Vegetative'): 0.80,
    ('Potato', 'Flowering'): 1.15,
    ('Potato', 'Harvest'): 0.75,
    ('Cotton', 'Sowing'): 0.35,
    ('Cotton', 'Vegetative'): 0.75,
    ('Cotton', 'Flowering'): 1.15,
    ('Cotton', 'Harvest'): 0.50,
}
KC_MAP: dict[str, float] = {f'{c}|{s}': v for (c, s), v in KC_TABLE.items()}

SOIL_AWC: dict[str, float] = {
    'Sandy': 70.0,
    'Loamy': 150.0,
    'Silt': 200.0,
    'Clay': 180.0,
}
SOIL_DRAIN: dict[str, float] = {
    'Sandy': 0.9,
    'Loamy': 0.6,
    'Silt': 0.4,
    'Clay': 0.2,
}

CROP_WATER_DEMAND: dict[str, float] = {
    'Rice': 3.0,
    'Sugarcane': 2.5,
    'Maize': 2.0,
    'Wheat': 1.8,
    'Cotton': 1.8,
    'Potato': 1.5,
}

IRR_EFFICIENCY: dict[str, float] = {
    'Drip': 0.90,
    'Sprinkler': 0.75,
    'Canal': 0.55,
    'Rainfed': 0.30,
}
SOURCE_RELIABILITY: dict[str, float] = {
    'River': 3.0,
    'Reservoir': 2.5,
    'Groundwater': 2.0,
    'Rainwater': 0.5,
}

NEW_NUM_FEATURES_V13: list[str] = [
    'VPD',
    'Kc',
    'ETc_effective',
    'Effective_Rainfall_mm',
    'Soil_AWC',
    'Crop_Water_Demand',
    'Irrigation_Efficiency',
    'Source_Reliability',
    'Water_Balance',
    'Aridity_Index',
    'Soil_Moisture_Deficit',
    'Relative_Moisture',
    'Salinity_Stress',
    'pH_Stress',
    'Rainfall_Excess',
    'Wind_x_VPD',
    'Sun_x_Temp',
    'Total_Water_Need_m3',
    'Moisture_x_Carbon',
    'Rain_x_Drainage',
    'AWC_x_OC',
    'Efficient_Irrigation',
    'Is_Dry_Season',
    'Prev_Irr_Effective',
    'Demand_vs_Source',
    'soil_lt_25',
    'temp_gt_30',
    'rain_lt_300',
    'wind_gt_10',
    'Log_Rainfall',
    'Log_Field_Area',
    'Log_Prev_Irrigation',
]

ONEHOT_COLS: list[str] = [
    'Soil_Type',
    'Crop_Type',
    'Season',
    'Irrigation_Type',
    'Water_Source',
    'Region',
]

GROWTH_ORDER: list[list[str]] = [['Sowing', 'Vegetative', 'Flowering', 'Harvest']]
MULCHING_ORDER: list[list[str]] = [['No', 'Yes']]


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply v1.3 domain features (threshold flags + golden features)."""
    df = df.with_columns(
        [
            (pl.col('Soil_Moisture') < 25).cast(pl.Int8).alias('soil_lt_25'),
            (pl.col('Temperature_C') > 30).cast(pl.Int8).alias('temp_gt_30'),
            (pl.col('Rainfall_mm') < 300).cast(pl.Int8).alias('rain_lt_300'),
            (pl.col('Wind_Speed_kmh') > 10).cast(pl.Int8).alias('wind_gt_10'),
        ]
    )
    return (
        df.with_columns(
            VPD=(
                0.6108
                * (17.27 * pl.col('Temperature_C') / (pl.col('Temperature_C') + 237.3)).exp()
                * (1 - pl.col('Humidity') / 100)
            ),
        )
        .with_columns(
            Kc=pl.concat_str([pl.col('Crop_Type'), pl.lit('|'), pl.col('Crop_Growth_Stage')]).replace_strict(
                KC_MAP, default=0.80
            ),
        )
        .with_columns(
            _ETo_proxy=(
                0.0023
                * (pl.col('Temperature_C') + 17.8)
                * (pl.col('Sunlight_Hours') * (1 - pl.col('Humidity') / 100) + 1e-6).sqrt()
                * (1 + 0.1 * pl.col('Wind_Speed_kmh') / 10)
            ),
        )
        .with_columns(
            ETc_effective=(pl.col('Kc') * pl.col('_ETo_proxy'))
            * (1 - 0.25 * (pl.col('Mulching_Used') == 'Yes').cast(pl.Float64)),
        )
        .drop(['_ETo_proxy'])
        .with_columns(
            Effective_Rainfall_mm=pl.when(pl.col('Rainfall_mm') < 250)
            .then(pl.col('Rainfall_mm') * (125 - 0.2 * pl.col('Rainfall_mm')) / 125)
            .otherwise(125 + 0.1 * pl.col('Rainfall_mm')),
            Soil_AWC=pl.col('Soil_Type').replace_strict(SOIL_AWC, default=130.0),
            Crop_Water_Demand=pl.col('Crop_Type').replace_strict(CROP_WATER_DEMAND, default=2.0),
            Irrigation_Efficiency=pl.col('Irrigation_Type').replace_strict(IRR_EFFICIENCY, default=0.5),
            Source_Reliability=pl.col('Water_Source').replace_strict(SOURCE_RELIABILITY, default=1.5),
        )
        .with_columns(
            Water_Balance=pl.col('Effective_Rainfall_mm')
            + pl.col('Previous_Irrigation_mm')
            - pl.col('ETc_effective') * 30,
            Aridity_Index=pl.col('Effective_Rainfall_mm') / (pl.col('ETc_effective') * 30 + 1e-3),
            Soil_Moisture_Deficit=pl.col('Soil_AWC') - pl.col('Soil_Moisture') * 10,
            Relative_Moisture=pl.col('Soil_Moisture') * 10 / (pl.col('Soil_AWC') + 1e-3),
            Salinity_Stress=pl.max_horizontal(pl.col('Electrical_Conductivity') - 2.0, pl.lit(0.0)),
            pH_Stress=(pl.col('Soil_pH') - 6.5).abs(),
        )
        .with_columns(
            Rainfall_Excess=pl.max_horizontal(
                pl.col('Effective_Rainfall_mm') - pl.col('ETc_effective') * 30,
                pl.lit(0.0),
            ),
        )
        .with_columns(
            Wind_x_VPD=pl.col('Wind_Speed_kmh') * pl.col('VPD'),
            Sun_x_Temp=pl.col('Sunlight_Hours') * pl.col('Temperature_C'),
            Total_Water_Need_m3=pl.col('ETc_effective') * pl.col('Field_Area_hectare') * 10,
            Moisture_x_Carbon=pl.col('Soil_Moisture') * pl.col('Organic_Carbon'),
            Rain_x_Drainage=pl.col('Effective_Rainfall_mm')
            * pl.col('Soil_Type').replace_strict(SOIL_DRAIN, default=0.5),
            AWC_x_OC=pl.col('Soil_AWC') * pl.col('Organic_Carbon'),
            Efficient_Irrigation=pl.col('Irrigation_Type').is_in(['Drip', 'Sprinkler']).cast(pl.Float64),
            Is_Dry_Season=pl.col('Season').is_in(['Rabi', 'Zaid']).cast(pl.Float64),
            Prev_Irr_Effective=pl.col('Previous_Irrigation_mm') * pl.col('Irrigation_Efficiency'),
            Demand_vs_Source=pl.col('Crop_Water_Demand') / (pl.col('Source_Reliability') + 1e-3),
            Log_Rainfall=pl.col('Rainfall_mm').log1p(),
            Log_Field_Area=pl.col('Field_Area_hectare').log1p(),
            Log_Prev_Irrigation=pl.col('Previous_Irrigation_mm').log1p(),
        )
        .with_columns(
        )
    )


def build_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    """Passthrough numerics + OHE + ordinal growth/mulch (v1.3)."""
    return ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ONEHOT_COLS),
            ('growth', OrdinalEncoder(categories=GROWTH_ORDER), ['Crop_Growth_Stage']),
            ('mulch', OrdinalEncoder(categories=MULCHING_ORDER), ['Mulching_Used']),
        ],
        remainder='drop',
    )


def build_processed_dataset(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> dict[str, Any]:
    """
    Engineer features, fit preprocessor on train, return arrays and feature names.

    Expects train with TARGET_COL; test must not have TARGET_COL.
    """
    num_base, _cat_base = split_columns(train_df)
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)

    num_cols_v13 = num_base + NEW_NUM_FEATURES_V13
    preprocess = build_preprocessor(num_cols_v13)

    y_train = train_fe[TARGET_COL].replace_strict(LABEL_MAP).to_numpy()
    x_train = preprocess.fit_transform(train_fe.drop(TARGET_COL).to_pandas())
    x_test = preprocess.transform(test_fe.to_pandas())
    feature_names = list(preprocess.get_feature_names_out())

    return {
        'X_train': x_train,
        'X_test': x_test,
        'y_train': y_train,
        'feature_names': feature_names,
    }


@app.command()
def main(
    version: str = typer.Option('v1.3', help='Preprocessing version label (folder name prefix).'),
    output_dir: Path | None = typer.Option(
        None,
        help='Directory to write X_train.pkl, X_test.pkl, y_train.pkl, feature_names.pkl.',
    ),
) -> None:
    """Load raw train/test, apply v1.3 preprocessing, and pickle artifacts."""
    out = output_dir if output_dir is not None else PROCESSED_DATA_DIR / f'preprocessed_{version}'
    out.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_train_test()
    train_df = train_df.drop(ID_COLS)
    test_df = test_df.drop(ID_COLS)

    logger.info('Building preprocessed dataset (v1.3 feature engineering)...')
    bundle = build_processed_dataset(train_df, test_df)

    with open(out / 'X_train.pkl', 'wb') as f:
        pickle.dump(bundle['X_train'], f)
    with open(out / 'X_test.pkl', 'wb') as f:
        pickle.dump(bundle['X_test'], f)
    with open(out / 'y_train.pkl', 'wb') as f:
        pickle.dump(bundle['y_train'], f)
    with open(out / 'feature_names.pkl', 'wb') as f:
        pickle.dump(bundle['feature_names'], f)

    logger.success(
        f'Saved {bundle["X_train"].shape} X_train, {bundle["X_test"].shape} X_test -> {out}'
    )


if __name__ == '__main__':
    app()
