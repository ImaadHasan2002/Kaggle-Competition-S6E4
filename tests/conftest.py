"""Shared pytest fixtures (small synthetic frames matching raw schema)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pred_irr_comp.schema import TARGET_COL


def _row(
    *,
    soil: str,
    crop: str,
    stage: str,
    season: str,
    irr: str,
    water: str,
    region: str,
    mulch: str,
    target: str,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    return {
        'Soil_Type': soil,
        'Soil_pH': float(rng.uniform(5.0, 8.0)),
        'Soil_Moisture': float(rng.uniform(15.0, 45.0)),
        'Organic_Carbon': float(rng.uniform(0.5, 2.0)),
        'Electrical_Conductivity': float(rng.uniform(1.0, 4.0)),
        'Temperature_C': float(rng.uniform(12.0, 38.0)),
        'Humidity': float(rng.uniform(25.0, 85.0)),
        'Rainfall_mm': float(rng.uniform(50.0, 1200.0)),
        'Sunlight_Hours': float(rng.uniform(4.0, 11.0)),
        'Wind_Speed_kmh': float(rng.uniform(2.0, 22.0)),
        'Crop_Type': crop,
        'Crop_Growth_Stage': stage,
        'Season': season,
        'Irrigation_Type': irr,
        'Water_Source': water,
        'Field_Area_hectare': float(rng.uniform(0.5, 25.0)),
        'Mulching_Used': mulch,
        'Previous_Irrigation_mm': float(rng.uniform(10.0, 150.0)),
        'Region': region,
        TARGET_COL: target,
    }


@pytest.fixture
def tiny_train_df() -> pl.DataFrame:
    """24 rows covering categorical levels used in encoding."""
    soils = ['Sandy', 'Loamy', 'Silt', 'Clay']
    crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Potato', 'Cotton']
    stages = ['Sowing', 'Vegetative', 'Flowering', 'Harvest']
    seasons = ['Kharif', 'Rabi', 'Zaid']
    irrs = ['Drip', 'Sprinkler', 'Canal', 'Rainfed']
    waters = ['River', 'Reservoir', 'Groundwater', 'Rainwater']
    regions = ['East', 'West', 'North', 'South']
    targets = ['Low', 'Medium', 'High']

    rows = []
    i = 0
    for soil in soils:
        for crop in crops[:4]:
            rows.append(
                _row(
                    soil=soil,
                    crop=crop,
                    stage=stages[i % 4],
                    season=seasons[i % 3],
                    irr=irrs[i % 4],
                    water=waters[i % 4],
                    region=regions[i % 4],
                    mulch='Yes' if i % 2 == 0 else 'No',
                    target=targets[i % 3],
                    seed=1000 + i,
                )
            )
            i += 1
    return pl.DataFrame(rows)


@pytest.fixture
def tiny_test_df(tiny_train_df: pl.DataFrame) -> pl.DataFrame:
    """Test split: same feature columns, no target."""
    return tiny_train_df.drop(TARGET_COL)
