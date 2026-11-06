import numpy as np
import pandas as pd
from functions import portfolio_volatilitycalculation
import pytest

def test_constant_returns_zero_volatility():

    test_returns = pd.DataFrame({
        "ASSET_A": [0.01, 0.01, 0.01, 0.01]
    })

    weights = np.array([1.0])

    # Calculate volatility
    calculated = portfolio_volatilitycalculation(test_returns, ["ASSET_A"], weights)

    # Verify that volatility is zero
    assert np.isclose(calculated, 0)

def test_nonzero_volatility():
        test_returns = pd.DataFrame({
                "ASSET_A": [0.01, -0.01, 0.01, -0.01]
            })
        weights = np.array([1.0])
        
        calculatedvolatility = portfolio_volatilitycalculation(test_returns, ["ASSET_A"], weights)

        expected_volatility = np.sqrt(4 * 0.01**2 / 3) * np.sqrt(252)

        assert np.isclose(calculatedvolatility, expected_volatility)

def test_perfect_negative_correlation():
    test_returns = pd.DataFrame({
                    "ASSET_A": [0.01, -0.01, 0.01, -0.01], "ASSET_B": [-0.01, 0.01, -0.01, 0.01]
                })
    weights = np.array([0.5, 0.5])
    calculatedvolatility = portfolio_volatilitycalculation(test_returns, ["ASSET_A", "ASSET_B"], weights)
    assert np.isclose(calculatedvolatility, 0)

def test_invalid_weights():

    test_returns = pd.DataFrame({
        "ASSET_A": [0.01, -0.01],
        "ASSET_B": [0.02, -0.02]
    })

    invalid_weights = np.array([0.8, 0.8])

    with pytest.raises(ValueError):
        portfolio_volatilitycalculation(test_returns, ["ASSET_A", "ASSET_B"], invalid_weights)
  