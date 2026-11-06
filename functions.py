import numpy as np

def portfolio_volatilitycalculation(returns, tickers, weights):
    if len(tickers) != len(weights):
        raise ValueError("Number of weights must match number of ETFs.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    
    cov_matrix = returns[tickers].cov()

    portfolio_variance = weights.T @ cov_matrix @ weights

    portfolio_volatility = np.sqrt(portfolio_variance * 252)


    return portfolio_volatility

def portfolio_return(returns, tickers, weights):
    if len(tickers) != len(weights):
           raise ValueError("Number of weights must match number of ETFs.")
   
    if not np.isclose(weights.sum(), 1.0):
         raise ValueError("Portfolio weights must sum to 1.")

    asset_returns = returns[tickers].mean() * 252

    portfolio_mean_return = weights @ asset_returns

    return portfolio_mean_return

def portfolio_sharpe(returns, tickers, weights, risk_free_rate=0.03):

    annual_return = portfolio_return(returns, tickers, weights)

    annual_volatility = portfolio_volatilitycalculation(returns, tickers, weights)

    # Calculate Sharpe ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    return sharpe_ratio


def cumulative_performance(returns, initial_investment=10_000):
    
    growth_factors = 1 + returns

    cumulative_growth = growth_factors.cumprod()

    # Calculate investment value
    investment_value = cumulative_growth*initial_investment

    return investment_value

def maximum_drawdown(returns):
    investment_value = cumulative_performance(returns)

    running_peak = investment_value.cummax()

    drawdown = investment_value / running_peak - 1


    maximum_drawdown = drawdown.min()

    return maximum_drawdown


def calculate_cagr(returns):

    investment_value = cumulative_performance(returns)

    # Step 2: Obtain the final investment value
    final_value = investment_value.iloc[-1]

    initial_value = 10_000

    years = len(returns)/252

    cagr = (final_value/initial_value)**(1/years) - 1

    return cagr
  