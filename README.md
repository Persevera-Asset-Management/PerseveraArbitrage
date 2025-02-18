# PerseveraArbitrage

Statistical arbitrage tools for Persevera Asset Management.

## Installation

```
pip install git+https://github.com/Persevera-Asset-Management/PerseveraArbitrage.git
```
## Usage
### Johansen Portfolio
```
from persevera_arbitrage.cointegration import JohansenPortfolio, CointegrationConfig

# Initialize portfolio
portfolio = JohansenPortfolio()

# Fit model
portfolio.fit(price_data)

# Get trading signals
signals = portfolio.get_trading_signals()
```
### Portfolio Analytics
```
# Get portfolio characteristics
print(f"Half-life: {portfolio.half_life:.1f} days")
print(f"Current z-score: {portfolio.zscore.iloc[-1]:.2f}")

# Access test statistics
eigen_stats = portfolio.johansen_eigen_statistic
trace_stats = portfolio.johansen_trace_statistic

# Get cointegration vectors and hedge ratios
vectors = portfolio.cointegration_vectors
ratios = portfolio.hedge_ratios
```