# Volatility Trading Strategy

A comprehensive algorithmic trading strategy for the Volatility Trading Case, designed to profit from options mispricing and volatility differentials.

## Strategy Overview

This strategy implements a sophisticated volatility trading system that:

1. **Forecasts Volatility**: Uses historical data and news sentiment to predict future volatility
2. **Identifies Mispricing**: Compares theoretical vs market option prices to find opportunities
3. **Executes Arbitrage**: Implements put-call parity and volatility arbitrage strategies
4. **Manages Risk**: Uses Greeks for portfolio hedging and risk management

## Key Features

### Volatility Forecasting
- Historical volatility calculation using rolling windows
- News sentiment analysis with keyword detection
- Mean reversion adjustments
- Multi-factor volatility prediction

### Options Pricing
- Black-Scholes model implementation
- Greeks calculation (Delta, Gamma, Theta, Vega)
- Implied volatility extraction
- Theoretical vs market price comparison

### Arbitrage Strategies
- **Put-Call Parity**: Identifies violations of put-call parity for risk-free profits
- **Volatility Arbitrage**: Exploits volatility spreads between options
- **Mispricing Detection**: Finds overvalued/undervalued options

### Risk Management
- Portfolio-level Greeks calculation
- Delta hedging with ETF positions
- Position size limits and leverage controls
- Value-at-Risk (VaR) monitoring
- Real-time risk limit enforcement

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python trade.py
```

The strategy will:
1. Connect to the trading API
2. Continuously monitor market data
3. Identify trading opportunities
4. Execute trades automatically
5. Manage portfolio risk

## Strategy Components

### 1. Market Data Collection
- Real-time ETF price monitoring
- Options chain data retrieval
- News feed analysis
- Historical price tracking

### 2. Volatility Model
```python
forecast_vol = hist_vol + sentiment_impact + mean_reversion
```
- Historical volatility (rolling 20-day window)
- News sentiment impact (up to 10% adjustment)
- Mean reversion factor (targeting 25% long-term volatility)

### 3. Options Pricing
- Black-Scholes formula implementation
- Greeks calculation for risk management
- Implied volatility extraction
- Mispricing detection algorithms

### 4. Arbitrage Detection
- **Put-Call Parity**: C - P = S - K*exp(-r*T)
- **Volatility Spreads**: Compare implied volatilities across options
- **Price Mispricing**: Theoretical vs market price differences

### 5. Risk Management
- Delta hedging with ETF positions
- Position size limits (max 1000 per position)
- Leverage controls (max 3x leverage)
- VaR limits (5% of portfolio value)

## Trading Logic

1. **Data Collection**: Fetch ETF prices, options data, and news
2. **Volatility Forecast**: Calculate predicted volatility using multiple factors
3. **Opportunity Detection**: Identify mispricing and arbitrage opportunities
4. **Risk Check**: Ensure portfolio stays within risk limits
5. **Trade Execution**: Execute profitable trades with proper position sizing
6. **Hedging**: Delta-hedge portfolio to reduce directional risk
7. **Monitoring**: Track portfolio performance and risk metrics

## Risk Controls

- **Position Limits**: Maximum 1000 units per position
- **Leverage Limits**: Maximum 3x gross leverage
- **VaR Limits**: Daily VaR cannot exceed 5% of portfolio value
- **Delta Hedging**: Automatic hedging when delta exposure > 10 units

## Performance Metrics

The strategy tracks:
- Portfolio value and leverage
- Daily Value-at-Risk (95% confidence)
- Active positions and their values
- Risk limit violations
- Trading opportunities identified

## API Integration

The strategy integrates with the trading API endpoints:
- `/v1/case` - Case information
- `/v1/securities` - Market data and positions
- `/v1/news` - News feed
- `/v1/orders` - Trade execution

## Configuration

Key parameters can be adjusted in the `VolatilityTradingStrategy` class:
- `risk_free_rate`: Risk-free interest rate (default: 5%)
- `max_position_size`: Maximum position size (default: 1000)
- Volatility bounds: 5% to 80%
- Profit thresholds: $1.00 minimum profit
- Risk limits: 3x leverage, 5% VaR

## Strategy Advantages

1. **Multi-Factor Approach**: Combines technical, fundamental, and sentiment analysis
2. **Risk Management**: Comprehensive risk controls and hedging
3. **Arbitrage Focus**: Prioritizes risk-free arbitrage opportunities
4. **Automated Execution**: Fully automated trading with minimal manual intervention
5. **Real-time Monitoring**: Continuous risk and performance monitoring

## Disclaimer

This strategy is designed for educational purposes in the Volatility Trading Case. Past performance does not guarantee future results. Always understand the risks involved in options trading.

