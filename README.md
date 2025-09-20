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



## ETF Arbitrage Strategy (RITC vs BULL/BEAR)

This repository also includes a dedicated ETF arbitrage system in `Arbitrage.py` for the RIT Market Simulator ETF Arbitrage Case. It trades the USD-quoted ETF `RITC` against its CAD-quoted constituents `BULL` and `BEAR`, using the FX instrument `USD` (USD/CAD) to convert between currencies.

### Instruments
- **RITC**: ETF quoted in USD
- **BULL, BEAR**: Underlying CAD stocks in the ETF basket
- **USD**: USD/CAD FX instrument

### Fair Value and Detection
- **Fair Value (CAD)**: `(BULL + BEAR) * FX_mid`, where `FX_mid = (USD_bid + USD_ask)/2`
- **Convert RITC to CAD**:
  - `RITC_bid_cad = RITC_bid_usd * USD_bid`
  - `RITC_ask_cad = RITC_ask_usd * USD_ask`
- **Mispricing threshold**: `ARB_THRESHOLD_CAD` (default 0.05 CAD per share)
  - **Overpriced**: `RITC_ask_cad > fair_value_bid + threshold`
  - **Underpriced**: `fair_value_ask > RITC_bid_cad + threshold`

### Execution Logic
- **Overpriced (ETF rich)**: Short `RITC` (USD), buy `BULL` and `BEAR` (CAD)
- **Underpriced (ETF cheap)**: Buy `RITC` (USD), short `BULL` and `BEAR` (CAD)
- **Dynamic sizing**: Quantity scales with edge via `min(edge/0.1, 3.0)` multiplier, capped by `MAX_SIZE_EQUITY`, starting from `ORDER_QTY` (default 8000)
- **Costs**: Market fee (`FEE_MKT` = 0.02), limit rebate (`REBATE_LMT` = 0.01)

### Converters and Tenders
- **Converters**: Optional creation/redemption path when converter cost (`CONVERTER_FEE`, default 1500 per 10k shares) is cheaper than estimated market slippage (`SLIPPAGE_THRESHOLD`, default 0.15/share)
- **Tenders**: Evaluates `/tenders` and accepts only positive expected PnL offers, then unwinds via converters or market depending on cost

### Risk Management (Enforced per `Arbitrage.py`)
- **Gross exposure cap**: `MAX_GROSS` (default 750,000), with ETF positions weighted double
- **Net exposure bounds**: `MAX_SHORT_NET` to `MAX_LONG_NET` (defaults −35,000 to +35,000 shares)
- Tracks realized/unrealized PnL, win rate, best edge, and adapts aggressiveness via `optimize_for_profits()`

### Running the ETF Arbitrage
Ensure the ETF Arbitrage case is ACTIVE on the RIT Market Simulator (`http://localhost:9999`) and the API key in `Arbitrage.py` matches your simulator.

```bash
python Arbitrage.py            # live (default)
python Arbitrage.py backtest   # backtest mode
python Arbitrage.py dry_run    # no orders; logs intended actions
```

Logs stream to console and `arbitrage.log`. A summary report prints on exit with trade counts, PnL, exposures, and win-rate metrics.