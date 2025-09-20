# Volatility Trading Strategy for RIT Simulator

## Overview

This implementation provides a comprehensive volatility trading strategy for the Rotman Interactive Trader (RIT) simulator. The strategy is designed to identify mispricing opportunities in options markets and execute delta-neutral trades to profit from volatility differentials.

## Key Features

### 1. Volatility Forecasting
- Processes news updates to extract volatility forecasts
- Maintains current and forecasted volatility estimates
- Updates strategy based on weekly volatility announcements

### 2. Mispricing Detection
- Calculates theoretical option prices using Black-Scholes model
- Compares market prices with theoretical prices
- Identifies overpriced and underpriced options
- Ranks opportunities by confidence and potential profit

### 3. Delta Hedging
- Calculates portfolio delta exposure in real-time
- Executes delta-neutral hedging using RTM shares
- Maintains delta exposure within specified limits (±7,000)
- Monitors and logs delta violations

### 4. Risk Management
- Enforces position limits for RTM (50,000 shares) and options (2,500 contracts)
- Implements transaction cost considerations
- Monitors portfolio exposure continuously
- Provides comprehensive logging and error handling

## Strategy Components

### VolatilityTradingStrategy Class

The main strategy class that orchestrates all trading activities:

```python
strategy = VolatilityTradingStrategy(session, api_key)
strategy.run_strategy()
```

#### Key Methods:

1. **`identify_mispricing_opportunities()`**
   - Analyzes all available options
   - Calculates theoretical prices using forecasted volatility
   - Identifies profitable trading opportunities
   - Returns ranked list of opportunities

2. **`calculate_portfolio_delta()`**
   - Calculates total portfolio delta exposure
   - Accounts for all option positions
   - Updates in real-time

3. **`execute_hedge()`**
   - Executes delta hedging trades
   - Respects position limits
   - Optimizes trade sizes

4. **`process_news_updates()`**
   - Processes volatility news announcements
   - Updates current and forecasted volatility
   - Extracts volatility ranges from news

## Trading Logic

### Option Analysis
For each option, the strategy:
1. Calculates theoretical price using forecasted volatility
2. Computes implied volatility from market price
3. Determines if option is overpriced or underpriced
4. Calculates Greeks (delta, gamma, theta, vega)
5. Ranks by confidence and potential profit

### Trading Decisions
- **Overpriced Options**: Sell when market price > theoretical price + threshold
- **Underpriced Options**: Buy when market price < theoretical price - threshold
- **Delta Hedging**: Execute RTM trades to maintain delta neutrality

### Risk Controls
- Maximum 3 concurrent option trades
- Delta exposure limit: ±7,000
- Position limits enforced for all securities
- Transaction costs considered in profit calculations

## Usage

### Basic Usage
```python
python volatility.py
```

### Advanced Configuration
You can modify key parameters at the top of the file:

```python
# Trading parameters
DELTA_LIMIT = 7000
PENALTY_RATE = 0.01

# Position limits
RTM_GROSS_LIMIT = 50000
OPTIONS_GROSS_LIMIT = 2500

# Transaction costs
RTM_COMMISSION = 0.01
OPTIONS_COMMISSION = 1.00
```

## Strategy Flow

1. **Initialization**
   - Set up API connection
   - Initialize volatility estimates
   - Configure logging

2. **Main Loop** (runs every 0.5 seconds)
   - Process news updates
   - Get current market data
   - Calculate portfolio delta
   - Identify trading opportunities
   - Execute option trades
   - Execute delta hedge
   - Log status updates

3. **News Processing**
   - Monitor for volatility announcements
   - Extract current week volatility
   - Extract next week volatility ranges
   - Update forecasted volatility

4. **Risk Management**
   - Monitor delta exposure
   - Check position limits
   - Execute hedging trades
   - Log violations and warnings

## Key Parameters

### Volatility Thresholds
- Minimum profit threshold: $0.05
- Volatility difference threshold: 2%
- Confidence calculation: Based on price difference percentage

### Position Sizing
- Base option quantity: 10 contracts
- Confidence multiplier: Applied to base quantity
- Maximum trade size: 100 contracts (options), 10,000 shares (RTM)

### Risk Limits
- Delta limit: ±7,000
- RTM net limit: 50,000 shares
- Options net limit: 1,000 contracts

## Logging and Monitoring

The strategy provides comprehensive logging:
- Trade executions
- Delta exposure updates
- Volatility forecast changes
- Risk limit violations
- Error handling and recovery

## Error Handling

- Graceful handling of API errors
- Robust option pricing calculations
- Fallback mechanisms for failed calculations
- Comprehensive exception handling

## Performance Considerations

- Optimized API calls
- Efficient option pricing calculations
- Minimal computational overhead
- Real-time risk monitoring

## Dependencies

- `py_vollib`: Black-Scholes option pricing
- `pandas`: Data manipulation
- `numpy`: Numerical calculations
- `requests`: API communication
- `logging`: Comprehensive logging

## Installation

```bash
# Install required packages
pip install py_vollib pandas numpy requests

# For py_vollib specifically:
conda install jholdom::py_vollib
# OR
pip install py_vollib
```

## Strategy Advantages

1. **Comprehensive Risk Management**: Maintains delta neutrality while respecting all position limits
2. **Real-time Adaptation**: Responds to volatility forecast updates
3. **Robust Error Handling**: Continues operation despite individual trade failures
4. **Efficient Execution**: Optimized for real-time trading environment
5. **Transparent Logging**: Full visibility into strategy decisions and performance

## Customization

The strategy can be easily customized by modifying:
- Volatility thresholds
- Position sizing rules
- Risk limits
- Trading frequency
- Opportunity ranking criteria

This implementation provides a solid foundation for volatility trading in the RIT simulator while maintaining flexibility for strategy refinement.
