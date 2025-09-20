"""
RIT Market Simulator Volatility Trading Case - Complete Strategy Implementation
Rotman BMO Finance Research and Trading Lab, University of Toronto (C)
All rights reserved.

This implementation provides a comprehensive volatility trading strategy that:
1. Forecasts volatility and identifies mispricing opportunities
2. Implements delta-neutral strategies with proper hedging
3. Manages risk within specified limits
4. Executes trades based on volatility differentials
"""

import warnings
import signal
import requests
from time import sleep
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Black-Scholes libraries
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
import py_vollib.black.implied_volatility as iv

"""
To install py_vollib, use conda install jholdom::py_vollib, since it requires Python versions between 3.6 and 3.8.
If that doesn't work, try:
    conda install anaconda::pip
    pip install py_vollib
"""

# Configure logging to both console and file
import datetime
import os

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Initial log filename (will be updated when case starts)
log_filename = f'logs/volatility_trading_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Function to setup logging with a specific filename
def setup_logging(filename):
    """Setup logging with a specific filename"""
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure new logging with debug level for better opportunity detection
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(filename, mode='w')  # Log file
        ]
    )
    return logging.getLogger(__name__)

# Initial logger setup
logger = setup_logging(log_filename)
logger.info(f"Initial logging setup - {log_filename}")
print(f"All console outputs will be saved to {log_filename}")

# Trading parameters
RISK_FREE_RATE = 0.0
TRADING_DAYS_PER_YEAR = 240
DAYS_PER_MONTH = 20
SECONDS_PER_DAY = 3600
TRADING_SECONDS = 300
DELTA_LIMIT = 7000
PENALTY_RATE = 0.01

# Additional risk and execution controls
MAX_OPTIONS_PER_TICK = 2
DELTA_HEDGE_TRIGGER = 200
NEAR_EXPIRY_TICKS_GATE = 30
SAME_TICKER_COOLDOWN_TICKS = 2
MIN_SMILE_R2 = 0.10

# Position limits - Much more conservative
RTM_GROSS_LIMIT = 10000
RTM_NET_LIMIT = 5000
OPTIONS_GROSS_LIMIT = 500
OPTIONS_NET_LIMIT = 200
MAX_TRADE_SIZE_RTM = 500  # Further reduced to prevent over-exposure
MAX_TRADE_SIZE_OPTIONS = 10  # Much smaller option trades

# Transaction costs
RTM_COMMISSION = 0.01
OPTIONS_COMMISSION = 1.00

class ApiException(Exception):
    """Custom exception for API errors"""
    pass

class VolatilityTradingStrategy:
    """
    Comprehensive volatility trading strategy for RIT simulator.
    
    This strategy:
    1. Monitors volatility forecasts and market prices
    2. Identifies mispricing opportunities in options
    3. Executes delta-neutral trades
    4. Manages risk within specified limits
    """
    
    def __init__(self, session, api_key):
        self.session = session
        self.api_key = api_key
        self.current_volatility = 0.20  # Starting volatility
        self.forecasted_volatility = 0.20
        self.volatility_range = None
        self.positions = {}
        self.portfolio_delta = 0
        self.trade_history = []
        self.news_updates = []
        
        # Initialize position tracking
        self.rtm_position = 0
        self.options_positions = {}
        
        # Case status tracking
        self.case_active = False
        self.last_tick = 0
        self.case_status = 'UNKNOWN'
        
        # No frequency limits - trade whenever opportunities arise
        
        # Risk management
        self.total_trades = 0
        self.case_start_tick = 0  # Track when current case started
        self.delta_penalty_rate = 0.01  # $0.01 per second per delta unit over limit
        self.last_hedge_tick = 0  # Track when last hedge was executed
        self.max_delta_exposure = 0  # Track maximum delta exposure
        self.total_penalties = 0  # Track total penalties incurred
        self.current_log_file = None  # Track current log file
        self.case_number = 0  # Track case number
        # News state tracking
        self.realized_vol_set_tick = None  # Tick when realized vol was last set
        # Time scaling (ticks per day parsed from news; default to 15)
        self.ticks_per_day = 15
        self.trading_days_per_year = TRADING_DAYS_PER_YEAR
        # Execution state
        self.last_trade_tick_by_ticker = {}
    
    def create_case_log_file(self, case_tick):
        """Create a new log file for the current case"""
        self.case_number += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        case_log_file = f'logs/volatility_case_{self.case_number}_{timestamp}_tick{case_tick}.log'
        
        # Setup new logging for this case
        global logger
        logger = setup_logging(case_log_file)
        self.current_log_file = case_log_file
        
        logger.info(f"=== NEW CASE #{self.case_number} STARTED ===")
        logger.info(f"Case log file: {case_log_file}")
        logger.info(f"Case tick: {case_tick}")
        print(f"📁 New case #{self.case_number} - Logging to: {case_log_file}")
        
        return case_log_file
    
    def log_case_summary(self):
        """Log case summary to a master log file"""
        if self.current_log_file:
            summary_file = f'logs/volatility_summary_{datetime.datetime.now().strftime("%Y%m%d")}.log'
            with open(summary_file, 'a') as f:
                f.write(f"\n=== CASE #{self.case_number} SUMMARY ===\n")
                f.write(f"Timestamp: {datetime.datetime.now()}\n")
                f.write(f"Log file: {self.current_log_file}\n")
                f.write(f"Total trades: {self.total_trades}\n")
                f.write(f"Max delta exposure: {self.max_delta_exposure:.0f}\n")
                f.write(f"Total penalties: ${self.total_penalties:.2f}\n")
                f.write(f"Avg penalty/second: ${self.total_penalties/300:.2f}\n")
                f.write("=" * 50 + "\n")
    
    def get_current_tick(self):
        """Get current simulation tick"""
        resp = self.session.get('http://localhost:9999/v1/case')
        if resp.ok:
            case = resp.json()
            return case['tick']
        raise ApiException('Failed to get current tick')
    
    def get_current_position(self, ticker):
        """Get current position for a specific ticker"""
        try:
            securities_data = self.get_securities()
            for security in securities_data:
                if security['ticker'] == ticker:
                    return security.get('position', 0)
            return 0
        except:
            return 0
    
    def would_exceed_limits(self, ticker, action, quantity):
        """Check if trade would exceed position limits"""
        try:
            securities_data = self.get_securities()
            for security in securities_data:
                if security['ticker'] == ticker:
                    current_position = security.get('position', 0)
                    current_size = security.get('size', 0)
                    
                    # Calculate new position after trade
                    if action == 'BUY':
                        new_position = current_position + quantity
                        new_size = current_size + quantity
                    else:  # SELL
                        new_position = current_position - quantity
                        new_size = current_size + quantity  # Size increases for both buy and sell
                    
                    # Check gross and net limits (with safety margin)
                    if abs(new_position) > OPTIONS_NET_LIMIT * 0.8:  # 80% of limit as safety margin
                        return True
                    if new_size > OPTIONS_GROSS_LIMIT * 0.8:  # 80% of limit as safety margin
                        return True
                    return False
            return False
        except:
            return True  # Err on the side of caution
    
    def check_case_status(self):
        """Check if the case is active and running"""
        try:
            resp = self.session.get('http://localhost:9999/v1/case')
            if resp.ok:
                case = resp.json()
                tick = case['tick']
                status = case.get('status', 'UNKNOWN')
                
                # Check if case is active
                if status == 'ACTIVE':
                    # Reset trade counter if this is a new case (tick reset to 0 or first time)
                    if tick < self.last_tick or (self.case_start_tick == 0 and tick > 0):
                        # Create new log file for this case
                        self.create_case_log_file(tick)
                        logger.info(f"New case detected - resetting trade counter (tick: {tick})")
                        self.total_trades = 0
                        self.case_start_tick = tick
                    
                    self.case_active = True
                    self.last_tick = tick
                    self.case_status = status
                    return True, tick, status
                else:
                    self.case_active = False
                    self.case_status = status
                    return False, tick, status
            else:
                logger.error(f"Failed to get case status: {resp.status_code}")
                self.case_active = False
                return False, 0, 'ERROR'
        except Exception as e:
            logger.error(f"Error checking case status: {e}")
            self.case_active = False
            return False, 0, 'ERROR'
    
    def get_securities(self):
        """Get current securities data"""
        resp = self.session.get('http://localhost:9999/v1/securities')
        if resp.ok:
            return resp.json()
        raise ApiException('Failed to get securities data')
    
    def get_news(self):
        """Get latest news updates"""
        resp = self.session.get('http://localhost:9999/v1/news')
        if resp.ok:
            return resp.json()
        raise ApiException('Failed to get news data')
    
    def time_to_expiry(self, tick):
        """Calculate time to expiry in years using ticks-per-year scaling"""
        ticks_remaining = max(0, TRADING_SECONDS - tick)
        ticks_per_year = max(1, self.ticks_per_day * self.trading_days_per_year)
        return ticks_remaining / ticks_per_year
    
    def calculate_implied_volatility(self, option_price, spot, strike, time_to_expiry, option_type):
        """Calculate implied volatility for an option"""
        try:
            if option_type.upper() == 'CALL':
                vol = iv.implied_volatility(option_price, spot, strike, RISK_FREE_RATE, 
                                           time_to_expiry, 'c')
            else:
                vol = iv.implied_volatility(option_price, spot, strike, RISK_FREE_RATE, 
                                           time_to_expiry, 'p')
            # Clamp to a reasonable band to avoid solver blow-ups
            return max(0.01, min(1.5, vol))
        except:
            return np.nan
    
    def calculate_black_scholes_price(self, spot, strike, time_to_expiry, volatility, option_type):
        """Calculate Black-Scholes theoretical price"""
        try:
            if option_type.upper() == 'CALL':
                return bs('c', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
            else:
                return bs('p', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
        except:
            return np.nan
    
    def calculate_greeks(self, spot, strike, time_to_expiry, volatility, option_type):
        """Calculate option Greeks"""
        try:
            if option_type.upper() == 'CALL':
                option_delta = delta('c', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
                option_gamma = gamma('c', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
                option_theta = theta('c', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
                option_vega = vega('c', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
            else:
                option_delta = delta('p', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
                option_gamma = gamma('p', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
                option_theta = theta('p', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
                option_vega = vega('p', spot, strike, time_to_expiry, RISK_FREE_RATE, volatility)
            
            return {
                'delta': option_delta,
                'gamma': option_gamma,
                'theta': option_theta,
                'vega': option_vega
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

    def _parse_option_from_ticker(self, ticker):
        """Extract (option_type, strike) from an RTM option ticker like RTM49C/RTM49P"""
        try:
            if ticker == 'RTM' or 'RTM' not in ticker:
                return None, None
            if 'C' in ticker:
                option_type = 'CALL'
                strike_str = ticker.replace('RTM', '').replace('C', '')
            elif 'P' in ticker:
                option_type = 'PUT'
                strike_str = ticker.replace('RTM', '').replace('P', '')
            else:
                return None, None
            strike = float(strike_str)
            return option_type, strike
        except:
            return None, None

    def _mid_price(self, security):
        """Return mid price if bid/ask available, else last."""
        bid = security.get('bid', None)
        ask = security.get('ask', None)
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            return (bid + ask) / 2.0
        return security.get('last', None)

    def get_iv_points(self, securities_data, rtm_price, time_to_expiry):
        """Collect implied volatility observations across strikes for current snapshot"""
        iv_points = []
        for security in securities_data:
            ticker = security['ticker']
            if ticker == 'RTM':
                continue
            option_type, strike = self._parse_option_from_ticker(ticker)
            if option_type is None or strike is None:
                continue
            market_price = self._mid_price(security)
            if market_price is None or market_price <= 0:
                continue
            implied_vol = self.calculate_implied_volatility(market_price, rtm_price, strike, time_to_expiry, option_type)
            if np.isnan(implied_vol) or implied_vol <= 0 or implied_vol > 5.0:
                continue
            iv_points.append({
                'ticker': ticker,
                'strike': strike,
                'option_type': option_type,
                'market_price': market_price,
                'iv': implied_vol
            })
        return iv_points

    def fit_quadratic_smile(self, iv_points, spot):
        """Fit IV = a + b*x + c*x^2 on log-moneyness x=ln(K/spot). Return coefficients, r2, and quantiles.
        Clamps later predictions to empirical IV quantiles to avoid extrapolation blow-ups.
        """
        if len(iv_points) < 3:
            return None
        strikes = np.array([p['strike'] for p in iv_points], dtype=float)
        ivs = np.array([p['iv'] for p in iv_points], dtype=float)
        try:
            # log-moneyness
            xs = np.log(np.maximum(1e-9, strikes / max(1e-9, spot)))
            coeffs = np.polyfit(xs, ivs, 2)  # returns [c, b, a]
            c2, b1, a0 = coeffs[0], coeffs[1], coeffs[2]
            pred = a0 + b1*xs + c2*(xs**2)
            # R^2
            ss_res = np.sum((ivs - pred) ** 2)
            ss_tot = np.sum((ivs - np.mean(ivs)) ** 2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
            # Empirical quantiles for clamping
            q10 = float(np.quantile(ivs, 0.10))
            q90 = float(np.quantile(ivs, 0.90))
            logger.info(f"Fitted IV smile (log-moneyness): a={a0:.4f}, b={b1:.6f}, c={c2:.8f}, R^2={r2:.3f}, q10={q10:.1%}, q90={q90:.1%}")
            return {'a': a0, 'b': b1, 'c': c2, 'r2': r2, 'q10': q10, 'q90': q90}
        except Exception as e:
            logger.debug(f"Smile fit failed: {e}")
            return None

    def predict_smile_iv(self, strike, spot, coeffs):
        """Predict IV from smile coefficients using log-moneyness x=ln(K/spot) with quantile clamping"""
        x = np.log(max(1e-9, strike / max(1e-9, spot)))
        iv_pred = coeffs['a'] + coeffs['b']*x + coeffs['c']*(x**2)
        # Clamp to quantile band (with small buffer)
        lo = max(0.01, coeffs.get('q10', 0.01) * 0.8)
        hi = min(5.0, coeffs.get('q90', 5.0) * 1.2)
        return min(max(iv_pred, lo), hi)

    def identify_smile_mispricings(self, securities_data):
        """Identify mispricings using quadratic smile of IV vs strike.
        Creates opportunities when market option price deviates from BS price computed at smile IV.
        """
        opportunities = []
        current_tick = self.get_current_tick()
        time_to_expiry = self.time_to_expiry(current_tick)
        if time_to_expiry <= 0:
            return opportunities
        # Find RTM price
        rtm_price = None
        security_by_ticker = {}
        for security in securities_data:
            security_by_ticker[security['ticker']] = security
            if security['ticker'] == 'RTM':
                rtm_price = self._mid_price(security)
                break
        if rtm_price is None or rtm_price <= 0:
            return opportunities
        # Collect IV points and fit smile
        iv_points = self.get_iv_points(securities_data, rtm_price, time_to_expiry)
        if len(iv_points) < 3:
            logger.debug("Insufficient IV points to fit smile")
            return opportunities
        coeffs = self.fit_quadratic_smile(iv_points, rtm_price)
        if coeffs is None:
            return opportunities
        # Require minimum smile fit quality
        if coeffs.get('r2', 0.0) < MIN_SMILE_R2:
            logger.debug(f"Smile fit R^2 below threshold ({coeffs.get('r2', 0.0):.3f} < {MIN_SMILE_R2:.2f}); skipping smile trades this tick")
            return opportunities
        # Generate opportunities by comparing market price to BS price using smile IV
        for p in iv_points:
            strike = p['strike']
            option_type = p['option_type']
            sec = security_by_ticker.get(p['ticker'], {})
            market_price = p['market_price']
            bid = sec.get('bid', None)
            ask = sec.get('ask', None)
            # Spread filters
            rel_spread_ok = True
            if bid is not None and ask is not None and bid > 0 and ask >= bid:
                mid = (bid + ask) / 2.0
                abs_spread = ask - bid
                rel_spread = abs_spread / max(0.01, mid)
                # Require reasonable liquidity
                if rel_spread > 0.08 or abs_spread > 0.10:
                    rel_spread_ok = False
            if not rel_spread_ok:
                continue

            # ATM proximity filter early in session
            moneyness = strike / max(1e-9, rtm_price)
            if abs(moneyness - 1.0) > 0.10 and current_tick < TRADING_SECONDS * 0.5:
                # Skip far OTM/ITM early; focus near-ATM to reduce tail risk
                continue

            smile_iv = max(0.0001, min(1.5, self.predict_smile_iv(strike, rtm_price, coeffs)))
            theo_smile = self.calculate_black_scholes_price(rtm_price, strike, time_to_expiry, smile_iv, option_type)
            # Anchor theo to analysts' forecast to capture learning lag: blend 60% smile, 40% forecast
            theo_forecast = self.calculate_black_scholes_price(rtm_price, strike, time_to_expiry, max(0.01, min(1.5, self.forecasted_volatility)), option_type)
            theo_price = np.nanmean([theo_smile, theo_forecast]) if not (np.isnan(theo_smile) and np.isnan(theo_forecast)) else theo_smile
            if np.isnan(theo_price) or theo_price <= 0:
                continue
            price_diff = market_price - theo_price  # positive => market overpriced
            # Fee-aware thresholds with vega-normalization
            iv_residual = p['iv'] - smile_iv
            greeks = self.calculate_greeks(rtm_price, strike, time_to_expiry, smile_iv, option_type)
            vega_val = abs(greeks.get('vega', 0.0))
            # Convert price diff to implied vol points via vega
            vol_equiv = abs(price_diff) / max(vega_val, 1e-6)
            # Commission + spread per contract: ~$1.02; require 1.5x to be safe
            fee_buffer = 1.5 * (OPTIONS_COMMISSION + 0.02)
            base_price_threshold = max(fee_buffer/100.0, 0.003 * theo_price)  # scaled per share of 100
            vol_equiv_threshold = 0.003  # 0.3 vol points
            min_vega = 0.02  # require minimum sensitivity
            if (abs(price_diff) < base_price_threshold and vol_equiv < vol_equiv_threshold) or vega_val < min_vega:
                continue
            # Bias to SELL unless BUY edge beats fees by 2x (to avoid small long theta drags)
            if price_diff > 0:
                action = 'SELL'
            else:
                if abs(price_diff) >= 2.0 * base_price_threshold:
                    action = 'BUY'
                else:
                    continue
            opportunity = {
                'ticker': p['ticker'],
                'option_type': option_type,
                'strike': strike,
                'market_price': market_price,
                'smile_iv': smile_iv,
                'iv_residual': iv_residual,
                'theo_price': theo_price,
                'price_diff': price_diff,
                'greeks': greeks,
                'action': action,
                'confidence': min(max(vol_equiv, abs(price_diff) / max(0.10, theo_price)), 1.0)
            }
            opportunities.append(opportunity)
        # Sort by absolute price mispricing first, fallback to residual magnitude
        opportunities.sort(key=lambda x: (abs(x['price_diff']), abs(x['iv_residual'])), reverse=True)
        logger.info(f"Smile-based opportunities found: {len(opportunities)}")
        if opportunities:
            top = opportunities[0]
            logger.info(f"Top smile opp: {top['ticker']} {top['action']} | diff=${top['price_diff']:.2f} | smileIV={top['smile_iv']:.1%} | res={top['iv_residual']:.2%}")
        return opportunities
    
    def identify_mispricing_opportunities(self, securities_data):
        """Identify options that are mispriced relative to forecasted volatility"""
        opportunities = []
        current_tick = self.get_current_tick()
        time_to_expiry = self.time_to_expiry(current_tick)
        
        if time_to_expiry <= 0:
            return opportunities
        
        # Get RTM price
        rtm_price = None
        for security in securities_data:
            if security['ticker'] == 'RTM':
                rtm_price = security['last']
                break
        
        if rtm_price is None:
            return opportunities
        
        # Analyze each option
        for security in securities_data:
            ticker = security['ticker']
            if ticker == 'RTM':
                continue
                
            # Determine option type and strike
            if 'C' in ticker:
                option_type = 'CALL'
                # Extract strike price from ticker like "RTM49C" -> 49
                strike_str = ticker.replace('RTM', '').replace('C', '')
                strike = float(strike_str)
            elif 'P' in ticker:
                option_type = 'PUT'
                # Extract strike price from ticker like "RTM49P" -> 49
                strike_str = ticker.replace('RTM', '').replace('P', '')
                strike = float(strike_str)
            else:
                continue
            
            market_price = security['last']
            if market_price <= 0:
                continue
            
            # Calculate implied volatility
            implied_vol = self.calculate_implied_volatility(
                market_price, rtm_price, strike, time_to_expiry, option_type
            )
            
            if np.isnan(implied_vol):
                continue
            
            # Calculate Greeks using forecasted volatility
            greeks = self.calculate_greeks(
                rtm_price, strike, time_to_expiry, self.forecasted_volatility, option_type
            )
            
            # Core strategy: Compare implied vs forecasted volatility
            vol_diff = implied_vol - self.forecasted_volatility
            
            # More reasonable criteria for trading - lower threshold for more opportunities
            min_vol_diff = 0.02  # 2% volatility difference (more reasonable threshold)
            
            # Only trade if volatility difference is significant AND reasonable
            if vol_diff > min_vol_diff and implied_vol < 5.0:  # Allow higher implied vol (up to 500%)
                # Implied vol > forecasted vol: Option is overpriced, SELL
                opportunity = {
                    'ticker': ticker,
                    'option_type': option_type,
                    'strike': strike,
                    'market_price': market_price,
                    'implied_vol': implied_vol,
                    'vol_diff': vol_diff,
                    'greeks': greeks,
                    'action': 'SELL',
                    'confidence': min(vol_diff / self.forecasted_volatility, 1.0)
                }
                opportunities.append(opportunity)
                logger.info(f"SELL opportunity: {ticker} - Implied vol: {implied_vol:.1%} vs Forecast: {self.forecasted_volatility:.1%}")
                
            elif vol_diff < -min_vol_diff and implied_vol > 0.01:  # Allow lower implied vol (down to 1%)
                # Implied vol < forecasted vol: Option is underpriced, BUY
                opportunity = {
                    'ticker': ticker,
                    'option_type': option_type,
                    'strike': strike,
                    'market_price': market_price,
                    'implied_vol': implied_vol,
                    'vol_diff': vol_diff,
                    'greeks': greeks,
                    'action': 'BUY',
                    'confidence': min(abs(vol_diff) / self.forecasted_volatility, 1.0)
                }
                opportunities.append(opportunity)
                logger.info(f"BUY opportunity: {ticker} - Implied vol: {implied_vol:.1%} vs Forecast: {self.forecasted_volatility:.1%}")
        
        # Sort by volatility difference (largest first)
        opportunities.sort(key=lambda x: abs(x['vol_diff']), reverse=True)
        return opportunities
    
    def calculate_portfolio_delta(self, securities_data):
        """Calculate current portfolio delta exposure"""
        total_delta = 0
        current_tick = self.get_current_tick()
        time_to_expiry = self.time_to_expiry(current_tick)
        
        if time_to_expiry <= 0:
            return 0
        
        # Get RTM price and include RTM position delta directly
        rtm_price = None
        for security in securities_data:
            if security['ticker'] == 'RTM':
                rtm_price = security.get('last', None)
                # Include current RTM position as delta (1 share = delta 1)
                try:
                    total_delta += security.get('position', 0)
                except Exception:
                    pass
                break
        
        if rtm_price is None:
            return total_delta
        
        # Calculate delta for each option position
        for security in securities_data:
            ticker = security['ticker']
            if ticker == 'RTM':
                continue
                
            position_contracts = security.get('position', 0)
            if position_contracts == 0:
                continue
            
            # Determine option type and strike
            if 'C' in ticker:
                option_type = 'CALL'
                # Extract strike price from ticker like "RTM49C" -> 49
                strike_str = ticker.replace('RTM', '').replace('C', '')
                strike = float(strike_str)
            elif 'P' in ticker:
                option_type = 'PUT'
                # Extract strike price from ticker like "RTM49P" -> 49
                strike_str = ticker.replace('RTM', '').replace('P', '')
                strike = float(strike_str)
            else:
                continue
            
            # Calculate delta for this option
            greeks = self.calculate_greeks(rtm_price, strike, time_to_expiry, self.forecasted_volatility, option_type)
            # Add to total delta: contracts * 100 shares per contract * option delta
            total_delta += position_contracts * 100 * greeks.get('delta', 0.0)
        
        return total_delta
    
    def calculate_required_hedge(self, portfolio_delta):
        """Calculate required RTM shares to hedge portfolio delta"""
        return -portfolio_delta
    
    def submit_order(self, ticker, action, quantity, order_type='MARKET'):
        """Submit order to RIT API"""
        try:
            order_params = {
                'ticker': ticker,
                'type': order_type,
                'quantity': abs(quantity),
                'action': action.upper()
            }
            
            
            resp = self.session.post('http://localhost:9999/v1/orders', params=order_params)
            if resp.ok:
                order_response = resp.json()
                logger.info(f"Order submitted: {ticker} {action} {quantity} - Order ID: {order_response.get('order_id', 'N/A')}")
                return order_response
            else:
                error_msg = resp.text
                if "exceed net trading limits" in error_msg:
                    logger.warning(f"Position limit reached for {ticker} - reducing future trade sizes")
                else:
                    logger.error(f"Failed to submit order: {error_msg}")
                return None
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return None
    
    def execute_hedge(self, required_hedge, rtm_price):
        """Execute delta hedge by trading RTM shares - Stay within position limits"""
        # Adaptive hedging based on market volatility
        # More aggressive hedging during high volatility periods
        vol_threshold = 50 if self.current_volatility < 0.25 else 25  # Lower threshold in high vol
        if abs(required_hedge) < vol_threshold:
            return
        
        # No cooldown for critical risk management
        current_tick = self.get_current_tick()
        if current_tick - self.last_hedge_tick < 1:  # Minimal cooldown for emergency hedging
            logger.debug(f"Skipping hedge - cooldown period (last hedge: tick {self.last_hedge_tick})")
            return
        
        # Balanced hedge sizes for effective risk management
        max_hedge_size = min(100, MAX_TRADE_SIZE_RTM)  # Balanced hedge size
        quantity = min(abs(required_hedge), max_hedge_size)
        
        # Determine action
        if required_hedge > 0:
            action = 'BUY'
        else:
            action = 'SELL'
        
        # Check position limits more strictly
        current_rtm_position = self.rtm_position
        if action == 'BUY':
            new_position = current_rtm_position + quantity
        else:
            new_position = current_rtm_position - quantity
        
        # Respect position limits - don't exceed RTM_NET_LIMIT
        if abs(new_position) > RTM_NET_LIMIT:
            # Calculate maximum allowed quantity
            max_allowed = RTM_NET_LIMIT - abs(current_rtm_position)
            if max_allowed <= 0:
                logger.debug(f"Skipping hedge - would exceed position limit (current: {current_rtm_position})")
                return
            quantity = min(quantity, max_allowed)
        
        if quantity > 0:
            self.submit_order('RTM', action, quantity)
            self.total_trades += 1
            self.last_hedge_tick = current_tick  # Update last hedge tick
            logger.info(f"Executing conservative hedge: {action} {quantity} RTM shares (exposure: {required_hedge}) (trade #{self.total_trades})")
        else:
            # Silently skip when would exceed position limits
            pass
    
    def execute_option_trade(self, opportunity):
        """Execute option trade based on volatility mispricing"""
        ticker = opportunity['ticker']
        action = opportunity['action']
        
        # Smart over-trading prevention:
        # 1. Only trade if we won't exceed position limits
        # 2. Avoid trading if we're already at risk of fines
        # 3. Focus on profitable opportunities
        
        # Check if this trade would exceed limits (avoid fines)
        if self.would_exceed_limits(ticker, action, 1):
            logger.debug(f"Skipping {ticker} - would exceed position limits")
            return
        
        # Check if we're already at high delta exposure (avoid fines)
        current_delta = self.calculate_portfolio_delta(self.get_securities())
        if abs(current_delta) > 5000:  # Don't trade if already at high delta risk
            logger.debug(f"Skipping {ticker} - high delta exposure: {current_delta:.0f}")
            return
        
        # Check if this trade would push us too close to delta limit
        # Calculate delta impact of this trade
        trade_delta = opportunity['greeks']['delta'] * (1 if action == 'BUY' else -1)
        new_delta = current_delta + trade_delta
        if abs(new_delta) > 6000:  # Don't trade if would push us too close to 7000 limit
            logger.debug(f"Skipping {ticker} - would push delta to {new_delta:.0f}")
            return
        
        # Simple position sizing: 1 contract per trade
        quantity = 1
        
        # Execute the trade
        self.submit_order(ticker, action, quantity)
        self.total_trades += 1
        logger.info(f"Executing volatility trade: {action} {quantity} {ticker} (trade #{self.total_trades})")
        
        # Defer hedging to centralized hedger in the main loop
    
    def process_news_updates(self):
        """Process news updates for volatility forecasts"""
        try:
            news_data = self.get_news()
            logger.debug(f"Processing {len(news_data)} news items")
            current_tick = self.get_current_tick()
            
            for news_item in news_data:
                headline = news_item.get('headline', '')
                body = news_item.get('body', '')
                logger.debug(f"News headline: {headline}")
                logger.debug(f"News body: {body}")
                
                if 'volatility' in body.lower():
                    self.news_updates.append(news_item)
                    logger.info(f"Found volatility news: {headline}")
                    
                    # Extract volatility information from news body
                    import re
                    lower_body = body.lower()

                    # Derive ticks-per-day and a first-week cutoff in ticks
                    first_week_cutoff = 5 * max(1, self.ticks_per_day)
                    try:
                        # e.g., "20 trading days that are each 15 ticks in length"
                        days_ticks_match = re.search(r'(\d+)\s+trading\s+days.*?each\s+(\d+)\s+ticks', lower_body)
                        if days_ticks_match:
                            ticks_per_day = int(days_ticks_match.group(2))
                            if ticks_per_day > 0:
                                self.ticks_per_day = ticks_per_day
                                first_week_cutoff = 5 * self.ticks_per_day
                                logger.info(f"Detected ticks_per_day from news: {self.ticks_per_day}")
                    except Exception:
                        pass

                    # Explicitly parse realized volatility to avoid capturing risk-free rate
                    realized_match = re.search(r'current\s+annualized\s+realized\s+volatility\s+is\s+(\d+(?:\.\d+)?)%', lower_body)
                    if realized_match:
                        # Only consider realized vol message during the first week and only once per case
                        if current_tick <= first_week_cutoff and self.realized_vol_set_tick is None:
                            realized_val = float(realized_match.group(1)) / 100
                            self.current_volatility = realized_val
                            self.realized_vol_set_tick = current_tick
                            logger.info(f"Updated current volatility to {self.current_volatility:.1%} (tick {current_tick} within first-week cutoff {first_week_cutoff})")
                        else:
                            logger.debug(f"Ignoring realized vol news at tick {current_tick} (cutoff {first_week_cutoff}, already_set={self.realized_vol_set_tick is not None})")

                    # Parse forecast (range preferred)
                    parsed_any = False
                    if 'next week' in lower_body or 'forecast' in lower_body:
                        range_match = re.search(r'between\s+(\d+(?:\.\d+)?)%\s+and\s+(\d+(?:\.\d+)?)%', lower_body)
                        if range_match:
                            vol_min = float(range_match.group(1)) / 100
                            vol_max = float(range_match.group(2)) / 100
                            self.forecasted_volatility = (vol_min + vol_max) / 2
                            logger.info(f"Updated forecasted volatility to {self.forecasted_volatility:.1%} (range: {vol_min:.1%}-{vol_max:.1%})")
                            parsed_any = True
                        else:
                            # Fallback: capture a percentage near 'next week' or 'forecast' keywords
                            single_forecast_match = re.search(r'(?:next\s+week|forecast)[^%]*?(\d+(?:\.\d+)?)%', lower_body)
                            if single_forecast_match:
                                self.forecasted_volatility = float(single_forecast_match.group(1)) / 100
                                logger.info(f"Updated forecasted volatility to {self.forecasted_volatility:.1%}")
                                parsed_any = True

                    # If nothing matched but there is a percentage, do not misassign risk-free to realized
                    if not parsed_any:
                        generic_match = re.findall(r'(\d+(?:\.\d+)?)%', lower_body)
                        if (not realized_match) and (not ('next week' in lower_body or 'forecast' in lower_body)) and generic_match:
                            # Keep behavior conservative: assign to forecasted, not current, to avoid 0% risk-free capture
                            fallback_val = float(generic_match[-1]) / 100  # prefer the last percentage
                            self.forecasted_volatility = fallback_val
                            logger.info(f"Updated forecasted volatility to {self.forecasted_volatility:.1%}")
                            parsed_any = True
                    if not parsed_any and not realized_match:
                        logger.warning(f"Could not extract volatility from: {body}")
                else:
                    logger.debug(f"Non-volatility news: {headline}")
            
            logger.info(f"News processing complete - Current: {self.current_volatility:.1%}, Forecasted: {self.forecasted_volatility:.1%}")
        except Exception as e:
            logger.error(f"Error processing news: {e}")
    
    def run_strategy(self):
        """Main strategy execution loop"""
        logger.info("Starting volatility trading strategy")
        logger.info("Waiting for case to become active...")
        
        while True:
            try:
                # Check if case is active first
                case_active, tick, status = self.check_case_status()
                
                if not case_active:
                    if status != 'ACTIVE':
                        logger.info(f"Case not active (status: {status}, tick: {tick})")
                    sleep(2)
                    continue
                
                # Check if we've reached the end of trading
                if tick >= TRADING_SECONDS:
                    logger.info(f"Trading session completed (tick {tick}/{TRADING_SECONDS})")
                    break
                
                logger.info(f"Processing tick {tick}/{TRADING_SECONDS} (status: {status})")
                
                # Process news updates
                logger.info(f"Before news processing - Current: {self.current_volatility:.1%}, Forecasted: {self.forecasted_volatility:.1%}")
                self.process_news_updates()
                logger.info(f"After news processing - Current: {self.current_volatility:.1%}, Forecasted: {self.forecasted_volatility:.1%}")
                
                # Get current market data
                securities_data = self.get_securities()
                
                # Calculate current portfolio delta
                portfolio_delta = self.calculate_portfolio_delta(securities_data)
                self.portfolio_delta = portfolio_delta
                
                # Track maximum delta exposure
                if abs(portfolio_delta) > self.max_delta_exposure:
                    self.max_delta_exposure = abs(portfolio_delta)
                
                # Check delta limits
                if abs(portfolio_delta) > DELTA_LIMIT:
                    logger.warning(f"WARNING: Portfolio delta {portfolio_delta} exceeds limit {DELTA_LIMIT}")
                
                # Get RTM price for hedging
                rtm_price = None
                for security in securities_data:
                    if security['ticker'] == 'RTM':
                        rtm_price = security['last']
                        break
                
                if rtm_price is None:
                    logger.error("ERROR: Could not get RTM price")
                    sleep(1)
                    continue
                
                # Identify mispricing opportunities using quadratic IV smile
                opportunities = self.identify_smile_mispricings(securities_data)
                
                # Fallback: if no smile-based opps, try forecast-based opportunities
                if not opportunities:
                    fb_opps = self.identify_mispricing_opportunities(securities_data)
                    if fb_opps:
                        logger.info(f"Using forecast-based opportunities: {len(fb_opps)} found")
                        # Map forecast-based opp into the structure expected by execute_option_trade
                        # by adding smile-compatible fields
                        mapped = []
                        for o in fb_opps:
                            # Estimate theo using forecasted vol for pricing-based diff
                            rtmsc = next(s for s in securities_data if s['ticker']=='RTM')
                            rtmmid = self._mid_price(rtmsc)
                            theo = self.calculate_black_scholes_price(
                                rtmmid,
                                o['strike'],
                                self.time_to_expiry(self.get_current_tick()),
                                self.forecasted_volatility,
                                o['option_type']
                            )
                            price_diff = o['market_price'] - (theo if not np.isnan(theo) else o['market_price'])
                            mapped.append({
                                'ticker': o['ticker'],
                                'option_type': o['option_type'],
                                'strike': o['strike'],
                                'market_price': o['market_price'],
                                'smile_iv': o['implied_vol'],
                                'iv_residual': o['vol_diff'],
                                'theo_price': theo,
                                'price_diff': price_diff,
                                'greeks': o['greeks'],
                                'action': o['action'],
                                'confidence': o['confidence']
                            })
                        # Prioritize by absolute price_diff
                        mapped.sort(key=lambda x: abs(x['price_diff']), reverse=True)
                        opportunities = mapped
                
                # Execute trades for volatility mispricing opportunities (with smart filtering)
                traded_this_tick = set()  # Track what we've already traded this tick
                trades_placed = 0
                for opportunity in opportunities:
                    # Don't trade the same option multiple times in one tick
                    if opportunity['ticker'] in traded_this_tick:
                        continue
                    # Per-ticker cooldown
                    last_tick_for_ticker = self.last_trade_tick_by_ticker.get(opportunity['ticker'])
                    if last_tick_for_ticker is not None and (tick - last_tick_for_ticker) < SAME_TICKER_COOLDOWN_TICKS:
                        continue
                    # Trade count limit per tick
                    if trades_placed >= MAX_OPTIONS_PER_TICK:
                        break
                    # Suppress new option trades when near expiry or delta exceeds gate
                    ticks_remaining = TRADING_SECONDS - tick
                    if abs(portfolio_delta) > DELTA_HEDGE_TRIGGER or ticks_remaining <= NEAR_EXPIRY_TICKS_GATE:
                        break
                    
                    # Execute the trade
                    self.execute_option_trade(opportunity)
                    traded_this_tick.add(opportunity['ticker'])
                    self.last_trade_tick_by_ticker[opportunity['ticker']] = tick
                    trades_placed += 1
                
                # Execute delta hedge - Stay within ±7000 limit
                required_hedge = self.calculate_required_hedge(portfolio_delta)
                
                # Target near-flat delta every loop if beyond small band
                if abs(portfolio_delta) > DELTA_HEDGE_TRIGGER:
                    logger.info(f"Hedging delta exposure: {portfolio_delta:.0f}")
                    self.execute_hedge(required_hedge, rtm_price)

                # End-of-session flattening: last 2 ticks, flatten RTM and place no new option trades
                if TRADING_SECONDS - tick <= 2:
                    live_pos = 0
                    try:
                        live_pos = self.get_current_position('RTM')
                    except Exception:
                        pass
                    if live_pos != 0:
                        flatten_action = 'SELL' if live_pos > 0 else 'BUY'
                        self.submit_order('RTM', flatten_action, abs(int(live_pos)))
                        logger.info(f"Final flatten RTM: {flatten_action} {abs(int(live_pos))} shares at tick {tick}")
                
                # Calculate delta penalty if over limit
                delta_penalty = 0
                if abs(portfolio_delta) > DELTA_LIMIT:
                    excess_delta = abs(portfolio_delta) - DELTA_LIMIT
                    delta_penalty = excess_delta * self.delta_penalty_rate
                    self.total_penalties += delta_penalty
                    logger.warning(f"ALERT: DELTA LIMIT EXCEEDED! Penalty: ${delta_penalty:.2f}/second (Total: ${self.total_penalties:.2f})")
                
                # Log current status with more detail
                logger.info(f"Portfolio Delta: {portfolio_delta:.0f}, Required Hedge: {required_hedge:.0f}")
                logger.info(f"Current Volatility: {self.current_volatility:.1%}, Forecasted: {self.forecasted_volatility:.1%}")
                logger.info(f"Found {len(opportunities)} trading opportunities")
                
                # Always use market-implied volatility as primary source
                logger.info("Calculating market-implied volatility from options")
                # Calculate average implied volatility from options
                total_implied_vol = 0
                vol_count = 0
                valid_vols = []
                
                for security in securities_data:
                    if 'RTM' in security['ticker'] and security['ticker'] != 'RTM':
                        # Calculate implied vol for this option
                        try:
                            strike_str = security['ticker'].replace('RTM', '').replace('C', '').replace('P', '')
                            strike = float(strike_str)
                            option_type = 'CALL' if 'C' in security['ticker'] else 'PUT'
                            
                            implied_vol = self.calculate_implied_volatility(
                                security['last'], rtm_price, strike,
                                self.time_to_expiry(tick), option_type
                            )
                            if not np.isnan(implied_vol) and 0.01 < implied_vol < 5.0:  # More reasonable range (up to 500%)
                                total_implied_vol += implied_vol
                                vol_count += 1
                                valid_vols.append(implied_vol)
                                logger.debug(f"Option {security['ticker']}: Implied vol {implied_vol:.1%}")
                            else:
                                logger.debug(f"Option {security['ticker']}: Invalid implied vol {implied_vol}")
                        except Exception as e:
                            logger.debug(f"Failed to calculate implied vol for {security['ticker']}: {e}")
                            continue
                
                if vol_count > 0:
                    # Use median instead of mean for more robust estimation
                    valid_vols.sort()
                    if len(valid_vols) % 2 == 0:
                        market_implied_vol = (valid_vols[len(valid_vols)//2 - 1] + valid_vols[len(valid_vols)//2]) / 2
                    else:
                        market_implied_vol = valid_vols[len(valid_vols)//2]
                    
                    self.forecasted_volatility = market_implied_vol
                    logger.info(f"Using market-implied volatility: {market_implied_vol:.1%} (from {vol_count} options)")
                else:
                    logger.warning("Could not calculate market-implied volatility - using default 20%")
                    logger.debug(f"Total implied vol: {total_implied_vol}, Vol count: {vol_count}")
                
                # Log top opportunities
                if opportunities:
                    for i, opp in enumerate(opportunities[:3]):  # Log top 3 opportunities
                        logger.info(f"Opportunity {i+1}: {opp['ticker']} {opp['action']} - price_diff=${opp['price_diff']:.2f}, smileIV={opp['smile_iv']:.1%}")
                
                # Add debugging for opportunity detection
                if len(opportunities) == 0:
                    logger.debug(f"No opportunities found at tick {tick} - checking market conditions...")
                    # Log some market data for debugging
                    for security in securities_data:
                        if 'RTM' in security['ticker'] and security['ticker'] != 'RTM':
                            # Calculate implied vol for debugging
                            try:
                                strike_str = security['ticker'].replace('RTM', '').replace('C', '').replace('P', '')
                                strike = float(strike_str)
                                option_type = 'CALL' if 'C' in security['ticker'] else 'PUT'
                                
                                implied_vol = self.calculate_implied_volatility(
                                    security['last'], rtm_price, strike,
                                    self.time_to_expiry(tick), option_type
                                )
                                vol_diff = implied_vol - self.forecasted_volatility if not np.isnan(implied_vol) else 0
                                logger.debug(f"Option {security['ticker']}: Price=${security['last']:.2f}, Implied vol={implied_vol:.1%}, Vol diff={vol_diff:.3f}")
                            except:
                                logger.debug(f"Option {security['ticker']}: Price=${security['last']:.2f}, Could not calculate implied vol")
                
                # Show details of top opportunity
                if opportunities:
                    top_opp = opportunities[0]
                    logger.info(f"Top opportunity: {top_opp['ticker']} {top_opp['action']} - price_diff=${top_opp['price_diff']:.2f}, smileIV={top_opp['smile_iv']:.1%}")
                
                logger.info(f"Total trades executed: {self.total_trades}")
                if delta_penalty > 0:
                    logger.warning(f"Delta penalty: ${delta_penalty:.2f}/second")
                
                sleep(0.5)  # Small delay to avoid overwhelming the API
                
            except KeyboardInterrupt:
                logger.info("Strategy stopped by user")
                break
            except Exception as e:
                logger.error(f"ERROR: Error in strategy execution: {e}")
                sleep(1)
        
        # Log performance summary
        logger.info("Trading session completed")
        logger.info(f"Performance Summary:")
        logger.info(f"  - Case number: {self.case_number}")
        logger.info(f"  - Log file: {self.current_log_file}")
        logger.info(f"  - Total trades executed: {self.total_trades}")
        logger.info(f"  - Maximum delta exposure: {self.max_delta_exposure:.0f}")
        logger.info(f"  - Total penalties incurred: ${self.total_penalties:.2f}")
        logger.info(f"  - Average penalty per second: ${self.total_penalties/300:.2f}")
        logger.info(f"=== CASE #{self.case_number} COMPLETED ===")
        
        # Log case summary to master log
        self.log_case_summary()

# Global variables for signal handling
shutdown = False
API_KEY = {'X-API-Key': '5E300MMZ'}

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True
    logger.info("Shutdown signal received")

def main():
    """Main execution function"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create session
    session = requests.Session()
    session.headers.update(API_KEY)
    
    try:
        # Initialize strategy
        strategy = VolatilityTradingStrategy(session, API_KEY)
        
        # Wait for case to start
        logger.info("Checking case status...")
        case_active, tick, status = strategy.check_case_status()
        
        if not case_active:
            logger.info(f"Case not yet active (status: {status}, tick: {tick})")
            logger.info("Waiting for case to start...")
            
            # Wait for case to become active
            while not case_active and not shutdown:
                sleep(2)
                case_active, tick, status = strategy.check_case_status()
                if not case_active:
                    logger.info(f"Still waiting... (status: {status}, tick: {tick})")
        
        if case_active:
            logger.info(f"Case is active! Starting strategy (tick: {tick}, status: {status})")
            # Reset trade counter for new case
            strategy.total_trades = 0
            strategy.case_start_tick = tick
            strategy.run_strategy()
        else:
            logger.warning("Case did not become active")
            
    except KeyboardInterrupt:
        logger.info("Strategy interrupted by user")
    except Exception as e:
        logger.error(f"ERROR: Strategy failed: {e}")
    finally:
        logger.info("Strategy execution completed")

# Legacy functions for backward compatibility
def get_tick(session):
    """Legacy function for getting current tick"""
    resp = session.get('http://localhost:9999/v1/case')
    if resp.ok:
        case = resp.json()
        return case['tick']
    raise ApiException('fail - cannot get tick')

def get_s(session):
    """Legacy function for getting securities"""
    price_act = session.get('http://localhost:9999/v1/securities')
    if price_act.ok:
        prices = price_act.json()
        return prices
    raise ApiException('fail - cannot get securities')

def years_r(mat, tick):
    """Legacy function for calculating time to expiry"""
    yr = (mat - tick)/3600 
    return yr

# Legacy main function for backward compatibility
def legacy_main():
    """Legacy main function with original logic"""
    vol = 0.25  # Default volatility estimate
    
    with requests.Session() as session:
        session.headers.update(API_KEY)
        while get_tick(session) < 300 and not shutdown:
            assets = pd.DataFrame(get_s(session))
            assets2 = assets.drop(columns=['vwap', 'nlv', 'bid_size', 'ask_size', 'volume', 'realized', 'unrealized', 'currency', 
                                           'total_volume', 'limits', 'is_tradeable', 'is_shortable', 'interest_rate', 'start_period', 'stop_period', 'unit_multiplier', 
                                           'description', 'unit_multiplier', 'display_unit', 'min_price', 'max_price', 'start_price', 'quoted_decimals', 'trading_fee', 'limit_order_rebate',
                                           'min_trade_size', 'max_trade_size', 'required_tickers', 'underlying_tickers', 'bond_coupon', 'interest_payments_per_period', 'base_security', 'fixing_ticker',
                                           'api_orders_per_second', 'execution_delay_ms', 'interest_rate_ticker', 'otc_price_range'])
            helper = pd.DataFrame(index = range(1),columns = ['share_exposure', 'required_hedge', 'must_be_traded', 'current_pos', 'required_pos', 'SAME?'])
            assets2['delta'] = np.nan
            assets2['i_vol'] = np.nan
            assets2['bsprice'] = np.nan
            assets2['diffcom'] = np.nan
            assets2['abs_val'] = np.nan
            assets2['decision'] = np.nan
            
            for row in assets2.index.values:
                if 'P' in assets2['ticker'].iloc[row]:
                    assets2['type'].iloc[row] = 'PUT'
                    market_price = assets2['last'].iloc[row]
                    
                    if get_tick(session) < 300:
                        assets2['delta'].iloc[row] = delta('p', assets2['last'].iloc[0], float(assets2['ticker'].iloc[row][3:5]), 
                                                           years_r(300, get_tick(session)), 0, vol)
                        assets2['bsprice'].iloc[row] = bs('p', assets2['last'].iloc[0], float(assets2['ticker'].iloc[row][3:5]), 
                                                           years_r(300, get_tick(session)), 0, vol)
                        
                        try:
                            assets2['i_vol'].iloc[row] = iv.implied_volatility(assets2['last'].iloc[row], assets2['last'].iloc[0],
                                                                      float(assets2['ticker'].iloc[row][3:5]), 0, years_r(300, get_tick(session)),
                                                                      'p')
                        except Exception as e:
                            print(f"Implied volatility error {e}")
                            assets2['i_vol'].iloc[row] = np.nan
                        
                elif 'C' in assets2['ticker'].iloc[row]:
                    assets2['type'].iloc[row] = 'CALL'
                    if get_tick(session) < 300:
                        assets2['delta'].iloc[row] = delta('c', assets2['last'].iloc[0], float(assets2['ticker'].iloc[row][3:5]), 
                                                           years_r(300, get_tick(session)), 0, vol)
                        assets2['bsprice'].iloc[row] = bs('c', assets2['last'].iloc[0], float(assets2['ticker'].iloc[row][3:5]), 
                                                           years_r(300, get_tick(session)), 0, vol)
                        
                        try:
                            assets2['i_vol'].iloc[row] = iv.implied_volatility(assets2['last'].iloc[row], assets2['last'].iloc[0],
                                                                      float(assets2['ticker'].iloc[row][3:5]), 0, years_r(300, get_tick(session)),
                                                                      'c')
                        except Exception as e:
                            print(f"Implied volatility error {e}")
                            assets2['i_vol'].iloc[row] = np.nan
                            
                if assets2['last'].iloc[row] - assets2['bsprice'].iloc[row] > 0:
                    assets2['diffcom'].iloc[row] = assets2['last'].iloc[row] - assets2['bsprice'].iloc[row] - 0.02
                    assets2['abs_val'].iloc[row] = abs(assets2['diffcom'].iloc[row])
                elif assets2['last'].iloc[row] - assets2['bsprice'].iloc[row] < 0:
                    assets2['diffcom'].iloc[row] = assets2['last'].iloc[row] - assets2['bsprice'].iloc[row] + 0.02
                    assets2['abs_val'].iloc[row] = abs(assets2['diffcom'].iloc[row])
                if assets2['diffcom'].iloc[row] > 0.02:
                    assets2['decision'].iloc[row] = 'SELL'
                elif assets2['diffcom'].iloc[row] < -0.02:
                    assets2['decision'].iloc[row] = 'BUY'
                else:
                    assets2['decision'].iloc[row] = 'NO DECISION'
                warnings.filterwarnings('ignore')
                
            a1 = np.array(assets2['position'].iloc[1:])
            a2 = np.array(assets2['size'].iloc[1:])
            a3 = np.array(assets2['delta'].iloc[1:])
            
            helper['share_exposure'] = np.nansum(a1 * a2 * a3)
            helper['required_hedge'] = helper['share_exposure'].iloc[0] * -1
            helper['must_be_traded'] = helper['required_hedge']/assets2['position'].iloc[0] - assets2['position'].iloc[0]
            if assets2['position'].iloc[0] > 0:
                helper['current_pos'] = 'LONG'
            elif assets2['position'].iloc[0] < 0:
                helper['current_pos'] = 'SHORT'
            else:
                helper['current_pos'] = 'NO POSITION'
            if helper['required_hedge'].iloc[0] > 0:
                helper['required_pos'] = 'LONG'
            elif helper['required_hedge'].iloc[0] < 0:
                helper['required_pos'] = 'SHORT'
            else:
                helper['required_pos'] = 'NO POSITION'
            helper['SAME?'] = (helper['required_pos'] == helper['current_pos'])
            print(assets2.to_markdown(), end='\n'*2)
            print(helper.to_markdown(), end='\n'*2)
            
            sleep(0.5)

if __name__ == '__main__':
    # Run the new comprehensive strategy by default
    main()
