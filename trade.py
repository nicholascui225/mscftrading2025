import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

API_KEY = {'X-API-key': '5E300MMZ'}

class VolatilityTradingStrategy:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(API_KEY)
        self.portfolio = {}
        self.positions = {}
        self.historical_data = []
        self.news_data = []
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.max_position_size = 1000
        
    def get_market_data(self):
        """Fetch current market data including ETF price and options"""
        try:
            # Get case info
            case_resp = self.session.get('http://localhost:9999/v1/case')
            if case_resp.ok:
                case = case_resp.json()
                tick = case['tick']
                print(f'Current tick: {tick}')
            
            # Get ETF price
            etf_resp = self.session.get('http://localhost:9999/v1/securities')
            if etf_resp.ok:
                securities = etf_resp.json()
                etf_price = None
                options_data = []
                
                for security in securities:
                    if security['ticker'] == 'RTM':
                        etf_price = security['last']
                    elif 'call' in security['ticker'].lower() or 'put' in security['ticker'].lower():
                        options_data.append(security)
                
                return etf_price, options_data, tick
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None, [], None
    
    def get_news_data(self):
        """Fetch news data for sentiment analysis"""
        try:
            news_resp = self.session.get('http://localhost:9999/v1/news')
            if news_resp.ok:
                return news_resp.json()
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def calculate_historical_volatility(self, prices, window=20):
        """Calculate historical volatility using rolling window"""
        if len(prices) < 2:
            return 0.2  # Default volatility
        
        returns = np.diff(np.log(prices))
        if len(returns) < window:
            return np.std(returns) * np.sqrt(252)  # Annualized
        
        rolling_vol = []
        for i in range(window, len(returns) + 1):
            vol = np.std(returns[i-window:i]) * np.sqrt(252)
            rolling_vol.append(vol)
        
        return rolling_vol[-1] if rolling_vol else 0.2
    
    def analyze_news_sentiment(self, news_data):
        """Simple sentiment analysis based on news keywords"""
        if not news_data:
            return 0
        
        positive_keywords = ['bull', 'rise', 'gain', 'up', 'positive', 'growth', 'strong']
        negative_keywords = ['bear', 'fall', 'drop', 'down', 'negative', 'weak', 'decline']
        
        sentiment_score = 0
        for news in news_data[-10:]:  # Last 10 news items
            headline = news.get('headline', '').lower()
            for word in positive_keywords:
                if word in headline:
                    sentiment_score += 1
            for word in negative_keywords:
                if word in headline:
                    sentiment_score -= 1
        
        return np.tanh(sentiment_score / 5)  # Normalize to [-1, 1]
    
    def forecast_volatility(self, etf_price, historical_prices, news_data):
        """Forecast future volatility using multiple factors"""
        # Historical volatility
        hist_vol = self.calculate_historical_volatility(historical_prices)
        
        # News sentiment impact
        sentiment = self.analyze_news_sentiment(news_data)
        sentiment_impact = abs(sentiment) * 0.1  # 10% max impact from sentiment
        
        # Mean reversion factor
        mean_vol = 0.25  # Assume long-term mean volatility of 25%
        mean_reversion = 0.1 * (mean_vol - hist_vol)
        
        # Forecast volatility
        forecast_vol = hist_vol + sentiment_impact + mean_reversion
        
        # Ensure reasonable bounds
        forecast_vol = max(0.05, min(0.8, forecast_vol))
        
        return forecast_vol
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type='call'):
        """Calculate Black-Scholes option price"""
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return max(price, 0.01)  # Minimum price of 1 cent
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 1 if (option_type == 'call' and S > K) or (option_type == 'put' and S < K) else 0,
                   'gamma': 0, 'theta': 0, 'vega': 0}
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r*T) * norm.cdf(d2)) if option_type == 'call' else \
                (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                r * K * np.exp(-r*T) * norm.cdf(-d2))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}
    
    def find_implied_volatility(self, market_price, S, K, T, r, option_type='call'):
        """Find implied volatility using binary search"""
        def objective(sigma):
            theoretical_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            return (theoretical_price - market_price) ** 2
        
        try:
            result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
            return result.x if result.success else 0.2
        except:
            return 0.2
    
    def identify_mispricing(self, etf_price, options_data, forecast_vol, T):
        """Identify overvalued/undervalued options"""
        opportunities = []
        
        for option in options_data:
            ticker = option['ticker']
            market_price = option['last']
            strike = option.get('strike', 0)
            option_type = 'call' if 'call' in ticker.lower() else 'put'
            
            if strike == 0:
                continue
            
            # Calculate theoretical price
            theoretical_price = self.black_scholes_price(etf_price, strike, T, 
                                                       self.risk_free_rate, forecast_vol, option_type)
            
            # Calculate implied volatility
            implied_vol = self.find_implied_volatility(market_price, etf_price, strike, T, 
                                                     self.risk_free_rate, option_type)
            
            # Determine if mispriced
            price_diff = theoretical_price - market_price
            vol_diff = forecast_vol - implied_vol
            
            if abs(price_diff) > 0.5 or abs(vol_diff) > 0.05:  # Significant mispricing
                opportunities.append({
                    'ticker': ticker,
                    'strike': strike,
                    'type': option_type,
                    'market_price': market_price,
                    'theoretical_price': theoretical_price,
                    'price_diff': price_diff,
                    'implied_vol': implied_vol,
                    'forecast_vol': forecast_vol,
                    'vol_diff': vol_diff,
                    'recommendation': 'buy' if price_diff > 0 else 'sell'
                })
        
        return opportunities
    
    def calculate_portfolio_greeks(self, positions):
        """Calculate portfolio-level Greeks"""
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        for ticker, position in positions.items():
            if 'RTM' in ticker:  # ETF position
                total_delta += position['quantity']
            else:  # Option position
                # This would need option details to calculate Greeks
                pass
        
        return {'delta': total_delta, 'gamma': total_gamma, 'theta': total_theta, 'vega': total_vega}
    
    def check_put_call_parity(self, etf_price, options_data, T):
        """Check for put-call parity violations"""
        arbitrage_opportunities = []
        
        # Group options by strike price
        strikes = {}
        for option in options_data:
            strike = option.get('strike', 0)
            if strike not in strikes:
                strikes[strike] = {'call': None, 'put': None}
            
            if 'call' in option['ticker'].lower():
                strikes[strike]['call'] = option
            elif 'put' in option['ticker'].lower():
                strikes[strike]['put'] = option
        
        # Check put-call parity for each strike
        for strike, options in strikes.items():
            if options['call'] and options['put']:
                call_price = options['call']['last']
                put_price = options['put']['last']
                
                # Put-call parity: C - P = S - K*exp(-r*T)
                theoretical_diff = etf_price - strike * np.exp(-self.risk_free_rate * T)
                actual_diff = call_price - put_price
                violation = abs(theoretical_diff - actual_diff)
                
                if violation > 0.5:  # Significant violation
                    arbitrage_opportunities.append({
                        'type': 'put_call_parity',
                        'strike': strike,
                        'violation': violation,
                        'call_ticker': options['call']['ticker'],
                        'put_ticker': options['put']['ticker'],
                        'call_price': call_price,
                        'put_price': put_price,
                        'theoretical_diff': theoretical_diff,
                        'actual_diff': actual_diff
                    })
        
        return arbitrage_opportunities
    
    def find_volatility_arbitrage(self, options_data, forecast_vol):
        """Find volatility arbitrage opportunities"""
        arbitrage_opportunities = []
        
        # Group options by expiration and type
        calls = [opt for opt in options_data if 'call' in opt['ticker'].lower()]
        puts = [opt for opt in options_data if 'put' in opt['ticker'].lower()]
        
        # Find options with significantly different implied volatilities
        for i, call1 in enumerate(calls):
            for call2 in calls[i+1:]:
                if call1.get('strike', 0) != call2.get('strike', 0):
                    continue
                
                # Calculate implied volatilities
                iv1 = self.find_implied_volatility(call1['last'], 100, call1.get('strike', 100), 
                                                 30/365, self.risk_free_rate, 'call')
                iv2 = self.find_implied_volatility(call2['last'], 100, call2.get('strike', 100), 
                                                 30/365, self.risk_free_rate, 'call')
                
                vol_spread = abs(iv1 - iv2)
                if vol_spread > 0.1:  # 10% volatility spread
                    arbitrage_opportunities.append({
                        'type': 'volatility_spread',
                        'option1': call1['ticker'],
                        'option2': call2['ticker'],
                        'iv1': iv1,
                        'iv2': iv2,
                        'spread': vol_spread,
                        'recommendation': 'buy_low_vol_sell_high_vol'
                    })
        
        return arbitrage_opportunities
    
    def hedge_portfolio(self, etf_price, positions):
        """Calculate hedging requirements"""
        greeks = self.calculate_portfolio_greeks(positions)
        
        # Delta hedging: buy/sell ETF to neutralize delta
        delta_hedge = -greeks['delta']
        
        return delta_hedge
    
    def calculate_portfolio_risk(self, positions, etf_price):
        """Calculate portfolio risk metrics"""
        total_value = sum(pos['market_value'] for pos in positions.values())
        if total_value == 0:
            return {'var_95': 0, 'max_drawdown': 0, 'leverage': 0, 'total_value': 0, 'gross_exposure': 0}
        
        # Simple VaR calculation (95% confidence)
        portfolio_vol = 0.2  # Assume 20% portfolio volatility
        var_95 = total_value * 1.645 * portfolio_vol * np.sqrt(1/252)  # Daily VaR
        
        # Calculate leverage
        gross_exposure = sum(abs(pos['market_value']) for pos in positions.values())
        leverage = gross_exposure / total_value if total_value > 0 else 0
        
        return {
            'var_95': var_95,
            'leverage': leverage,
            'total_value': total_value,
            'gross_exposure': gross_exposure
        }
    
    def check_risk_limits(self, positions, etf_price):
        """Check if portfolio exceeds risk limits"""
        risk_metrics = self.calculate_portfolio_risk(positions, etf_price)
        
        violations = []
        
        # Leverage limit
        if risk_metrics['leverage'] > 3.0:
            violations.append(f"Leverage too high: {risk_metrics['leverage']:.2f}")
        
        # Position size limits
        for ticker, position in positions.items():
            if abs(position['quantity']) > self.max_position_size:
                violations.append(f"Position too large: {ticker} = {position['quantity']}")
        
        # VaR limit
        if risk_metrics['var_95'] > risk_metrics['total_value'] * 0.05:  # 5% of portfolio
            violations.append(f"VaR too high: {risk_metrics['var_95']:.2f}")
        
        return violations
    
    def execute_trade(self, ticker, quantity, action='buy'):
        """Execute a trade"""
        try:
            if action == 'buy':
                resp = self.session.post(f'http://localhost:9999/v1/orders', 
                                       json={'ticker': ticker, 'type': 'MARKET', 
                                            'quantity': abs(quantity), 'action': 'BUY'})
            else:
                resp = self.session.post(f'http://localhost:9999/v1/orders', 
                                       json={'ticker': ticker, 'type': 'MARKET', 
                                            'quantity': abs(quantity), 'action': 'SELL'})
            
            if resp.ok:
                order = resp.json()
                print(f"Order executed: {action.upper()} {abs(quantity)} {ticker}")
                return order
            else:
                print(f"Order failed: {resp.text}")
                return None
        except Exception as e:
            print(f"Error executing trade: {e}")
            return None
    
    def update_positions(self):
        """Update current positions"""
        try:
            resp = self.session.get('http://localhost:9999/v1/securities')
            if resp.ok:
                securities = resp.json()
                for security in securities:
                    ticker = security['ticker']
                    position = security.get('position', 0)
                    self.positions[ticker] = {
                        'quantity': position,
                        'market_value': security.get('market_value', 0)
                    }
        except Exception as e:
            print(f"Error updating positions: {e}")
    
    def run_strategy(self):
        """Main strategy execution loop"""
        print("Starting Volatility Trading Strategy...")
        
        while True:
            try:
                # Get market data
                etf_price, options_data, tick = self.get_market_data()
                if etf_price is None:
                    print("Failed to get market data")
                    continue
                
                # Get news data
                news_data = self.get_news_data()
                
                # Update historical data
                self.historical_data.append(etf_price)
                if len(self.historical_data) > 100:
                    self.historical_data.pop(0)
                
                # Update positions
                self.update_positions()
                
                # Calculate time to expiration (assuming 1-month options)
                T = 30 / 365  # 30 days to expiration
                
                # Forecast volatility
                forecast_vol = self.forecast_volatility(etf_price, self.historical_data, news_data)
                print(f"Forecasted volatility: {forecast_vol:.3f}")
                
                # Check risk limits first
                risk_violations = self.check_risk_limits(self.positions, etf_price)
                if risk_violations:
                    print(f"Risk violations detected: {risk_violations}")
                    # Reduce positions if risk limits exceeded
                    for violation in risk_violations:
                        if "Position too large" in violation:
                            ticker = violation.split(": ")[1].split(" = ")[0]
                            if ticker in self.positions:
                                excess = abs(self.positions[ticker]['quantity']) - self.max_position_size
                                self.execute_trade(ticker, excess, 'sell' if self.positions[ticker]['quantity'] > 0 else 'buy')
                
                # Identify mispricing opportunities
                opportunities = self.identify_mispricing(etf_price, options_data, forecast_vol, T)
                
                # Check for arbitrage opportunities
                put_call_arbitrage = self.check_put_call_parity(etf_price, options_data, T)
                vol_arbitrage = self.find_volatility_arbitrage(options_data, forecast_vol)
                
                all_opportunities = opportunities + put_call_arbitrage + vol_arbitrage
                
                if all_opportunities:
                    print(f"Found {len(opportunities)} mispricing, {len(put_call_arbitrage)} put-call parity, {len(vol_arbitrage)} volatility arbitrage opportunities")
                    
                    # Execute put-call parity arbitrage first (risk-free)
                    for arb in put_call_arbitrage:
                        if arb['violation'] > 1.0:  # Significant violation
                            # Buy the undervalued side, sell the overvalued side
                            if arb['actual_diff'] > arb['theoretical_diff']:
                                # Call is overpriced, put is underpriced
                                self.execute_trade(arb['put_ticker'], 10, 'buy')
                                self.execute_trade(arb['call_ticker'], 10, 'sell')
                            else:
                                # Put is overpriced, call is underpriced
                                self.execute_trade(arb['call_ticker'], 10, 'buy')
                                self.execute_trade(arb['put_ticker'], 10, 'sell')
                    
                    # Execute volatility arbitrage
                    for arb in vol_arbitrage:
                        if arb['spread'] > 0.15:  # 15% volatility spread
                            # Buy low volatility, sell high volatility
                            if arb['iv1'] < arb['iv2']:
                                self.execute_trade(arb['option1'], 5, 'buy')
                                self.execute_trade(arb['option2'], 5, 'sell')
                            else:
                                self.execute_trade(arb['option2'], 5, 'buy')
                                self.execute_trade(arb['option1'], 5, 'sell')
                    
                    # Execute regular mispricing trades
                    if opportunities:
                        # Sort by potential profit
                        opportunities.sort(key=lambda x: abs(x['price_diff']), reverse=True)
                        
                        # Execute trades for top opportunities
                        for opp in opportunities[:2]:  # Top 2 opportunities
                            if abs(opp['price_diff']) > 1.0:  # Minimum profit threshold
                                quantity = min(10, self.max_position_size // 10)  # Position sizing
                                
                                if opp['recommendation'] == 'buy':
                                    self.execute_trade(opp['ticker'], quantity, 'buy')
                                else:
                                    self.execute_trade(opp['ticker'], quantity, 'sell')
                
                # Hedge portfolio
                delta_hedge = self.hedge_portfolio(etf_price, self.positions)
                if abs(delta_hedge) > 10:  # Significant delta exposure
                    self.execute_trade('RTM', delta_hedge, 'buy' if delta_hedge > 0 else 'sell')
                
                # Print portfolio status
                risk_metrics = self.calculate_portfolio_risk(self.positions, etf_price)
                print(f"Portfolio value: ${risk_metrics['total_value']:.2f}")
                print(f"Leverage: {risk_metrics['leverage']:.2f}")
                print(f"Daily VaR (95%): ${risk_metrics['var_95']:.2f}")
                print(f"Active positions: {len([p for p in self.positions.values() if p['quantity'] != 0])}")
                
                # Print detailed position info
                for ticker, pos in self.positions.items():
                    if pos['quantity'] != 0:
                        print(f"  {ticker}: {pos['quantity']} @ ${pos['market_value']:.2f}")
                
                # Wait before next iteration
                import time
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("Strategy stopped by user")
                break
            except Exception as e:
                print(f"Error in strategy loop: {e}")
                import time
                time.sleep(5)

def main():
    strategy = VolatilityTradingStrategy()
    strategy.run_strategy()

if __name__ == '__main__':
    main()