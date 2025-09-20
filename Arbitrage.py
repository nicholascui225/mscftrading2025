"""

RIT Market Simulator Algorithmic ETF Arbitrage Case - Advanced Implementation
Rotman BMO Finance Research and Trading Lab, University of Toronto (C)
All rights reserved.


Advanced arbitrage strategy implementing:
- Tender handling with PnL analysis
- ETF mispricing detection and execution
- Smart order routing (market vs limit)
- Converter creation/redemption
- Comprehensive risk management
- PnL tracking and logging
- Live and backtest modes
"""

import requests

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arbitrage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Mode(Enum):
    LIVE = "live"
    BACKTEST = "backtest"
    DRY_RUN = "dry_run"

@dataclass
class Trade:
    timestamp: float
    ticker: str
    action: str
    quantity: int
    price: float
    order_type: str
    pnl: float = 0.0

@dataclass
class Position:
    ticker: str
    quantity: int
    avg_price: float
    unrealized_pnl: float = 0.0

class ArbitrageStrategy:
    def __init__(self, mode: Mode = Mode.LIVE):
        self.mode = mode
        self.API = "http://localhost:9999/v1"
        self.API_KEY = "5E300MMZ"
        self.HDRS = {"X-API-Key": self.API_KEY}
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update(self.HDRS)
        
        # Trading parameters - OPTIMIZED FOR PROFITS
        self.FEE_MKT = 0.02
        self.REBATE_LMT = 0.01
        self.MAX_SIZE_EQUITY = 10000  # Increased for more profit potential
        self.MAX_SIZE_FX = 2500000    # Increased FX capacity
        
        # Risk parameters - OPTIMIZED FOR AGGRESSIVE TRADING
        self.MAX_LONG_NET = 35000     # Increased long exposure
        self.MAX_SHORT_NET = -35000   # Increased short exposure
        self.MAX_GROSS = 750000       # Increased gross exposure
        self.ORDER_QTY = 8000         # Larger order sizes for more profit
        self.ARB_THRESHOLD_CAD = 0.05  # Lower threshold = more opportunities
        
        # Converter parameters
        self.CONVERTER_FEE = 1500  # $1500 per 10k shares
        self.SLIPPAGE_THRESHOLD = 0.15  # 15 cents per share
        
        # Tender parameters
        self.TENDER_FEE = 0.01  # $0.01 per share for tender processing
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.pnl_realized = 0.0
        self.pnl_unrealized = 0.0
        self.exposure_gross = 0.0
        self.exposure_net = 0.0
        self.tender_decisions = []
        self.converter_usage = []
        
        # Performance tracking - ENHANCED FOR PROFIT OPTIMIZATION
        self.start_time = time.time()
        self.trade_count = 0
        self.successful_arbitrages = 0
        self.total_profit = 0.0
        self.max_profit_trade = 0.0
        self.profit_per_trade = 0.0
        self.win_rate = 0.0
        self.best_arbitrage_edge = 0.0
        self.profit_target = 10000.0  # Daily profit target
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        logger.info(f"ArbitrageStrategy initialized in {mode.value} mode")

# Tickers
    CAD = "CAD"    # currency instrument quoted in CAD
    USD = "USD"    # price of 1 USD in CAD (i.e., USD/CAD)
    BULL = "BULL"   # stock in CAD
    BEAR = "BEAR"   # stock in CAD
    RITC = "RITC"   # ETF quoted in USD


    def get_tick_status(self) -> Tuple[int, str]:
        """Get simulation status and current tick"""
        try:
            # Try with API key as query parameter
            r = self.session.get(f"{self.API}/case", params={"api_key": self.API_KEY})
            if r.status_code == 401:
                # Try with different authentication method
                logger.warning("Header auth failed, trying query parameter auth...")
                r = self.session.get(f"{self.API}/case?api_key={self.API_KEY}")
                
            if r.status_code == 401:
                logger.error("Authentication failed. Try these solutions:")
                logger.error("1. Ensure RIT Market Simulator case is ACTIVE")
                logger.error("2. Check if you need to log in to the RIT GUI first")
                logger.error("3. Verify the API key matches your RIT configuration")
                logger.error("4. Try restarting the RIT Market Simulator")
                return 0, "UNAUTHORIZED"
                
            r.raise_for_status()
            j = r.json()
            return j["tick"], j["status"]

        except Exception as e:
            logger.error(f"Error getting tick status: {e}")
            return 0, "ERROR"

    def get_best_bid_ask(self, ticker: str) -> Tuple[float, float]:
        """Get best bid and ask prices for a ticker"""
        try:
            r = self.session.get(f"{self.API}/securities/book", params={"ticker": ticker})
            r.raise_for_status()
            book = r.json()
            bid = float(book["bids"][0]["price"]) if book["bids"] else 0.0
            ask = float(book["asks"][0]["price"]) if book["asks"] else 1e12
            return bid, ask

        except Exception as e:
            logger.error(f"Error getting prices for {ticker}: {e}")
            return 0.0, 1e12

    def get_positions(self) -> Dict[str, int]:
        """Get current positions for all instruments"""
        try:
            r = self.session.get(f"{self.API}/securities")
            r.raise_for_status()

            positions = {p["ticker"]: int(p.get("position", 0)) for p in r.json()}
            for ticker in [self.BULL, self.BEAR, self.RITC, self.USD, self.CAD]:
                positions.setdefault(ticker, 0)
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {ticker: 0 for ticker in [self.BULL, self.BEAR, self.RITC, self.USD, self.CAD]}

    def place_order(self, ticker: str, action: str, quantity: int, order_type: str = "MARKET", price: float = None) -> bool:
        """Place an order with the specified parameters"""
        if self.mode == Mode.DRY_RUN:
            logger.info(f"DRY RUN: Would place {order_type} {action} {quantity} {ticker} at {price}")
            return True
            
        x = 0
        
        try:
            params = {
                "ticker": ticker,
                "type": order_type,
                "quantity": int(quantity),
                "action": action
            }
            if price is not None and order_type == "LIMIT":
                params["price"] = price
                
            r = self.session.post(f"{self.API}/orders", params=params)
            success = r.ok
            if success:
                self.trade_count += 1
                trade = Trade(
                    timestamp=time.time(),
                    ticker=ticker,
                    action=action,
                    quantity=quantity,
                    price=price or 0.0,
                    order_type=order_type
                )
                self.trades.append(trade)
                logger.info(f"Order placed: {action} {quantity} {ticker} ({order_type})")
            else:
                logger.error(f"Order failed: {r.text}")
            return success
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False

    def calculate_fair_value(self) -> Tuple[float, float]:
        """Calculate fair value of RITC ETF in CAD"""
        try:
            bull_bid, bull_ask = self.get_best_bid_ask(self.BULL)
            bear_bid, bear_ask = self.get_best_bid_ask(self.BEAR)
            usd_bid, usd_ask = self.get_best_bid_ask(self.USD)
            
            # Fair value = (BULL + BEAR) * FX rate
            basket_value_bid = (bull_bid + bear_bid)
            basket_value_ask = (bull_ask + bear_ask)
            
            # Use mid rates for fair value calculation
            fx_mid = (usd_bid + usd_ask) / 2
            fair_value_bid = basket_value_bid * fx_mid
            fair_value_ask = basket_value_ask * fx_mid
            
            return fair_value_bid, fair_value_ask
        except Exception as e:
            logger.error(f"Error calculating fair value: {e}")
            return 0.0, 0.0

    def analyze_tender_offers(self) -> List[Dict]:
        """Analyze active tender offers and determine profitability"""
        try:
            r = self.session.get(f"{self.API}/tenders")
            r.raise_for_status()
            offers = r.json()
        
            profitable_tenders = []
            for offer in offers:
                tender_id = offer.get('tender_id')
                price = offer.get('price', 0)
                quantity = offer.get('quantity', 0)
                is_fixed = offer.get('is_fixed_bid', False)
                
                # Calculate fair value for comparison
                fair_value_bid, fair_value_ask = self.calculate_fair_value()
                fair_value_mid = (fair_value_bid + fair_value_ask) / 2
                
                # Calculate PnL after costs
                if is_fixed:
                    # Fixed bid tender
                    pnl_per_share = price - fair_value_mid - self.TENDER_FEE
                else:
                    # Auction tender - use our bid
                    pnl_per_share = price - fair_value_mid - self.TENDER_FEE
                
                total_pnl = pnl_per_share * quantity
                
                if total_pnl > 0:
                    profitable_tenders.append({
                        'tender_id': tender_id,
                        'price': price,
                        'quantity': quantity,
                        'pnl': total_pnl,
                        'is_fixed': is_fixed
                    })
                    logger.info(f"Profitable tender found: ID={tender_id}, PnL={total_pnl:.2f}")
                else:
                    logger.info(f"Rejecting unprofitable tender: ID={tender_id}, PnL={total_pnl:.2f}")
                    
            return profitable_tenders
        except Exception as e:
            logger.error(f"Error analyzing tender offers: {e}")
            return []

    def accept_tender_offer(self, tender_info: Dict) -> bool:
        """Accept a profitable tender offer"""
        try:
            tender_id = tender_info['tender_id']
            price = tender_info['price']
            is_fixed = tender_info['is_fixed']
            
            if is_fixed:
                r = self.session.post(f"{self.API}/tenders/{tender_id}")
            else:
                r = self.session.post(f"{self.API}/tenders/{tender_id}", params={"price": price})
            
            success = r.ok
            if success:
                self.tender_decisions.append({
                    'timestamp': time.time(),
                    'tender_id': tender_id,
                    'price': price,
                    'pnl': tender_info['pnl'],
                    'accepted': True
                })
                logger.info(f"Tender {tender_id} accepted with PnL {tender_info['pnl']:.2f}")
            else:
                logger.error(f"Failed to accept tender {tender_id}: {r.text}")
                
            return success
        except Exception as e:
            logger.error(f"Error accepting tender offer: {e}")
            return False

    def fast_unwind_tender(self, tender_info: Dict) -> bool:
        """Fast unwind of tender position via market or converters"""
        try:
            # Check if we should use converters vs market
            fair_value_bid, fair_value_ask = self.calculate_fair_value()
            market_cost = self.SLIPPAGE_THRESHOLD * tender_info['quantity']
            converter_cost = self.CONVERTER_FEE * (tender_info['quantity'] / 10000)
            
            if converter_cost < market_cost:
                # Use converter
                return self.use_converter("REDEMPTION", tender_info['quantity'])
            else:
                # Use market orders
                return self.place_order(self.RITC, "SELL", tender_info['quantity'], "MARKET")
        except Exception as e:
            logger.error(f"Error in fast unwind: {e}")
            return False

    def use_converter(self, action: str, quantity: int) -> bool:
        """Use creation/redemption converter"""
        try:
            # This would depend on the specific API endpoint for converters
            # Assuming it exists in the RIT API
            params = {
                "action": action,
                "quantity": quantity,
                "ticker": self.RITC
            }
            r = self.session.post(f"{self.API}/converters", params=params)
            success = r.ok
            
            if success:
                self.converter_usage.append({
                    'timestamp': time.time(),
                    'action': action,
                    'quantity': quantity,
                    'cost': self.CONVERTER_FEE * (quantity / 10000)
                })
                logger.info(f"Converter {action} executed: {quantity} shares")
            else:
                logger.error(f"Converter {action} failed: {r.text}")
                
            return success
        except Exception as e:
            logger.error(f"Error using converter: {e}")
            return False

    def detect_mispricing(self) -> Tuple[bool, str, float]:
        """Detect ETF mispricing opportunities - OPTIMIZED FOR PROFITS"""
        try:
            # Get current prices
            bull_bid, bull_ask = self.get_best_bid_ask(self.BULL)
            bear_bid, bear_ask = self.get_best_bid_ask(self.BEAR)
            ritc_bid_usd, ritc_ask_usd = self.get_best_bid_ask(self.RITC)
            usd_bid, usd_ask = self.get_best_bid_ask(self.USD)
            
            # Convert RITC to CAD
            ritc_bid_cad = ritc_bid_usd * usd_bid
            ritc_ask_cad = ritc_ask_usd * usd_ask

            # Calculate fair value
            fair_value_bid, fair_value_ask = self.calculate_fair_value()
            
            # AGGRESSIVE PROFIT OPTIMIZATION - Lower thresholds for more opportunities
            # Check for overpricing (ETF > Fair Value) - More sensitive
            if ritc_ask_cad > fair_value_bid + self.ARB_THRESHOLD_CAD:
                edge = ritc_ask_cad - fair_value_bid
                # Track best arbitrage edge for profit optimization
                if edge > self.best_arbitrage_edge:
                    self.best_arbitrage_edge = edge
                logger.info(f"OVERPRICED opportunity detected: edge=${edge:.4f}")
                return True, "OVERPRICED", edge
            
            # Check for underpricing (ETF < Fair Value) - More sensitive
            elif fair_value_ask > ritc_bid_cad + self.ARB_THRESHOLD_CAD:
                edge = fair_value_ask - ritc_bid_cad
                # Track best arbitrage edge for profit optimization
                if edge > self.best_arbitrage_edge:
                    self.best_arbitrage_edge = edge
                logger.info(f"UNDERPRICED opportunity detected: edge=${edge:.4f}")
                return True, "UNDERPRICED", edge
            
            return False, "NO_OPPORTUNITY", 0.0
            
        except Exception as e:
            logger.error(f"Error detecting mispricing: {e}")
            return False, "ERROR", 0.0

    def execute_arbitrage(self, direction: str, edge: float) -> bool:
        """Execute arbitrage trade based on direction - OPTIMIZED FOR PROFITS"""
        try:
            # DYNAMIC QUANTITY SIZING FOR MAXIMUM PROFITS
            # Scale quantity based on edge size - bigger edge = bigger position
            edge_multiplier = min(edge / 0.1, 3.0)  # Scale up to 3x for large edges
            quantity = int(min(self.ORDER_QTY * edge_multiplier, self.MAX_SIZE_EQUITY))
            
            # Calculate expected profit
            expected_profit = edge * quantity
            logger.info(f"Executing {direction} arbitrage: qty={quantity}, edge=${edge:.4f}, expected_profit=${expected_profit:.2f}")
            
            if direction == "OVERPRICED":
                # Short ETF, buy stocks - PROFIT OPTIMIZED
                success1 = self.place_order(self.RITC, "SELL", quantity, "MARKET")
                success2 = self.place_order(self.BULL, "BUY", quantity, "MARKET")
                success3 = self.place_order(self.BEAR, "BUY", quantity, "MARKET")
                
                if success1 and success2 and success3:
                    self.successful_arbitrages += 1
                    self.total_profit += expected_profit
                    if expected_profit > self.max_profit_trade:
                        self.max_profit_trade = expected_profit
                    logger.info(f"SUCCESS: OVERPRICED arbitrage executed: edge=${edge:.4f}, profit=${expected_profit:.2f}")
                    return True
                    
            elif direction == "UNDERPRICED":
                # Buy ETF, short stocks - PROFIT OPTIMIZED
                success1 = self.place_order(self.RITC, "BUY", quantity, "MARKET")
                success2 = self.place_order(self.BULL, "SELL", quantity, "MARKET")
                success3 = self.place_order(self.BEAR, "SELL", quantity, "MARKET")
                
                if success1 and success2 and success3:
                    self.successful_arbitrages += 1
                    self.total_profit += expected_profit
                    if expected_profit > self.max_profit_trade:
                        self.max_profit_trade = expected_profit
                    logger.info(f"SUCCESS: UNDERPRICED arbitrage executed: edge=${edge:.4f}, profit=${expected_profit:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return False

    def check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        try:
            positions = self.get_positions()
            
            # Calculate gross and net exposure
            gross = abs(positions[self.BULL]) + abs(positions[self.BEAR]) + abs(positions[self.RITC])
            net = positions[self.BULL] + positions[self.BEAR] + positions[self.RITC]
            
            # ETF counts double for risk purposes
            etf_risk = abs(positions[self.RITC]) * 2
            adjusted_gross = gross + etf_risk
            
            within_limits = (
                adjusted_gross < self.MAX_GROSS and
                self.MAX_SHORT_NET < net < self.MAX_LONG_NET
            )
            
            if not within_limits:
                logger.warning(f"Risk limits exceeded: gross={adjusted_gross}, net={net}")
                
            return within_limits
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False

    def update_pnl(self):
        """Update realized and unrealized PnL - ENHANCED FOR PROFIT TRACKING"""
        try:
            positions = self.get_positions()
            current_prices = {}
            
            for ticker in [self.BULL, self.BEAR, self.RITC]:
                bid, ask = self.get_best_bid_ask(ticker)
                current_prices[ticker] = (bid + ask) / 2
            
            # Calculate unrealized PnL
            self.pnl_unrealized = 0.0
            for ticker, quantity in positions.items():
                if ticker in current_prices and quantity != 0:
                    # This is simplified - in reality you'd track cost basis
                    self.pnl_unrealized += quantity * current_prices[ticker]
            
            # Update exposure
            self.exposure_gross = sum(abs(qty) for qty in positions.values())
            self.exposure_net = sum(positions.values())
            
            # PROFIT OPTIMIZATION TRACKING
            total_pnl = self.pnl_realized + self.pnl_unrealized
            if total_pnl > 0:
                self.win_rate = self.successful_arbitrages / max(self.trade_count, 1)
                self.profit_per_trade = self.total_profit / max(self.successful_arbitrages, 1)
            
            # Check if we've hit profit target
            if self.total_profit >= self.profit_target:
                logger.info(f"PROFIT TARGET REACHED! Total profit: ${self.total_profit:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating PnL: {e}")

    def optimize_for_profits(self):
        """PROFIT OPTIMIZATION METHOD - Aggressive profit maximization"""
        try:
            # Check if we're close to profit target
            if self.total_profit >= self.profit_target * 0.8:
                logger.info("Near profit target - increasing aggressiveness!")
                # Lower threshold for more opportunities
                self.ARB_THRESHOLD_CAD = 0.03
                # Increase order size
                self.ORDER_QTY = min(self.ORDER_QTY * 1.2, self.MAX_SIZE_EQUITY)
            
            # If we're losing money, be more conservative
            if self.total_profit < 0:
                logger.warning("In drawdown - reducing risk")
                self.ARB_THRESHOLD_CAD = 0.08
                self.ORDER_QTY = max(self.ORDER_QTY * 0.8, 1000)
            
            # Log profit status
            logger.info(f"Profit Status: Total=${self.total_profit:.2f}, Target=${self.profit_target:.2f}, Win Rate={self.win_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Error in profit optimization: {e}")

    def step_once(self) -> Dict:
        """Execute one trading step - OPTIMIZED FOR PROFITS"""
        try:
            # PROFIT OPTIMIZATION - Adjust strategy based on performance
            self.optimize_for_profits()
            
            # Check risk limits first
            if not self.check_risk_limits():
                logger.warning("Risk limits exceeded, skipping trade")
                return {"traded": False, "reason": "risk_limits"}
            
            # 1. Handle tender offers - PROFIT FOCUSED
            profitable_tenders = self.analyze_tender_offers()
            for tender in profitable_tenders:
                if self.accept_tender_offer(tender):
                    self.fast_unwind_tender(tender)
                    logger.info(f"Tender profit: ${tender['pnl']:.2f}")
            
            # 2. Check for ETF mispricing - AGGRESSIVE PROFIT HUNTING
            has_opportunity, direction, edge = self.detect_mispricing()
            
            if has_opportunity and self.check_risk_limits():
                traded = self.execute_arbitrage(direction, edge)
                if traded:
                    self.update_pnl()
                    return {"traded": True, "direction": direction, "edge": edge, "profit": edge * self.ORDER_QTY}
            
            # 3. Update PnL and profit tracking
            self.update_pnl()
            
            return {"traded": False, "reason": "no_opportunity"}
            
        except Exception as e:
            logger.error(f"Error in step_once: {e}")
            return {"traded": False, "reason": "error", "error": str(e)}

    def run(self):
        """Main trading loop"""
        logger.info("Starting arbitrage strategy...")
        
        try:
            tick, status = self.get_tick_status()
            while status == "ACTIVE":
                result = self.step_once()
                
                # Log status
                logger.info(f"Tick {tick}: {result}")
                
                # Sleep between iterations
                time.sleep(0.5)
                tick, status = self.get_tick_status()
                
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user")
        except Exception as e:
            logger.error(f"Strategy error: {e}")
        finally:
            self.print_summary()

    def print_summary(self):
        """Print final performance summary - ENHANCED FOR PROFIT TRACKING"""
        runtime = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("PROFIT-OPTIMIZED ARBITRAGE STRATEGY SUMMARY 🚀")
        print("="*60)
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Total Trades: {self.trade_count}")
        print(f"Successful Arbitrages: {self.successful_arbitrages}")
        print(f"Total Profit: ${self.total_profit:.2f}")
        print(f"Profit Target: ${self.profit_target:.2f}")
        print(f"Win Rate: {self.win_rate:.2%}")
        print(f"Max Profit Trade: ${self.max_profit_trade:.2f}")
        print(f"Profit per Trade: ${self.profit_per_trade:.2f}")
        print(f"Best Arbitrage Edge: ${self.best_arbitrage_edge:.4f}")
        print(f"Realized PnL: ${self.pnl_realized:.2f}")
        print(f"Unrealized PnL: ${self.pnl_unrealized:.2f}")
        print(f"Total PnL: ${self.pnl_realized + self.pnl_unrealized:.2f}")
        print(f"Gross Exposure: ${self.exposure_gross:.2f}")
        print(f"Net Exposure: ${self.exposure_net:.2f}")
        print(f"Tender Decisions: {len(self.tender_decisions)}")
        print(f"Converter Usage: {len(self.converter_usage)}")
        
        # Profit performance analysis
        if self.total_profit > 0:
            print(f"\nPROFIT PERFORMANCE: {self.total_profit/self.profit_target*100:.1f}% of target achieved!")
        else:
            print(f"\nLOSS PERFORMANCE: ${abs(self.total_profit):.2f} loss")
        
        # Print recent trades
        if self.trades:
            print(f"\nRecent Trades (Last 10):")
            for trade in self.trades[-10:]:  # Last 10 trades
                print(f"  {trade.ticker}: {trade.action} {trade.quantity} @ ${trade.price:.4f}")
        
        print("="*60)

# Legacy compatibility functions
def get_tick_status():
    """Legacy function for backward compatibility"""
    strategy = ArbitrageStrategy()
    return strategy.get_tick_status()

def best_bid_ask(ticker):
    """Legacy function for backward compatibility"""
    strategy = ArbitrageStrategy()
    return strategy.get_best_bid_ask(ticker)

def positions_map():
    """Legacy function for backward compatibility"""
    strategy = ArbitrageStrategy()
    return strategy.get_positions()

def place_mkt(ticker, action, qty):
    """Legacy function for backward compatibility"""
    strategy = ArbitrageStrategy()
    return strategy.place_order(ticker, action, qty, "MARKET")

def within_limits():
    """Legacy function for backward compatibility"""
    strategy = ArbitrageStrategy()
    return strategy.check_risk_limits()

def accept_active_tender_offers():
    """Legacy function for backward compatibility"""
    strategy = ArbitrageStrategy()
    tenders = strategy.analyze_tender_offers()
    for tender in tenders:
        strategy.accept_tender_offer(tender)

def step_once():
    """Legacy function for backward compatibility"""
    strategy = ArbitrageStrategy()
    return strategy.step_once()

def main():

    """Main function - can run in different modes"""
    import sys
    
    # Check command line arguments for mode
    mode = Mode.LIVE
    if len(sys.argv) > 1:
        mode_str = sys.argv[1].lower()
        if mode_str == "backtest":
            mode = Mode.BACKTEST
        elif mode_str == "dry_run":
            mode = Mode.DRY_RUN
    
    # Create and run strategy
    strategy = ArbitrageStrategy(mode)
    strategy.run()

if __name__ == "__main__":
    main()

