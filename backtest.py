import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from config.trading_config import TradingConfig
from strategies.ma_cross import MACrossStrategy
from connectors.mt5_connector import MT5Connector
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Configure matplotlib to only show warnings and above
mpl.set_loglevel('WARNING')

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trades_history = []
        self.equity_curve = []
        self.BE_BUFFER = 0.3  # 3 pips buffer for breakeven
        self.hit_breakeven_positions = set()  # Track positions that hit breakeven
        self.position_counter = 1  # Add position counter for unique ticket numbers
        
        # Record initial equity point
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'balance': self.balance,
            'equity': self.balance
        })
        
    def calculate_position_size(self, price, stop_loss, risk_amount):
        """Calculate position size based on risk amount and stop loss distance"""
        stop_distance = abs(price - stop_loss)
        stop_distance_pips = stop_distance * 10  # Convert to pips for XAUUSD
        
        # For XAUUSD: 1 pip = $1.00 for 0.01 lot
        pip_value = 1.0  # $1 per pip for 0.01 lot
        risk_pips = stop_distance_pips
        
        # Calculate lot size based on risk
        raw_lot_size = round((risk_amount / (risk_pips * pip_value * 100)), 2)
        
        # Constrain lot size between 0.20 and 0.70
        return max(0.20, min(raw_lot_size, 0.70))
        
    def open_position(self, timestamp, trade_type, entry_price, stop_loss, take_profit, size):
        """Open a new position"""
        position = {
            'ticket': self.position_counter,  # Add unique ticket number
            'timestamp': timestamp,
            'type': trade_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': size,
            'original_size': size,  # Store original size for partial closes
            'pnl': 0,
            'status': 'open',
            'partial_close_pnl': 0  # Track PnL from partial closes
        }
        self.positions.append(position)
        self.position_counter += 1  # Increment counter for next position
        
    def close_position(self, position, timestamp, close_price, reason='signal'):
        """Close an existing position"""
        if position['type'] == 'BUY':
            pips = (close_price - position['entry_price']) * 10  # Convert to pips
        else:  # SELL
            pips = (position['entry_price'] - close_price) * 10  # Convert to pips
        
        # Calculate PnL: pips * lot_size * pip_value
        # For XAUUSD: 1 pip = $1.00 for 0.01 lot, so multiply by 100
        position['pnl'] = pips * position['size'] * 10  # $1 per pip per 0.01 lot
        position['exit_price'] = close_price
        position['exit_time'] = timestamp
        position['exit_reason'] = reason
        position['status'] = 'closed'
        position['total_pnl'] = position['pnl'] + position['partial_close_pnl']
        
        self.balance += position['pnl']
        self.trades_history.append(position)
        self.positions.remove(position)
        
    def close_partial_position(self, position, timestamp, close_price, close_size):
        """Close part of a position"""
        if position['type'] == 'BUY':
            pips = (close_price - position['entry_price']) * 10
        else:  # SELL
            pips = (position['entry_price'] - close_price) * 10
        
        # Calculate PnL for closed portion
        partial_pnl = pips * close_size * 10  # $1 per pip per 0.01 lot
        
        # Update position
        position['size'] -= close_size
        position['partial_close_pnl'] += partial_pnl
        self.balance += partial_pnl
        
        return partial_pnl
        
    def update_positions(self, timestamp, current_price):
        """Update open positions with breakeven logic"""
        for position in self.positions[:]:  # Create a copy of the list to avoid modification during iteration
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if position['type'] == 'BUY':
                # Check for SL hit
                if current_price <= stop_loss:
                    self.close_position(position, timestamp, stop_loss, 'stop_loss')
                    continue
                # Check for TP hit
                elif current_price >= take_profit:
                    self.close_position(position, timestamp, take_profit, 'take_profit')
                    continue
                    
                # Check for breakeven opportunity
                target_distance = entry_price - stop_loss
                if position['ticket'] not in self.hit_breakeven_positions and current_price >= entry_price + target_distance:
                    # Close half position at current price
                    half_size = round(position['size'] / 2, 2)
                    partial_pnl = self.close_partial_position(position, timestamp, current_price, half_size)
                    
                    # Move SL to breakeven with buffer
                    position['stop_loss'] = entry_price + self.BE_BUFFER
                    self.hit_breakeven_positions.add(position['ticket'])
                    
            else:  # SELL position
                # Check for SL hit
                if current_price >= stop_loss:
                    self.close_position(position, timestamp, stop_loss, 'stop_loss')
                    continue
                # Check for TP hit
                elif current_price <= take_profit:
                    self.close_position(position, timestamp, take_profit, 'take_profit')
                    continue
                    
                # Check for breakeven opportunity
                target_distance = stop_loss - entry_price
                if position['ticket'] not in self.hit_breakeven_positions and current_price <= entry_price - target_distance:
                    # Close half position at current price
                    half_size = round(position['size'] / 2, 2)
                    partial_pnl = self.close_partial_position(position, timestamp, current_price, half_size)
                    
                    # Move SL to breakeven with buffer
                    position['stop_loss'] = entry_price - self.BE_BUFFER
                    self.hit_breakeven_positions.add(position['ticket'])
            
        # Update equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.balance,
            'equity': self.calculate_equity(current_price)
        })
                    
    def calculate_equity(self, current_price):
        """Calculate current equity including unrealized PnL"""
        equity = self.balance
        
        for position in self.positions:
            if position['type'] == 'BUY':
                unrealized_pips = (current_price - position['entry_price']) * 10
            else:  # SELL
                unrealized_pips = (position['entry_price'] - current_price) * 10
                
            # Calculate unrealized PnL
            unrealized_pnl = unrealized_pips * position['size'] * 10  # $1 per pip per 0.01 lot
            equity += unrealized_pnl
            
        return equity

def run_backtest(start_date=None, end_date=None, initial_balance=10000):
    """Run backtest using live trading configuration"""
    
    config = TradingConfig()
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)  # Default to last 30 days
    if end_date is None:
        end_date = datetime.now()
        
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    logger.info(f"Strategy Parameters: Fast MA={config.FAST_PERIOD}, Slow MA={config.SLOW_PERIOD}")
    logger.info(f"Risk Parameters: Risk/Trade=${config.RISK_PER_TRADE}, "
               f"Stop Loss Range={config.MIN_STOP_PIPS}-{config.MAX_STOP_PIPS} pips")
    
    # Initialize components
    connector = MT5Connector()
    strategy = MACrossStrategy(
        fast_period=config.FAST_PERIOD,
        slow_period=config.SLOW_PERIOD,
        min_stop_pips=config.MIN_STOP_PIPS,
        max_stop_pips=config.MAX_STOP_PIPS
    )
    engine = BacktestEngine(initial_balance=initial_balance)
    
    # Get historical data
    data = connector.get_historical_data(config.SYMBOL, config.TIMEFRAME, start_date, end_date)
    if data is None or len(data) == 0:
        logger.error("Failed to get historical data")
        return None
        
    logger.info(f"Loaded {len(data)} candles of historical data")
    
    # Generate signals
    signals = strategy.generate_signals(data.copy())
    last_signal = None
    
    # Ensure index is datetime
    if not isinstance(signals.index, pd.DatetimeIndex):
        signals.index = pd.to_datetime(signals.index)
    
    # Run backtest
    for i in range(len(signals)):
        current_bar = signals.iloc[i]
        timestamp = signals.index[i]  # Get timestamp from index
        
        # Update existing positions first
        engine.update_positions(timestamp, current_bar['close'])
        
        # Get spread from data or use typical spread
        spread = current_bar.get('spread', 0.30)  # Default 3 pip spread for XAUUSD
        
        # Check for new signals
        if current_bar['buy_signal'] == 1 and last_signal != 'buy':
            last_signal = 'buy'
            entry_price = current_bar['close'] + spread/2  # Simulate ask price
            
            # Get the most recent swing low
            swing_lows = signals['swing_low'].iloc[:i+1].dropna()
            
            if not swing_lows.empty:
                # Get the most recent swing low
                nearest_swing_low = swing_lows.iloc[-1]
                stop_loss = nearest_swing_low - (spread * 2)  # Add spread buffer
                
                # Check if SL is too close (less than 10 pips)
                sl_distance = entry_price - stop_loss
                if sl_distance < 1.0:  # 10 pips = 1.0 for gold
                    stop_loss = entry_price - 1.0 - (spread * 2)
            else:
                # Default to 10 pips SL if no swing low found
                stop_loss = entry_price - 1.0 - (spread * 2)
            
            # Calculate take profit (2.5:1 RR)
            take_profit = entry_price + ((entry_price - stop_loss) * 2.5) + (spread * 2)
            
            # Calculate position size
            position_size = engine.calculate_position_size(
                price=entry_price,
                stop_loss=stop_loss,
                risk_amount=config.RISK_PER_TRADE
            )
            
            # Open position
            engine.open_position(timestamp, 'BUY', entry_price, stop_loss, take_profit, position_size)
            
        elif current_bar['sell_signal'] == 1 and last_signal != 'sell':
            last_signal = 'sell'
            entry_price = current_bar['close'] - spread/2  # Simulate bid price
            
            # Get the most recent swing high
            swing_highs = signals['swing_high'].iloc[:i+1].dropna()
            
            if not swing_highs.empty:
                # Get the most recent swing high
                nearest_swing_high = swing_highs.iloc[-1]
                stop_loss = nearest_swing_high + (spread * 2)  # Add spread buffer
                
                # Check if SL is too close (less than 10 pips)
                sl_distance = stop_loss - entry_price
                if sl_distance < 1.0:  # 10 pips = 1.0 for gold
                    stop_loss = entry_price + 1.0 + (spread * 2)
            else:
                # Default to 10 pips SL if no swing high found
                stop_loss = entry_price + 1.0 + (spread * 2)
            
            # Calculate take profit (2.5:1 RR)
            take_profit = entry_price - ((stop_loss - entry_price) * 2.5) - (spread * 2)
            
            # Calculate position size
            position_size = engine.calculate_position_size(
                price=entry_price,
                stop_loss=stop_loss,
                risk_amount=config.RISK_PER_TRADE
            )
            
            # Open position
            engine.open_position(timestamp, 'SELL', entry_price, stop_loss, take_profit, position_size)
            
        # Reset last_signal if we have a crossover in the opposite direction
        elif (current_bar['buy_signal'] == 1 and last_signal == 'sell') or \
             (current_bar['sell_signal'] == 1 and last_signal == 'buy'):
            last_signal = None
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(engine)
    plot_equity_curve(engine)
    print_trade_summary(engine)
    
    return engine, metrics

def calculate_performance_metrics(engine):
    """Calculate trading performance metrics"""
    if not engine.trades_history:
        return None
        
    total_trades = len(engine.trades_history)
    winning_trades = len([t for t in engine.trades_history if t['total_pnl'] > 0])
    losing_trades = len([t for t in engine.trades_history if t['total_pnl'] <= 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    net_profit = sum(trade['total_pnl'] for trade in engine.trades_history)
    
    # Calculate max drawdown
    equity_curve = pd.DataFrame(engine.equity_curve)
    equity_curve['peak'] = equity_curve['equity'].expanding().max()
    equity_curve['drawdown'] = (equity_curve['peak'] - equity_curve['equity']) / equity_curve['peak'] * 100
    max_drawdown = equity_curve['drawdown'].max()
    
    # Calculate profit factor
    gross_profit = sum(trade['total_pnl'] for trade in engine.trades_history if trade['total_pnl'] > 0)
    gross_loss = abs(sum(trade['total_pnl'] for trade in engine.trades_history if trade['total_pnl'] <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate average trade metrics
    avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
    
    roi = (net_profit / engine.initial_balance * 100) if engine.initial_balance > 0 else 0
    
    print("\nPerformance Metrics:")
    print("-" * 50)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Net Profit: ${net_profit:.2f}")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Return on Initial Balance: {roi:.2f}%")
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'net_profit': net_profit,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'roi': roi
    }

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown percentage"""
    equity = [e['equity'] for e in equity_curve]
    peak = equity[0]
    max_dd = 0
    
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return max_dd

def calculate_profit_factor(trades):
    """Calculate profit factor"""
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    return f"{(gross_profit / gross_loss if gross_loss != 0 else float('inf')):.2f}"

def plot_equity_curve(engine):
    """Plot equity curve"""
    timestamps = [e['timestamp'] for e in engine.equity_curve]
    equity = [e['equity'] for e in engine.equity_curve]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, equity, label='Equity')
    plt.title('Backtest Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    Path('logs').mkdir(exist_ok=True)
    plt.savefig(f'logs/equity_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def print_trade_summary(engine):
    """Print summary of all trades"""
    print("\nTrade Summary:")
    print("-" * 120)
    print(f"{'Entry Time':<20} {'Type':<6} {'Size':<7} {'Entry':<9} {'Exit':<9} {'SL':<9} {'TP':<9} {'P/L':<12} {'Reason':<10}")
    print("-" * 120)
    
    total_pnl = 0
    for trade in engine.trades_history:
        total_pnl += trade['total_pnl']
        print(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{trade['type']:<6} "
              f"{trade['original_size']:<7.2f} "
              f"{trade['entry_price']:<9.2f} "
              f"{trade['exit_price']:<9.2f} "
              f"{trade['stop_loss']:<9.2f} "
              f"{trade['take_profit']:<9.2f} "
              f"${trade['total_pnl']:>9.2f} "
              f"{trade['exit_reason']:<10}")
    print("-" * 120)
    print(f"{'Total P/L:':<83} ${total_pnl:>9.2f}")
    print("-" * 120)

if __name__ == "__main__":
    # Run backtest for the last 30 days
    engine, metrics = run_backtest()
    
    if metrics:
        print("\nPerformance Metrics:")
        print("-" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value}")