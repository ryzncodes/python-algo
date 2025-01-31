import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class BacktestEngine:
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[Dict] = []
        self.trades_history: List[Dict] = []
        self.equity_curve: List[Dict] = []  # List of {timestamp, equity} dicts
        
    def reset(self):
        """Reset the backtesting engine"""
        self.balance = self.initial_balance
        self.positions = []
        self.trades_history = []
        self.equity_curve = []
        
    def calculate_position_size(self, price: float, risk_percent: float, stop_loss_pips: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Parameters:
        -----------
        price : float
            Current price
        risk_percent : float
            Risk percentage per trade (0-100)
        stop_loss_pips : float
            Stop loss in pips
            
        Returns:
        --------
        float
            Position size in units
        """
        risk_amount = self.balance * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_pips * price)
        return position_size
        
    def open_position(self, 
                     timestamp: pd.Timestamp,
                     symbol: str,
                     order_type: str,
                     entry_price: float,
                     stop_loss: float,
                     take_profit: float,
                     position_size: float) -> Dict:
        """
        Open a new trading position
        
        Parameters:
        -----------
        timestamp : pd.Timestamp
            Entry time
        symbol : str
            Trading symbol
        order_type : str
            'buy' or 'sell'
        entry_price : float
            Entry price
        stop_loss : float
            Stop loss price
        take_profit : float
            Take profit price
        position_size : float
            Position size in units
            
        Returns:
        --------
        Dict
            Position information
        """
        position = {
            'timestamp': timestamp,
            'symbol': symbol,
            'type': order_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': position_size,
            'pnl': 0.0,
            'status': 'open'
        }
        self.positions.append(position)
        return position
        
    def calculate_pnl(self, position, current_price):
        """Calculate P&L for a position"""
        price_diff = current_price - position['entry_price'] if position['type'] == 'buy' else position['entry_price'] - current_price
        pips = price_diff / 0.1  # Convert price difference to pips
        pnl = (pips * position['size'] * 10)  # Each pip is worth $1 for 0.01 lot
        return round(pnl, 2)

    def close_position(self, timestamp, position, current_price, reason=""):
        """Close a position and calculate final P&L"""
        pnl = self.calculate_pnl(position, current_price)
        position['exit_price'] = current_price
        position['pnl'] = pnl
        position['exit_time'] = timestamp
        position['exit_reason'] = reason
        
        # Update account balance
        self.balance += pnl
        
        # Add to trades history
        self.trades_history.append({
            'timestamp': timestamp,
            'type': position['type'],
            'size': position['size'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'pnl': pnl
        })
        
        position['status'] = 'closed'
        self.positions.remove(position)
        
    def update_positions(self, timestamp: pd.Timestamp, current_price: float):
        """
        Update open positions and check for stop loss/take profit hits
        """
        # Calculate current equity including unrealized P&L
        current_equity = self.balance
        for position in self.positions:
            unrealized_pnl = self.calculate_pnl(position, current_price)
            current_equity += unrealized_pnl
        
        # Update equity curve with current equity
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity
        })
        
        # Check for stop loss/take profit hits
        for position in self.positions[:]:  # Create a copy of the list to iterate
            if position['type'] == 'buy':
                if current_price <= position['stop_loss']:
                    # Close at stop loss price exactly
                    self.close_position(timestamp, position, position['stop_loss'], 'stop_loss')
                elif current_price >= position['take_profit']:
                    # Close at take profit price exactly
                    self.close_position(timestamp, position, position['take_profit'], 'take_profit')
            else:  # sell position
                if current_price >= position['stop_loss']:
                    # Close at stop loss price exactly
                    self.close_position(timestamp, position, position['stop_loss'], 'stop_loss')
                elif current_price <= position['take_profit']:
                    # Close at take profit price exactly
                    self.close_position(timestamp, position, position['take_profit'], 'take_profit')
                    
    def calculate_drawdown_metrics(self) -> Dict:
        """Calculate detailed drawdown metrics"""
        if not self.equity_curve:
            return {
                'max_drawdown': 0,
                'max_drawdown_duration': 0,
                'max_drawdown_amount': 0,
                'avg_drawdown': 0,
                'num_drawdowns': 0,
                'recovery_time_days': 0,
                'drawdown_periods': []
            }
            
        equity_values = [point['equity'] for point in self.equity_curve]
        timestamps = [point['timestamp'] for point in self.equity_curve]
        peak = equity_values[0]
        max_drawdown = 0
        max_drawdown_amount = 0
        current_drawdown_start = None
        drawdown_periods = []
        in_drawdown = False
        max_drawdown_start = None
        max_drawdown_end = None
        
        for i, (equity, timestamp) in enumerate(zip(equity_values, timestamps)):
            if equity > peak:
                # If we were in a drawdown, record it
                if in_drawdown:
                    duration = (timestamp - current_drawdown_start).days
                    drawdown_amount = peak - min(equity_values[drawdown_start_idx:i])
                    drawdown_pct = drawdown_amount / peak
                    drawdown_periods.append({
                        'start_date': current_drawdown_start,
                        'end_date': timestamp,
                        'duration_days': duration,
                        'drawdown_pct': drawdown_pct,
                        'drawdown_amount': drawdown_amount
                    })
                    in_drawdown = False
                peak = equity
            else:
                drawdown = (peak - equity) / peak
                drawdown_amount = peak - equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_amount = drawdown_amount
                    max_drawdown_start = current_drawdown_start if in_drawdown else timestamp
                    max_drawdown_end = timestamp
                
                if not in_drawdown:
                    in_drawdown = True
                    current_drawdown_start = timestamp
                    drawdown_start_idx = i
        
        # Calculate recovery time from max drawdown
        recovery_time = 0
        if max_drawdown_end:
            for timestamp in timestamps[timestamps.index(max_drawdown_end):]:
                if equity_values[timestamps.index(timestamp)] >= equity_values[timestamps.index(max_drawdown_start)]:
                    recovery_time = (timestamp - max_drawdown_end).days
                    break
        
        # Calculate average drawdown
        avg_drawdown = sum(period['drawdown_pct'] for period in drawdown_periods) / len(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': (max_drawdown_end - max_drawdown_start).days if max_drawdown_start and max_drawdown_end else 0,
            'max_drawdown_amount': max_drawdown_amount,
            'avg_drawdown': avg_drawdown,
            'num_drawdowns': len(drawdown_periods),
            'recovery_time_days': recovery_time,
            'drawdown_periods': drawdown_periods
        }
        
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics from trading history
        
        Returns:
        --------
        Dict
            Dictionary containing performance metrics
        """
        if not self.trades_history:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
            
        profits = [trade['pnl'] for trade in self.trades_history if trade['pnl'] > 0]
        losses = [trade['pnl'] for trade in self.trades_history if trade['pnl'] <= 0]
        
        total_trades = len(self.trades_history)
        profitable_trades = len(profits)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(trade['pnl'] for trade in self.trades_history)
        
        # Get detailed drawdown metrics
        drawdown_metrics = self.calculate_drawdown_metrics()
            
        # Calculate Sharpe Ratio (assuming risk-free rate = 0)
        equity_values = [point['equity'] for point in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Calculate Profit Factor
        total_profits = sum(profits) if profits else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_profits / total_losses if total_losses != 0 else float('inf')
        
        metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor
        }
        
        # Add detailed drawdown metrics
        metrics.update(drawdown_metrics)
        
        return metrics 