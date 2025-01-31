import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from data.mt5_connector import MT5Connector
from backtesting.engine import BacktestEngine
from strategy.ma_cross import MACrossStrategy
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format
    handlers=[
        logging.StreamHandler()
    ]
)

# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)  # Also suppress Pillow debug messages

logger = logging.getLogger(__name__)

def calculate_position_size(price, stop_loss, risk_amount, min_lot=0.01, max_lot=0.5):
    """
    Calculate position size based on:
    - Fixed risk amount (e.g. $150)
    - Dynamic stop loss distance
    - Min/max lot size constraints
    """
    stop_distance = abs(price - stop_loss)
    stop_distance_pips = stop_distance / 0.1  # Convert price distance to pips
    
    # For gold: 10 pips = $1 for 0.01 lot
    # So if we want to risk $150 over 50 pips:
    # $150 = (50 pips / 10) * lot_size * 100
    # Therefore: lot_size = $150 / ((50/10) * 100) = 0.3
    lot_size = round(risk_amount / ((stop_distance_pips/10) * 100), 2)
    
    # Constrain to min/max lot sizes
    lot_size = max(min_lot, min(lot_size, max_lot))
    return lot_size

def print_trade_details(trades_history):
    """Print detailed information about each trade"""
    print("\nTrade Details:")
    print("-" * 110)
    print(f"{'Entry Time':<20} {'Type':<6} {'Size':<6} {'Entry':<9} {'Exit':<9} {'SL':<9} {'TP':<9} {'P/L':<8} {'E->SL Pips':<9} {'E->TP Pips':<9}")
    print("-" * 110)
    
    for trade in trades_history:
        # Calculate pip distances
        entry_to_sl_pips = abs(trade['stop_loss'] - trade['entry_price']) * 10
        entry_to_tp_pips = abs(trade['take_profit'] - trade['entry_price']) * 10
        
        print(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{trade['type']:<6} "
              f"{trade['size']:<6.2f} "
              f"{trade['entry_price']:<9.2f} "
              f"{trade['exit_price']:<9.2f} "
              f"{trade['stop_loss']:<9.2f} "
              f"{trade['take_profit']:<9.2f} "
              f"${trade['pnl']:<7.2f} "
              f"{entry_to_sl_pips:<9.1f} "
              f"{entry_to_tp_pips:<9.1f}")
    print("-" * 110)

def test_strategy_all_timeframes():
    """Test strategy on multiple timeframes"""
    timeframes = {
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1
    }
    
    results = {}
    best_results = {}
    
    for tf_name, tf in timeframes.items():
        print(f"\n{'='*20} Testing {tf_name} Timeframe {'='*20}")
        result = test_xauusd_strategy(
            fast_ma_range=(5, 20),
            slow_ma_range=(20, 50),
            timeframe=tf
        )
        
        if result:
            engine, metrics, params = result
            best_results[tf_name] = {
                "metrics": metrics,
                "params": params
            }
            # Comment out trade details
            # print_trade_details(engine.trades_history)
    
    # Print best results for each timeframe
    print("\nBest Results by Timeframe:")
    print("=" * 80)
    for tf_name, result in best_results.items():
        metrics = result["metrics"]
        params = result["params"]
        print(f"\n{tf_name} Timeframe:")
        print("-" * 40)
        print(f"Fast MA Period: {params[0]}")
        print(f"Slow MA Period: {params[1]}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print("=" * 80)

def test_xauusd_strategy(fast_ma_range=(5, 20), slow_ma_range=(20, 50), timeframe=mt5.TIMEFRAME_H1):
    """
    Test XAUUSD strategy with different MA parameters
    """
    try:
        # Initialize MT5 connection
        connector = MT5Connector()
        
        # Set test parameters
        symbol = "XAUUSDc"  # Updated symbol name for HFM
        initial_balance = 2000  # Starting with $2000
        
        # Gold-specific parameters
        # Standard lot (1.0) = 100 ounces, pip value = $10
        # Mini lot (0.1) = 10 ounces, pip value = $1
        # Micro lot (0.01) = 1 ounce, pip value = $0.10
        min_lot = 0.20  # Starting with 20 ounces
        max_lot = 0.70  # Maximum 70 ounces
        
        # Risk parameters
        risk_per_trade = 200  # Fixed $200 risk per trade
        
        # Dynamic stop loss parameters
        min_stop_pips = 10  # Minimum 10 pips stop loss
        max_stop_pips = 80  # Maximum 80 pips stop loss
        take_profit_ratio = 2.5  # Higher reward:risk ratio
        
        # Set date range (last 9 months for comprehensive testing)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=270)  # Changed from 365 to 270 days
        
        print("\nBacktest Parameters:")
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Initial Balance: ${initial_balance}")
        print(f"Leverage: 500:1")
        print(f"Position Size: {min_lot:.2f} to {max_lot:.2f} lots (dynamic based on stop distance)")
        print(f"Stop Loss: {min_stop_pips} to {max_stop_pips} pips (dynamic)")
        print(f"Take Profit: {take_profit_ratio}x stop loss distance")
        print(f"Pip Value (per 10 pips): ${min_lot * 10:.2f} to ${max_lot * 10:.2f}")  # 10 pips = $1 for 0.01 lot
        print(f"Leverage: 500:1\n")
        
        # Get historical data
        data = connector.get_historical_data(symbol, timeframe, start_date, end_date)
        if data is None:
            print("Error: Failed to get historical data. Please check your MT5 connection.")
            return
            
        print(f"Testing MA combinations...")
        print("-" * 50)
        
        best_result = None
        best_metrics = None
        best_params = None
        
        # Test different MA combinations
        total_combinations = (fast_ma_range[1] - fast_ma_range[0]) * (slow_ma_range[1] - slow_ma_range[0])
        current_combination = 0
        
        for fast_period in range(fast_ma_range[0], fast_ma_range[1]):
            for slow_period in range(slow_ma_range[0], slow_ma_range[1]):
                if fast_period >= slow_period:
                    continue
                    
                current_combination += 1
                    
                strategy = MACrossStrategy(
                    fast_period=fast_period,
                    slow_period=slow_period,
                    min_stop_pips=min_stop_pips,
                    max_stop_pips=max_stop_pips
                )
                
                engine = BacktestEngine(initial_balance=initial_balance)
                
                # Generate signals and run backtest
                test_data = strategy.generate_signals(data.copy())
                
                for i in range(len(test_data)):
                    current_bar = test_data.iloc[i]
                    timestamp = current_bar['time']
                    price = current_bar['close']
                    
                    engine.update_positions(timestamp, price)
                    
                    if current_bar['buy_signal']:
                        # Calculate dynamic stop loss based on volatility
                        atr = current_bar.get('atr', (max_stop_pips * 0.1))  # Default to max if no ATR
                        stop_loss_pips = max(min_stop_pips, min(max_stop_pips, int(atr * 1.2)))  # Reduced from 1.5 to 1.2
                        stop_loss = price - (stop_loss_pips * 0.1)
                        take_profit = price + (stop_loss_pips * take_profit_ratio * 0.1)
                        
                        # Calculate position size based on stop distance
                        position_size = calculate_position_size(
                            price=price,
                            stop_loss=stop_loss,
                            risk_amount=risk_per_trade,
                            min_lot=min_lot,
                            max_lot=max_lot
                        )
                        
                        engine.open_position(timestamp, symbol, 'buy', price, stop_loss, take_profit, position_size)
                        
                    elif current_bar['sell_signal']:
                        # Calculate dynamic stop loss based on volatility
                        atr = current_bar.get('atr', (max_stop_pips * 0.1))  # Default to max if no ATR
                        stop_loss_pips = max(min_stop_pips, min(max_stop_pips, int(atr * 1.2)))  # Reduced from 1.5 to 1.2
                        stop_loss = price + (stop_loss_pips * 0.1)
                        take_profit = price - (stop_loss_pips * take_profit_ratio * 0.1)
                        
                        # Calculate position size based on stop distance
                        position_size = calculate_position_size(
                            price=price,
                            stop_loss=stop_loss,
                            risk_amount=risk_per_trade,
                            min_lot=min_lot,
                            max_lot=max_lot
                        )
                        
                        engine.open_position(timestamp, symbol, 'sell', price, stop_loss, take_profit, position_size)
                
                metrics = engine.get_performance_metrics()
                
                # Update best result if this combination is better
                if best_metrics is None or metrics['total_profit'] > best_metrics['total_profit']:
                    best_metrics = metrics
                    best_params = (fast_period, slow_period)
                    best_result = (engine, test_data)
                    
                print(f"MA({fast_period:2d}, {slow_period:2d}) - Profit: ${metrics['total_profit']:8.2f}, "
                      f"Win Rate: {metrics['win_rate']:.2%}")
        
        if best_params is None:
            print("\nError: No valid results found")
            return
            
        # Print best results
        print("\n" + "=" * 50)
        print("Best Strategy Found:")
        print("=" * 50)
        print(f"Fast MA Period: {best_params[0]}")
        print(f"Slow MA Period: {best_params[1]}")
        print("\nPerformance Metrics:")
        print(f"Total Trades: {best_metrics['total_trades']}")
        print(f"Win Rate: {best_metrics['win_rate']:.2%}")
        print(f"Total Profit: ${best_metrics['total_profit']:.2f}")
        print(f"Max Drawdown: {best_metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        print(f"Profit Factor: {best_metrics['profit_factor']:.2f}")
        print("=" * 50)
        
        # Plot results
        engine, test_data = best_result
        plt.figure(figsize=(15, 10))
        
        # Plot price and MAs
        plt.subplot(2, 1, 1)
        plt.plot(test_data['time'], test_data['close'], label=f'{symbol} Price')
        plt.plot(test_data['time'], test_data['fast_ma'], label=f'Fast MA({best_params[0]})')
        plt.plot(test_data['time'], test_data['slow_ma'], label=f'Slow MA({best_params[1]})')
        plt.title(f'{symbol} Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot equity curve
        plt.subplot(2, 1, 2)
        timestamps = [point['timestamp'] for point in engine.equity_curve]
        equity_values = [point['equity'] for point in engine.equity_curve]
        plt.plot(timestamps, equity_values)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Account Balance ($)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return engine, best_metrics, best_params
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

def test_best_ma_strategy(timeframe=mt5.TIMEFRAME_H1):
    """
    Test XAUUSD strategy with the best performing MA combination (19,20)
    """
    try:
        # Initialize MT5 connection
        connector = MT5Connector()
        
        # Set test parameters
        symbol = "XAUUSDc"  # Updated symbol name for HFM
        initial_balance = 2000  # Starting with $2000
        
        # Gold-specific parameters
        min_lot = 0.20  # Starting with 20 ounces
        max_lot = 0.70  # Maximum 70 ounces
        
        # Risk parameters
        risk_per_trade = 200  # Fixed $200 risk per trade
        
        # Dynamic stop loss parameters
        min_stop_pips = 10  # Minimum 10 pips stop loss
        max_stop_pips = 80  # Maximum 80 pips stop loss
        take_profit_ratio = 2.5  # Higher reward:risk ratio
        
        # Set date range (last 9 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=270)  # Changed from 365 to 270 days
        
        print("\nBacktest Parameters:")
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Initial Balance: ${initial_balance}")
        print(f"Leverage: 500:1")
        print(f"Position Size: {min_lot:.2f} to {max_lot:.2f} lots (dynamic based on stop distance)")
        print(f"Stop Loss: {min_stop_pips} to {max_stop_pips} pips (dynamic)")
        print(f"Take Profit: {take_profit_ratio}x stop loss distance")
        print(f"Pip Value (per 10 pips): ${min_lot * 10:.2f} to ${max_lot * 10:.2f}")
        print(f"Moving Averages: Fast(19), Slow(20)\n")
        
        # Get historical data
        data = connector.get_historical_data(symbol, timeframe, start_date, end_date)
        if data is None:
            print("Error: Failed to get historical data. Please check your MT5 connection.")
            return
            
        strategy = MACrossStrategy(
            fast_period=19,
            slow_period=20,
            min_stop_pips=min_stop_pips,
            max_stop_pips=max_stop_pips
        )
        
        engine = BacktestEngine(initial_balance=initial_balance)
        
        # Generate signals and run backtest
        test_data = strategy.generate_signals(data.copy())
        
        for i in range(len(test_data)):
            current_bar = test_data.iloc[i]
            timestamp = current_bar['time']
            price = current_bar['close']
            
            engine.update_positions(timestamp, price)
            
            if current_bar['buy_signal']:
                # Calculate dynamic stop loss based on volatility
                atr = current_bar.get('atr', (max_stop_pips * 0.1))
                stop_loss_pips = max(min_stop_pips, min(max_stop_pips, int(atr * 1.2)))
                stop_loss = price - (stop_loss_pips * 0.1)
                take_profit = price + (stop_loss_pips * take_profit_ratio * 0.1)
                
                position_size = calculate_position_size(
                    price=price,
                    stop_loss=stop_loss,
                    risk_amount=risk_per_trade,
                    min_lot=min_lot,
                    max_lot=max_lot
                )
                
                engine.open_position(timestamp, symbol, 'buy', price, stop_loss, take_profit, position_size)
                
            elif current_bar['sell_signal']:
                atr = current_bar.get('atr', (max_stop_pips * 0.1))
                stop_loss_pips = max(min_stop_pips, min(max_stop_pips, int(atr * 1.2)))
                stop_loss = price + (stop_loss_pips * 0.1)
                take_profit = price - (stop_loss_pips * take_profit_ratio * 0.1)
                
                position_size = calculate_position_size(
                    price=price,
                    stop_loss=stop_loss,
                    risk_amount=risk_per_trade,
                    min_lot=min_lot,
                    max_lot=max_lot
                )
                
                engine.open_position(timestamp, symbol, 'sell', price, stop_loss, take_profit, position_size)
        
        metrics = engine.get_performance_metrics()
        
        # Print results
        print("\nPerformance Metrics:")
        print("=" * 50)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print("=" * 50)
        
        # Comment out trade details
        # print_trade_details(engine.trades_history)
        
        return engine, metrics
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

if __name__ == "__main__":
    print("\nStarting XAUUSDc Strategy Backtest...")
    try:
        # Define timeframes to test
        timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4
        }
        
        # Colors for plotting
        colors = {
            "M5": "purple",
            "M15": "blue",
            "M30": "green",
            "H1": "red",
            "H4": "orange"
        }

        print("\nTesting strategy across timeframes...")
        print("-" * 50)
        
        # Store results for final summary
        all_results = {}
        
        # Create figure for equity curves
        plt.figure(figsize=(15, 6))
        
        for tf_name, tf in timeframes.items():
            print(f"\n{'='*20} Testing {tf_name} Timeframe {'='*20}")
            result = test_best_ma_strategy(timeframe=tf)
            
            if result:
                engine, metrics = result
                all_results[tf_name] = {
                    'engine': engine,
                    'metrics': metrics
                }
                
                # Plot equity curve for this timeframe
                timestamps = [point['timestamp'] for point in engine.equity_curve]
                equity_values = [point['equity'] for point in engine.equity_curve]
                plt.plot(timestamps, equity_values, label=f'{tf_name}', color=colors[tf_name])
        
        # Customize equity curves plot
        plt.title('Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Account Balance ($)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed analytics summary
        print("\n" + "="*50)
        print("DETAILED ANALYTICS SUMMARY")
        print("="*50)
        
        for tf_name, result in all_results.items():
            metrics = result['metrics']
            trades = result['engine'].trades_history
            equity_curve = result['engine'].equity_curve
            initial_balance = result['engine'].initial_balance  # Get initial balance from engine
            
            # Calculate additional metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = len([t for t in trades if t['pnl'] <= 0])
            avg_win = sum([t['pnl'] for t in trades if t['pnl'] > 0]) / winning_trades if winning_trades > 0 else 0
            avg_loss = sum([t['pnl'] for t in trades if t['pnl'] <= 0]) / losing_trades if losing_trades > 0 else 0
            
            # Calculate overall return
            final_equity = equity_curve[-1]['equity'] if equity_curve else initial_balance
            overall_return = ((final_equity - initial_balance) / initial_balance) * 100
            
            # Calculate weekly returns
            weekly_returns = []
            current_week = None
            week_start_equity = initial_balance
            
            for point in equity_curve:
                week = point['timestamp'].isocalendar()[1]  # Get week number
                if week != current_week:
                    if current_week is not None:
                        weekly_return = ((week_end_equity - week_start_equity) / week_start_equity) * 100
                        weekly_returns.append(weekly_return)
                    current_week = week
                    week_start_equity = point['equity']
                week_end_equity = point['equity']
            
            # Add last week
            if current_week is not None:
                weekly_return = ((week_end_equity - week_start_equity) / week_start_equity) * 100
                weekly_returns.append(weekly_return)
            
            # Calculate daily trade counts
            trade_dates = [t['timestamp'].date() for t in trades]
            unique_dates = len(set(trade_dates))
            avg_trades_per_day = total_trades / unique_dates if unique_dates > 0 else 0
            
            print(f"\n{tf_name} Timeframe Analysis:")
            print("-"*40)
            print(f"Total Trades: {total_trades}")
            print(f"Average Trades per Day: {avg_trades_per_day:.1f}")
            print(f"Winning Trades: {winning_trades} ({(winning_trades/total_trades*100):.2f}%)")
            print(f"Losing Trades: {losing_trades} ({(losing_trades/total_trades*100):.2f}%)")
            print(f"Average Win: ${avg_win:.2f}")
            print(f"Average Loss: ${avg_loss:.2f}")
            print(f"Total Net Profit: ${metrics['total_profit']:.2f}")
            print(f"Overall Return: {overall_return:.2f}%")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Max Drawdown Amount: ${metrics['max_drawdown_amount']:.2f}")
            print(f"Max Drawdown Duration: {metrics['max_drawdown_duration']} days")
            print(f"Average Drawdown: {metrics['avg_drawdown']:.2%}")
            print(f"Number of Drawdowns: {metrics['num_drawdowns']}")
            print(f"Recovery Time from Max DD: {metrics['recovery_time_days']} days")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            
            if weekly_returns:
                print(f"\nWeekly Returns:")
                print(f"Average: {sum(weekly_returns)/len(weekly_returns):.2f}%")
                print(f"Highest: {max(weekly_returns):.2f}%")
                print(f"Lowest: {min(weekly_returns):.2f}%")
            
            # Trade distribution
            print("\nTrade Distribution:")
            print(f"{'Profit Range':<15} {'Count':<10} {'Percentage':<10}")
            print("-"*35)
            
            # Create profit ranges
            profits = [t['pnl'] for t in trades]
            min_profit = min(profits)
            max_profit = max(profits)
            ranges = [
                (-float('inf'), -200),
                (-200, -100),
                (-100, -50),
                (-50, 0),
                (0, 50),
                (50, 100),
                (100, 200),
                (200, float('inf'))
            ]
            
            for r in ranges:
                count = len([p for p in profits if r[0] <= p < r[1]])
                if count > 0:
                    if r[0] == -float('inf'):
                        range_str = f"< ${r[1]}"
                    elif r[1] == float('inf'):
                        range_str = f"> ${r[0]}"
                    else:
                        range_str = f"${r[0]} to ${r[1]}"
                    print(f"{range_str:<15} {count:<10} {count/total_trades*100:.1f}%")
            
            print("-"*40)
            
    except Exception as e:
        print(f"\nError: {str(e)}") 