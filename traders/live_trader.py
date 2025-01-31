import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
import time
import logging
import os
import threading
from typing import Optional
from .base_trader import BaseTrader
from strategies.base_strategy import BaseStrategy
from connectors.base_connector import BaseConnector
from utils.position_sizing import calculate_position_size
import sys

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)

class LiveTrader(BaseTrader):
    def __init__(self,
                 strategy: BaseStrategy,
                 connector: BaseConnector,
                 symbol: str = "XAUUSD",
                 timeframe: int = mt5.TIMEFRAME_M1,
                 risk_per_trade: float = 500,
                 min_lot: float = 0.20,
                 max_lot: float = 0.70,
                 telegram_config: Optional[dict] = None,
                 test_mode: bool = False):
        """Initialize live trader"""
        
        super().__init__(
            strategy=strategy,
            connector=connector,
            symbol=symbol,
            timeframe=timeframe,
            risk_per_trade=risk_per_trade,
            min_lot=min_lot,
            max_lot=max_lot,
            telegram_config=telegram_config
        )
        
        # State variables
        self.last_signal = None
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=0)
        self.positions = []
        self.previous_positions = []
        self.trade_history = []
        self.daily_start_balance = None
        self.position_messages = {}  # Store message IDs for each position
        self.BE_BUFFER = 0.3  # 3 pips buffer for breakeven SL (0.3 points = 3 pips for XAUUSD)
        self.breakeven_messages = {}  # Store breakeven message info
        self.hit_breakeven_positions = set()  # Track positions that have hit breakeven
        self.test_mode = test_mode  # Store test mode flag
        
        # Configure logging with timestamp in filename
        self.setup_logging()
        
        # Place test trade if in test mode
        if test_mode:
            logger.info("Test mode enabled, placing test trade...")
            self._place_test_trade()
            # Wait a moment for the trade to be processed
            time.sleep(2)
            # Update positions list
            self.positions = self.connector.get_positions()
            logger.info(f"Updated positions after test trade. Current positions: {len(self.positions)}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_filename = f'logs/live_trading_{current_time}.log'
        
        file_handler = logging.FileHandler(self.log_filename, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    def get_latest_data(self, lookback_bars=200):
        """Get latest price data for signal generation"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        data = self.connector.get_historical_data(
            self.symbol,
            self.timeframe,
            start_date,
            end_date
        )
        
        if data is None or len(data) < self.strategy.slow_period + 1:
            logger.error(f"Insufficient data: Got {len(data) if data is not None else 0} bars, need at least {self.strategy.slow_period + 1}")
            return None
            
        return data
        
    def can_trade(self):
        """Check if enough time has passed since last trade"""
        if self.last_trade_time is None:
            return True
        return datetime.now() - self.last_trade_time >= self.min_trade_interval
        
    def check_for_signals(self):
        """Check for new trading signals"""
        # Get latest data
        data = self.get_latest_data()
        if data is None:
            logger.error("Failed to get latest data")
            sys.exit(1)  # Exit if we can't get data
        
        # When a position is closed, remove it from hit_breakeven_positions
        for pos in self.positions[:]:
            if pos['ticket'] not in [p['ticket'] for p in self.connector.get_positions()]:
                self.hit_breakeven_positions.discard(pos['ticket'])
        
        # Generate signals first (this calculates the MAs)
        data = self.strategy.generate_signals(data)
        
        # Log detailed MA information
        logger.info("\n=== Signal Check Details ===")
        logger.info(f"Last 5 Fast MA values:\n{data['fast_ma'].tail()}")
        logger.info(f"Last 5 Slow MA values:\n{data['slow_ma'].tail()}")
        
        # Get latest bar
        current_bar = data.iloc[-1]
        previous_bar = data.iloc[-2] if len(data) > 1 else None
        
        # Log detailed signal information
        logger.info("\nSignal Analysis:")
        logger.info(f"Current Bar - Buy Signal: {current_bar['buy_signal']}, Sell Signal: {current_bar['sell_signal']}")
        if previous_bar is not None:
            logger.info(f"Previous Bar - Buy Signal: {previous_bar['buy_signal']}, Sell Signal: {previous_bar['sell_signal']}")
        logger.info(f"Last Signal: {self.last_signal}")
        
        # Get latest bar and spread
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error("Failed to get symbol info")
            sys.exit(1)  # Exit if we can't get symbol info
            
        bid = symbol_info.bid
        ask = symbol_info.ask
        spread = ask - bid
        
        logger.info(f"\nPrice Information:")
        logger.info(f"Bid: {bid:.2f}")
        logger.info(f"Ask: {ask:.2f}")
        logger.info(f"Spread: {spread:.2f}")
        
        # Check if we can trade
        if not self.can_trade():
            logger.info(f"Trade not allowed - Last trade time: {self.last_trade_time}")
            return
            
        # Check for signals
        if current_bar['buy_signal'] and self.last_signal != 'buy':
            logger.info("\n=== BUY SIGNAL DETECTED ===")
            logger.info("Buy signal detected!")
            self.last_signal = 'buy'
            self.last_trade_time = datetime.now()
            
            # Get the most recent swing low before our signal
            current_idx = data.index[-1]
            swing_lows = data['swing_low'].loc[:current_idx].dropna()
            
            # For buy: Entry at ask
            entry_price = ask
            
            if not swing_lows.empty:
                # Get the most recent swing low
                nearest_swing_low = swing_lows.iloc[-1]
                stop_loss = nearest_swing_low - (spread * 2)  # Increased buffer below swing low
                
                # Check if SL is too close (less than 10 pips)
                sl_distance = entry_price - stop_loss
                if sl_distance < 1.0:  # 10 pips = 1.0 for gold
                    stop_loss = entry_price - 1.0 - (spread * 2)  # Add spread buffer to 10 pips SL
                    logger.info(f"Adjusted SL to minimum 10 pips + spread buffer: {stop_loss:.2f}")
                else:
                    logger.info(f"Using ICT Swing Low SL: {stop_loss:.2f} (Nearest Swing Low: {nearest_swing_low:.2f})")
            else:
                # Default to 10 pips SL if no swing low found
                stop_loss = entry_price - 1.0 - (spread * 2)
                logger.info(f"Using default 10 pips SL + spread buffer: {stop_loss:.2f}")
            
            # TP with spread adjustment
            take_profit = entry_price + ((entry_price - stop_loss) * 2.5) + (spread * 2)
            
            # Calculate position size based on SL distance and risk amount
            sl_distance_points = abs(entry_price - stop_loss)  # Distance in points
            raw_position_size = self.risk_per_trade / (sl_distance_points * 100)  # 100 is pip value for 1.0 lot XAUUSD
            position_size = self.validate_volume(raw_position_size)
            
            logger.info(f"Position size calculation:")
            logger.info(f"Risk amount: ${self.risk_per_trade}")
            logger.info(f"SL distance: {sl_distance_points*10:.1f} pips")
            logger.info(f"Raw position size: {raw_position_size:.3f} lots")
            logger.info(f"Adjusted position size: {position_size:.2f} lots")
            
            # Execute order first
            order_ticket = None
            if self.connector.execute_order(
                self.symbol,
                'buy',
                position_size,
                price=0.0,  # Market order
                stop_loss=stop_loss,
                take_profit=take_profit
            ):
                logger.info("Buy order executed successfully")
                # Get the order ticket from the result
                result = self.connector.last_order_result
                order_ticket = result.order if result else None
            else:
                logger.error("Failed to execute buy order")
            
            # Send notification after we have the order ticket
            if self.telegram:
                message_id = self.telegram.send_signal_notification(
                    signal_type="BUY",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size
                )
                if order_ticket:  # Only store if we have a valid ticket
                    self.position_messages[order_ticket] = message_id

        elif current_bar['sell_signal'] and self.last_signal != 'sell':
            logger.info("\n=== SELL SIGNAL DETECTED ===")
            logger.info("Sell signal detected!")
            self.last_signal = 'sell'
            self.last_trade_time = datetime.now()
            
            # Get the most recent swing high before our signal
            current_idx = data.index[-1]
            swing_highs = data['swing_high'].loc[:current_idx].dropna()
            
            # For sell: Entry at bid
            entry_price = bid
            
            if not swing_highs.empty:
                # Get the most recent swing high
                nearest_swing_high = swing_highs.iloc[-1]
                stop_loss = nearest_swing_high + (spread * 2)  # Increased buffer above swing high
                
                # Check if SL is too close (less than 10 pips)
                sl_distance = stop_loss - entry_price
                if sl_distance < 1.0:  # 10 pips = 1.0 for gold
                    stop_loss = entry_price + 1.0 + (spread * 2)  # Add spread buffer to 10 pips SL
                    logger.info(f"Adjusted SL to minimum 10 pips + spread buffer: {stop_loss:.2f}")
                else:
                    logger.info(f"Using ICT Swing High SL: {stop_loss:.2f} (Nearest Swing High: {nearest_swing_high:.2f})")
            else:
                # Default to 10 pips SL if no swing high found
                stop_loss = entry_price + 1.0 + (spread * 2)
                logger.info(f"Using default 10 pips SL + spread buffer: {stop_loss:.2f}")
            
            # TP with spread adjustment
            take_profit = entry_price - ((stop_loss - entry_price) * 2.5) - (spread * 2)
            
            # Calculate position size based on SL distance and risk amount
            sl_distance_points = abs(entry_price - stop_loss)  # Distance in points
            raw_position_size = self.risk_per_trade / (sl_distance_points * 100)  # 100 is pip value for 1.0 lot XAUUSD
            position_size = self.validate_volume(raw_position_size)
            
            logger.info(f"Position size calculation:")
            logger.info(f"Risk amount: ${self.risk_per_trade}")
            logger.info(f"SL distance: {sl_distance_points*10:.1f} pips")
            logger.info(f"Raw position size: {raw_position_size:.3f} lots")
            logger.info(f"Adjusted position size: {position_size:.2f} lots")
            
            # Execute order first
            order_ticket = None
            if self.connector.execute_order(
                self.symbol,
                'sell',
                position_size,
                price=0.0,  # Market order
                stop_loss=stop_loss,
                take_profit=take_profit
            ):
                logger.info("Sell order executed successfully")
                # Get the order ticket from the result
                result = self.connector.last_order_result
                order_ticket = result.order if result else None
            else:
                logger.error("Failed to execute sell order")
            
            # Send notification after we have the order ticket
            if self.telegram:
                message_id = self.telegram.send_signal_notification(
                    signal_type="SELL",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size
                )
                if order_ticket:  # Only store if we have a valid ticket
                    self.position_messages[order_ticket] = message_id

        else:
            logger.info("\nNo new signals detected")
            logger.info("=== End Signal Check ===\n")

    def update_positions(self):
        """Update open positions and check for closed positions"""
        current_positions = self.connector.get_positions()
        current_tickets = [p['ticket'] for p in current_positions]
        
        logger.debug(f"Current positions: {len(current_positions)}")
        logger.debug(f"Tracked positions: {len(self.positions)}")
        
        # Update position tracking
        for pos in self.positions[:]:  # Use slice to avoid modification during iteration
            logger.debug(f"Checking position {pos['ticket']}")
            
            # Position is still open
            if pos['ticket'] in current_tickets:
                current_pos = next(p for p in current_positions if p['ticket'] == pos['ticket'])
                entry_price = pos['price_open']
                current_price = current_pos['price_current']
                stop_loss = pos['sl']
                take_profit = pos['tp']
                
                # Skip breakeven check if position already hit breakeven
                if pos['ticket'] in self.hit_breakeven_positions:
                    continue
                
                # Calculate distances in pips (multiply by 10 for XAU/USD)
                sl_distance = abs(current_price - stop_loss) * 10
                tp_distance = abs(current_price - take_profit) * 10
                
                logger.debug(f"SL Distance: {sl_distance:.1f} pips, TP Distance: {tp_distance:.1f} pips")
                
                # Check for SL to breakeven opportunity
                if pos['type'] == 0:  # Buy position
                    target_distance = entry_price - stop_loss
                    logger.debug(f"Buy position - Target distance for BE: {target_distance*10:.1f} pips, Current distance: {(current_price - entry_price)*10:.1f} pips")
                    if current_price >= entry_price + target_distance:
                        logger.info(f"Position {pos['ticket']} reached 1:1 R:R, attempting to move to breakeven")
                        # Check if SL is already at breakeven
                        if abs(stop_loss - entry_price) > 0.01:  # Only modify if SL is not already at entry
                            original_volume = pos['volume']
                            half_volume = round(original_volume / 2, 2)  # Round to 2 decimal places
                            
                            # Close half position at market
                            if self.connector.close_partial_position(pos['ticket'], half_volume):
                                logger.info(f"Closed 50% ({half_volume} lots) of position {pos['ticket']} at breakeven")
                                # Add to hit breakeven set
                                self.hit_breakeven_positions.add(pos['ticket'])
                                # Calculate profit for closed portion
                                pip_value = (current_price - entry_price) * (half_volume * 100)  # 100 is pip value for 1.0 lot XAUUSD
                                partial_close_profit = round(pip_value, 2)
                                
                                # Move SL to entry for remaining position (with buffer)
                                breakeven_sl = entry_price - self.BE_BUFFER  # Set SL below entry
                                
                                if self.connector.modify_position(pos['ticket'], breakeven_sl, take_profit):
                                    logger.info(f"Moving SL to breakeven (with {self.BE_BUFFER*10:.1f} pips buffer) for remaining {half_volume} lots of ticket {pos['ticket']}")
                                    if self.telegram:
                                        # Use the original signal message ID for reply
                                        reply_to = self.position_messages.get(pos['ticket'])
                                        message_id = self.telegram.send_breakeven_notification(
                                            ticket=pos['ticket'],
                                            half_volume=half_volume,
                                            partial_close_profit=partial_close_profit,
                                            breakeven_sl=breakeven_sl,
                                            current_price=current_price,
                                            entry_price=entry_price,
                                            reply_to=reply_to
                                        )
                                        # Store breakeven message info
                                        self.breakeven_messages[pos['ticket']] = {
                                            'breakeven_msg_id': message_id,
                                            'original_msg_id': reply_to,
                                            'entry_price': entry_price,
                                            'partial_profit': partial_close_profit
                                        }
                            else:
                                logger.warning(f"Failed to close partial position for ticket {pos['ticket']}")
                        else:
                            logger.debug(f"SL already at breakeven for ticket {pos['ticket']}")
                elif pos['type'] == 1:  # Sell position
                    target_distance = stop_loss - entry_price
                    logger.debug(f"Sell position - Target distance for BE: {target_distance*10:.1f} pips, Current distance: {(entry_price - current_price)*10:.1f} pips")
                    if current_price <= entry_price - target_distance:
                        logger.info(f"Position {pos['ticket']} reached 1:1 R:R, attempting to move to breakeven")
                        # Check if SL is already at breakeven
                        if abs(stop_loss - entry_price) > 0.01:  # Only modify if SL is not already at entry
                            original_volume = pos['volume']
                            half_volume = round(original_volume / 2, 2)  # Round to 2 decimal places
                            
                            # Close half position at market
                            if self.connector.close_partial_position(pos['ticket'], half_volume):
                                logger.info(f"Closed 50% ({half_volume} lots) of position {pos['ticket']} at breakeven")
                                # Add to hit breakeven set
                                self.hit_breakeven_positions.add(pos['ticket'])
                                # Calculate profit for closed portion
                                pip_value = (entry_price - current_price) * (half_volume * 100)  # 100 is pip value for 1.0 lot XAUUSD
                                partial_close_profit = round(pip_value, 2)
                                
                                # Move SL to entry for remaining position (with buffer)
                                breakeven_sl = entry_price + self.BE_BUFFER  # Set SL above entry
                                
                                if self.connector.modify_position(pos['ticket'], breakeven_sl, take_profit):
                                    logger.info(f"Moving SL to breakeven (with {self.BE_BUFFER*10:.1f} pips buffer) for remaining {half_volume} lots of ticket {pos['ticket']}")
                                    if self.telegram:
                                        # Use the original signal message ID for reply
                                        reply_to = self.position_messages.get(pos['ticket'])
                                        message_id = self.telegram.send_breakeven_notification(
                                            ticket=pos['ticket'],
                                            half_volume=half_volume,
                                            partial_close_profit=partial_close_profit,
                                            breakeven_sl=breakeven_sl,
                                            current_price=current_price,
                                            entry_price=entry_price,
                                            reply_to=reply_to
                                        )
                                        # Store breakeven message info
                                        self.breakeven_messages[pos['ticket']] = {
                                            'breakeven_msg_id': message_id,
                                            'original_msg_id': reply_to,
                                            'entry_price': entry_price,
                                            'partial_profit': partial_close_profit
                                        }
                            else:
                                logger.warning(f"Failed to close partial position for ticket {pos['ticket']}")
                        else:
                            logger.debug(f"SL already at breakeven for ticket {pos['ticket']}")
            
            # Position closed - determine closure type
            else:
                logger.info(f"Position {pos['ticket']} no longer in open positions")
                
                if self.telegram:
                    # Get message IDs for reply
                    breakeven_info = self.breakeven_messages.get(pos['ticket'])
                    reply_to = None
                    additional_info = ""
                    
                    if breakeven_info:
                        # If position had hit breakeven, reply to that message
                        reply_to = breakeven_info['breakeven_msg_id']
                        additional_info = f"\n↩️ Related to: Breakeven at ${breakeven_info['entry_price']:.2f} (Partial Profit: ${breakeven_info['partial_profit']:.2f})"
                    else:
                        # If position didn't hit breakeven, reply to original message
                        reply_to = self.position_messages.get(pos['ticket'])
                    
                    # Calculate exit price based on last known price
                    exit_price = pos['price_current']
                    entry_price = pos['price_open']
                    
                    # Calculate pip distance between entry and exit
                    pip_distance = abs(exit_price - entry_price) * 10
                    
                    # Calculate profit with proper decimal places
                    if pos['type'] == 0:  # Buy
                        profit = round((exit_price - entry_price) * pos['volume'] * 100, 2)
                    else:  # Sell
                        profit = round((entry_price - exit_price) * pos['volume'] * 100, 2)

                    # Determine closure type based on pip distance and profit
                    BE_THRESHOLD = 3.0  # Consider within 3 pips as breakeven
                    if pip_distance <= BE_THRESHOLD:
                        closure_type = "BE"  # Breakeven
                        title = "Breakeven Exit"
                        emoji = "🔄"
                    else:
                        if profit > 0:
                            closure_type = "TP"
                            title = "Take Profit Hit"
                            emoji = "🎯"
                        else:
                            closure_type = "SL"
                            title = "Stop Loss Hit"
                            emoji = "🛑"

                    # Convert position time from timestamp to datetime
                    pos_time = datetime.fromtimestamp(pos['time']) if isinstance(pos['time'], (int, float)) else pos['time']
                    
                    # Construct message
                    message = (
                        f"{emoji} <b>{title}</b> {emoji}\n"
                        f"Type: {pos['type']}\n"
                        f"Entry: ${entry_price:.2f}\n"
                        f"Exit: ${exit_price:.2f}\n"
                        f"Distance: {pip_distance:.1f} pips\n"
                        f"💰 Profit: ${profit:.2f}\n"
                        f"⏱ Duration: {str(datetime.now() - pos_time)}"
                        f"{additional_info}"
                    )
                    
                    # Send notification
                    self.telegram.send_message(message, reply_to)
                
                self.positions.remove(pos)
                self.trade_history.append(pos)
                # Clean up message tracking
                if pos['ticket'] in self.position_messages:
                    del self.position_messages[pos['ticket']]
                if pos['ticket'] in self.breakeven_messages:
                    del self.breakeven_messages[pos['ticket']]
        
        # Update current positions list
        self.positions = current_positions

    def generate_daily_summary(self):
        """Generate daily performance statistics"""
        if not self.trade_history:
            return None
        
        total_profit = 0
        total_loss = 0
        winning_trades = 0
        max_drawdown = 0
        current_drawdown = 0
        
        for trade in self.trade_history:
            if trade['profit'] > 0:
                total_profit += trade['profit']
                winning_trades += 1
            else:
                total_loss += abs(trade['profit'])
                current_drawdown += trade['profit']
                if current_drawdown < max_drawdown:
                    max_drawdown = current_drawdown
                else:
                    current_drawdown = 0
        
        total_trades = len(self.trade_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = (total_profit - total_loss) / total_trades if total_trades > 0 else 0
        
        return {
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_drawdown': abs(max_drawdown)
        }

    def schedule_daily_summary(self):
        """Schedule daily summary report"""
        now = datetime.now()
        target_time = now.replace(hour=23, minute=59, second=0, microsecond=0)
        if now > target_time:
            target_time += timedelta(days=1)
        
        delta = (target_time - now).total_seconds()
        
        def daily_task():
            summary = self.generate_daily_summary()
            if summary and self.telegram:
                self.telegram.send_daily_summary(summary)
            self.schedule_daily_summary()  # Reschedule
        
        timer = threading.Timer(delta, daily_task)
        timer.daemon = True
        timer.start()

    def run(self, check_interval=5):
        """Run the trading strategy"""
        start_time = datetime.now()
        logger.info("=" * 50)
        logger.info(f"Live Trading Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log file: {self.log_filename}")
        logger.info("=" * 50)
        logger.info(f"\nTrading Parameters:")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Strategy: {self.strategy.__class__.__name__}")
        logger.info(f"Timeframe: {self.timeframe_to_str(self.timeframe)}")
        logger.info(f"Risk per trade: ${self.risk_per_trade}")
        logger.info("=" * 50)
        
        # Send startup notification
        if self.telegram:
            startup_message = (
                "🤖 Trading Bot Started!\n\n"
                f"Symbol: {self.symbol}\n"
                f"Strategy: {self.strategy.__class__.__name__}\n"
                f"Timeframe: {self.timeframe_to_str(self.timeframe)}\n"
                f"Risk per trade: ${self.risk_per_trade}\n\n"
                f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.telegram.send_message(startup_message)
        
        # Schedule daily summary
        self.schedule_daily_summary()
        
        try:
            while True:
                self.check_for_signals()
                self.update_positions()
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("\nStopping live trading...")
            if self.telegram:
                self.telegram.send_message("⚠️ Trading Bot Stopped!")
            self.connector.disconnect()
            
    def timeframe_to_str(self, timeframe):
        """Convert MT5 timeframe constant to human-readable string"""
        timeframes = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4"
        }
        return timeframes.get(timeframe, f"Unknown ({timeframe})")

    def _place_test_trade(self):
        """Place test trades with small size and tight SL/TP"""
        logger.info("\n=== Starting Test Trade Placement ===")
        logger.info(f"Current symbol setting: {self.symbol}")
        
        # Ensure connection and symbol initialization
        if not self.connector.connected:
            logger.info("Connector not connected, attempting to connect...")
            if not self.connector.connect():
                logger.error("Failed to connect to MT5")
                sys.exit(1)  # Exit if connection fails
        
        logger.info("Attempting to select and verify symbol...")
        if not mt5.symbol_select(self.symbol, True):
            error = mt5.last_error()
            logger.error(f"Failed to select symbol {self.symbol}. Error: {error[0]} - {error[1]}")
            sys.exit(1)  # Exit if symbol selection fails
            
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            error = mt5.last_error()
            logger.error(f"Failed to get symbol info for {self.symbol}. Error: {error[0]} - {error[1]}")
            sys.exit(1)  # Exit if symbol info retrieval fails
            
        logger.info("Symbol info retrieved successfully:")
        symbol_dict = symbol_info._asdict()
        important_fields = ['bid', 'ask', 'point', 'trade_mode', 'trade_contract_size', 'volume_min', 'volume_max', 'trade_stops_level', 'digits']
        for field in important_fields:
            logger.info(f"- {field}: {symbol_dict.get(field)}")
            
        # Get current price
        current_tick = mt5.symbol_info_tick(self.symbol)
        if current_tick is None:
            error = mt5.last_error()
            logger.error(f"Failed to get current price. Error: {error[0]} - {error[1]}")
            sys.exit(1)  # Exit if current price retrieval fails
            
        # Calculate common parameters
        min_stop_distance = (symbol_info.trade_stops_level + 10) * symbol_info.point  # Add 10 points buffer
        digits = symbol_info.digits  # Get the number of decimal places for the symbol
        test_risk = 10  # Small test risk amount ($10)
        
        # BUY ORDER PARAMETERS
        buy_entry = current_tick.ask  # Use ask price for buy order
        buy_sl = round(buy_entry - (min_stop_distance * 2), digits)  # Double the minimum distance for safety
        buy_tp = round(buy_entry + (min_stop_distance * 4), digits)  # 2:1 reward:risk ratio
        
        # Calculate test position size for buy
        buy_sl_distance = abs(buy_entry - buy_sl)
        raw_buy_test_volume = test_risk / (buy_sl_distance * 100)  # 100 is pip value for 1.0 lot XAUUSD
        buy_test_volume = self.validate_volume(raw_buy_test_volume)
        
        # SELL ORDER PARAMETERS
        sell_entry = current_tick.bid  # Use bid price for sell order
        sell_sl = round(sell_entry + (min_stop_distance * 2), digits)  # Double the minimum distance for safety
        sell_tp = round(sell_entry - (min_stop_distance * 4), digits)  # 2:1 reward:risk ratio
        
        # Calculate test position size for sell
        sell_sl_distance = abs(sell_entry - sell_sl)
        raw_sell_test_volume = test_risk / (sell_sl_distance * 100)  # 100 is pip value for 1.0 lot XAUUSD
        sell_test_volume = self.validate_volume(raw_sell_test_volume)
        
        # Validate stop levels for both orders
        if abs(buy_entry - buy_sl) < (symbol_info.trade_stops_level * symbol_info.point):
            logger.error(f"Buy stop loss too close to entry price. Minimum distance: {symbol_info.trade_stops_level} points")
            sys.exit(1)  # Exit if buy stop loss validation fails
            
        if abs(buy_tp - buy_entry) < (symbol_info.trade_stops_level * symbol_info.point):
            logger.error(f"Buy take profit too close to entry price. Minimum distance: {symbol_info.trade_stops_level} points")
            sys.exit(1)  # Exit if buy take profit validation fails
            
        if abs(sell_entry - sell_sl) < (symbol_info.trade_stops_level * symbol_info.point):
            logger.error(f"Sell stop loss too close to entry price. Minimum distance: {symbol_info.trade_stops_level} points")
            sys.exit(1)  # Exit if sell stop loss validation fails
            
        if abs(sell_tp - sell_entry) < (symbol_info.trade_stops_level * symbol_info.point):
            logger.error(f"Sell take profit too close to entry price. Minimum distance: {symbol_info.trade_stops_level} points")
            sys.exit(1)  # Exit if sell take profit validation fails
        
        logger.info("\nPreparing test trade parameters:")
        logger.info("BUY ORDER:")
        logger.info(f"- Entry Price (Ask): {buy_entry:.{digits}f}")
        logger.info(f"- Stop Loss: {buy_sl:.{digits}f} ({round(abs(buy_entry - buy_sl)/symbol_info.point)} points below entry)")
        logger.info(f"- Take Profit: {buy_tp:.{digits}f} ({round(abs(buy_tp - buy_entry)/symbol_info.point)} points above entry)")
        logger.info(f"- Position Size: {buy_test_volume} lots")
        logger.info("\nSELL ORDER:")
        logger.info(f"- Entry Price (Bid): {sell_entry:.{digits}f}")
        logger.info(f"- Stop Loss: {sell_sl:.{digits}f} ({round(abs(sell_entry - sell_sl)/symbol_info.point)} points above entry)")
        logger.info(f"- Take Profit: {sell_tp:.{digits}f} ({round(abs(sell_tp - sell_entry)/symbol_info.point)} points below entry)")
        logger.info(f"- Position Size: {sell_test_volume} lots")
        logger.info(f"\nCommon Parameters:")
        logger.info(f"- Minimum Stop Level: {symbol_info.trade_stops_level} points")
        logger.info(f"- Point Size: {symbol_info.point}")
        logger.info(f"- Digits: {digits}")
        
        # Execute test orders
        success = True
        logger.info("\nAttempting to execute test orders...")
        
        # Place buy order
        logger.info("\nPlacing BUY test order...")
        if self.connector.execute_order(
            self.symbol,
            'buy',
            buy_test_volume,
            price=0.0,  # Market order
            stop_loss=buy_sl,
            take_profit=buy_tp
        ):
            logger.info("Buy test trade placed successfully")
            buy_result = self.connector.last_order_result
            if buy_result:
                logger.info(f"Buy test trade details:")
                logger.info(f"- Ticket: {buy_result.order}")
                logger.info(f"- Execution Price: {buy_result.price}")
                logger.info(f"- Volume: {buy_result.volume}")
                
                # Send notification for buy test trade
                if self.telegram:
                    buy_message = (
                        f"🧪 <b>Buy Test Trade Placed</b>\n"
                        f"Symbol: {self.symbol}\n"
                        f"📈 Entry Price: {buy_result.price:.{digits}f}\n"
                        f"🛑 Stop Loss: {buy_sl:.{digits}f} ({round(abs(buy_entry - buy_sl)/symbol_info.point)} points)\n"
                        f"🎯 Take Profit: {buy_tp:.{digits}f} ({round(abs(buy_tp - buy_entry)/symbol_info.point)} points)\n"
                        f"📊 Position Size: {buy_test_volume} lots\n"
                        f"🎫 Ticket: {buy_result.order}"
                    )
                    buy_message_id = self.telegram.send_message(buy_message)
                    if buy_result.order:
                        self.position_messages[buy_result.order] = buy_message_id
                        logger.info(f"Buy test trade notification sent with message ID: {buy_message_id}")
        else:
            logger.error("Failed to place buy test trade")
            success = False
            sys.exit(1)  # Exit if buy test trade fails
            
        # Place sell order
        logger.info("\nPlacing SELL test order...")
        if self.connector.execute_order(
            self.symbol,
            'sell',
            sell_test_volume,
            price=0.0,  # Market order
            stop_loss=sell_sl,
            take_profit=sell_tp
        ):
            logger.info("Sell test trade placed successfully")
            sell_result = self.connector.last_order_result
            if sell_result:
                logger.info(f"Sell test trade details:")
                logger.info(f"- Ticket: {sell_result.order}")
                logger.info(f"- Execution Price: {sell_result.price}")
                logger.info(f"- Volume: {sell_result.volume}")
                
                # Send notification for sell test trade
                if self.telegram:
                    sell_message = (
                        f"🧪 <b>Sell Test Trade Placed</b>\n"
                        f"Symbol: {self.symbol}\n"
                        f"📉 Entry Price: {sell_result.price:.{digits}f}\n"
                        f"🛑 Stop Loss: {sell_sl:.{digits}f} ({round(abs(sell_entry - sell_sl)/symbol_info.point)} points)\n"
                        f"🎯 Take Profit: {sell_tp:.{digits}f} ({round(abs(sell_tp - sell_entry)/symbol_info.point)} points)\n"
                        f"📊 Position Size: {sell_test_volume} lots\n"
                        f"🎫 Ticket: {sell_result.order}"
                    )
                    sell_message_id = self.telegram.send_message(sell_message)
                    if sell_result.order:
                        self.position_messages[sell_result.order] = sell_message_id
                        logger.info(f"Sell test trade notification sent with message ID: {sell_message_id}")
        else:
            logger.error("Failed to place sell test trade")
            success = False
            sys.exit(1)  # Exit if sell test trade fails
            
        if success:
            logger.info("=== Test Trade Placement Complete ===\n")
        else:
            logger.error("=== Test Trade Placement Failed ===\n")
            sys.exit(1)  # Exit if any test trade failed
        return success 

    def validate_volume(self, volume: float, min_volume: float = 0.01, max_volume: float = 200.0, step: float = 0.01) -> float:
        """
        Validate and adjust trading volume according to broker requirements.
        
        Parameters:
        -----------
        volume : float
            The calculated position size
        min_volume : float
            Minimum allowed volume (default: 0.01)
        max_volume : float
            Maximum allowed volume (default: 200.0)
        step : float
            Volume step size (default: 0.01)
            
        Returns:
        --------
        float
            Validated volume that meets broker requirements
        """
        # Round to nearest step
        volume = round(volume / step) * step
        
        # Ensure within limits
        volume = max(min_volume, min(volume, max_volume))
        
        # Ensure exactly 2 decimal places
        return round(volume, 2) 