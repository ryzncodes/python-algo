import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5Connector:
    def __init__(self):
        self.connected = False
        self.last_order_result = None  # Track last order result
        
    def connect(self):
        """Initialize connection to MT5 Terminal"""
        logger.info("Attempting to connect to MT5...")
        
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed. Error code: {error[0]}, Description: {error[1]}")
            return False
            
        # Get terminal info
        terminal_info = mt5.terminal_info()._asdict()
        if terminal_info:
            logger.info("MT5 Terminal Info:")
            for key, value in terminal_info.items():
                logger.info(f"- {key}: {value}")
                
            if not terminal_info.get('connected', False):
                logger.error("MT5 is not connected to a broker")
                return False
                
            if not terminal_info.get('trade_allowed', False):
                logger.error("Trading is not allowed")
                return False
                
            if terminal_info.get('tradeapi_disabled', True):
                logger.error("Trade API is disabled")
                return False
        else:
            error = mt5.last_error()
            logger.error(f"Could not get terminal info. Error code: {error[0]}, Description: {error[1]}")
            return False
            
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            account_dict = account_info._asdict()
            logger.info("\nAccount Info:")
            for key, value in account_dict.items():
                logger.info(f"- {key}: {value}")
        else:
            error = mt5.last_error()
            logger.error(f"Could not get account info. Error code: {error[0]}, Description: {error[1]}")
            return False

        # Initialize symbol
        symbol = "XAUUSD"  # Use exact symbol
        logger.info(f"Attempting to initialize symbol: {symbol}")
        
        if mt5.symbol_select(symbol, True):
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                logger.info(f"Successfully initialized symbol: {symbol}")
                logger.info("Symbol Info:")
                symbol_dict = symbol_info._asdict()
                for key, value in symbol_dict.items():
                    logger.info(f"- {key}: {value}")
            else:
                error = mt5.last_error()
                logger.error(f"Symbol info not available. Error code: {error[0]}, Description: {error[1]}")
                return False
        else:
            error = mt5.last_error()
            logger.error(f"Failed to select symbol. Error code: {error[0]}, Description: {error[1]}")
            return False
            
        self.connected = True
        logger.info("Successfully connected to MT5")
        return True
    
    def disconnect(self):
        """Shutdown connection to MT5 Terminal"""
        logger.info("Disconnecting from MT5...")
        mt5.shutdown()
        self.connected = False
    
    def get_historical_data(self, symbol, timeframe, start_date, end_date=None):
        """
        Fetch historical data from MT5
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., "XAUUSD")
        timeframe : mt5.TIMEFRAME_*
            Timeframe for the data
        start_date : datetime
            Start date for historical data
        end_date : datetime, optional
            End date for historical data
            
        Returns:
        --------
        pd.DataFrame
            Historical price data with columns: time, open, high, low, close, volume
        """
        logger.info(f"Fetching historical data for {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Start date: {start_date}")
        logger.info(f"End date: {end_date}")
        
        if not self.connected:
            logger.info("Not connected to MT5, attempting to connect...")
            if not self.connect():
                return None
        
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found in MT5")
            logger.info("Available symbols:")
            symbols = mt5.symbols_get()
            if symbols:
                for sym in symbols[:10]:  # Show first 10 symbols
                    logger.info(f"- {sym.name}")
                logger.info("... and more")
            else:
                logger.error("No symbols available")
            return None
                
        # Enable symbol for trading if needed
        if not symbol_info.visible:
            logger.info(f"Enabling symbol {symbol} for trading")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to enable symbol {symbol}")
                return None
                
        timezone = pytz.timezone("UTC")
        start_date = pd.to_datetime(start_date).tz_localize(timezone)
        if end_date:
            end_date = pd.to_datetime(end_date).tz_localize(timezone)
        else:
            end_date = datetime.now(timezone)
            
        logger.info("Requesting historical data from MT5...")
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.error(f"Error getting historical data: {error}")
            return None
            
        logger.info(f"Received {len(rates)} data points")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def get_symbols(self):
        """Get all available symbols"""
        if not self.connected:
            if not self.connect():
                return None
        return mt5.symbols_get()

    def __del__(self):
        """Destructor to ensure MT5 connection is closed"""
        if self.connected:
            self.disconnect()

    def execute_order(self, symbol: str, order_type: str, volume: float, price: float = 0.0,
                     stop_loss: float = 0.0, take_profit: float = 0.0, deviation: int = 20) -> bool:
        """Execute a market order"""
        logger.info(f"\nExecuting order:")
        logger.info(f"- Symbol: {symbol}")
        logger.info(f"- Type: {order_type}")
        logger.info(f"- Volume: {volume}")
        logger.info(f"- Price: {price}")
        logger.info(f"- Stop Loss: {stop_loss}")
        logger.info(f"- Take Profit: {take_profit}")
        
        if not self.connected:
            logger.info("Not connected, attempting to connect...")
            if not self.connect():
                logger.error("Failed to connect to MT5")
                return False
                
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            error = mt5.last_error()
            logger.error(f"Symbol {symbol} not found. Error code: {error[0]}, Description: {error[1]}")
            return False
            
        # Enable symbol for trading if needed
        if not symbol_info.visible:
            logger.info(f"Symbol {symbol} not visible, attempting to select...")
            if not mt5.symbol_select(symbol, True):
                error = mt5.last_error()
                logger.error(f"Failed to select symbol {symbol}. Error code: {error[0]}, Description: {error[1]}")
                return False
            logger.info(f"Successfully selected symbol {symbol}")
                
        # Get current tick info
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            error = mt5.last_error()
            logger.error(f"Failed to get tick info. Error code: {error[0]}, Description: {error[1]}")
            return False
            
        logger.info(f"Current tick info:")
        logger.info(f"- Bid: {tick.bid}")
        logger.info(f"- Ask: {tick.ask}")
        logger.info(f"- Spread: {tick.ask - tick.bid}")
                
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if order_type.lower() == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if order_type.lower() == 'buy' else tick.bid,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": deviation,
            "magic": 234000,
            "comment": "python script order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        logger.info("Sending order request:")
        for key, value in request.items():
            logger.info(f"- {key}: {value}")
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error()
            logger.error(f"Order failed, retcode={result.retcode}")
            logger.error(f"MT5 Error: code={error[0]}, description={error[1]}")
            logger.error(f"Result Info:")
            for key, value in result._asdict().items():
                logger.error(f"- {key}: {value}")
            return False
            
        logger.info("Order executed successfully!")
        logger.info(f"- Order ID: {result.order}")
        logger.info(f"- Execution Price: {result.price}")
        self.last_order_result = result
        return True
        
    def get_positions(self) -> list:
        """Get all open positions"""
        if not self.connected:
            if not self.connect():
                return []
                
        positions = mt5.positions_get()
        if positions is None:
            logger.error(f"No positions found, error code={mt5.last_error()}")
            return []
            
        return [pos._asdict() for pos in positions]
        
    def close_position(self, position_id: int) -> bool:
        """
        Close a specific position
        
        Parameters:
        -----------
        position_id : int
            Position ticket number
            
        Returns:
        --------
        bool
            True if position was closed successfully
        """
        if not self.connected:
            if not self.connect():
                return False
                
        # Get position info
        position = mt5.positions_get(ticket=position_id)
        if position is None:
            logger.error(f"Position {position_id} not found")
            return False
            
        position = position[0]._asdict()
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position['symbol'],
            "volume": position['volume'],
            "type": mt5.ORDER_TYPE_SELL if position['type'] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": position_id,
            "price": mt5.symbol_info_tick(position['symbol']).ask if position['type'] == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position['symbol']).bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close request
        logger.info(f"Closing position {position_id}...")
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed, retcode={result.retcode}")
            return False
            
        logger.info("Position closed successfully!")
        return True
        
    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        """Modify an existing position's SL/TP"""
        try:
            # Create the request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": mt5.positions_get(ticket=ticket)[0].symbol,
            }
            
            # Only add SL/TP to request if they are provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
                
            # Send order to MT5
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to modify position {ticket}. Error code: {result.retcode}")
                return False
                
            logger.info(f"Successfully modified position {ticket}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False

    def get_balance(self):
        if not self.connected:
            self.connect()
        return mt5.account_info().balance

    def get_equity(self):
        return mt5.account_info().equity

    def get_margin(self):
        return mt5.account_info().margin

    def get_position_info(self, ticket: int) -> Optional[dict]:
        """Get information about a specific position"""
        if not self.connected:
            if not self.connect():
                return None
            
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:  # Added length check
            # Try to get from history if position is closed
            history_deals = mt5.history_deals_get(
                date_from=datetime.now() - timedelta(days=7)  # Look back 7 days
            )
            
            if history_deals is None:
                logger.debug(f"Failed to get history deals")
                return None
                
            # Find the deal that closed this position
            closing_deal = None
            for deal in history_deals:
                if deal.position_id == ticket:  # Match by position ID
                    if deal.entry == 1:  # This is a closing deal
                        closing_deal = deal
                        break
            
            if closing_deal is None:
                logger.debug(f"Position {ticket} not found in open positions or recent history")
                return None
                
            # Convert deal info to position info format
            return {
                'ticket': ticket,
                'time': closing_deal.time,
                'time_close': closing_deal.time,
                'type': 0 if closing_deal.type == mt5.DEAL_TYPE_SELL else 1,  # Reverse because closing deal type is opposite
                'price_open': closing_deal.price_open,
                'close_price': closing_deal.price,
                'profit': closing_deal.profit,
                'volume': closing_deal.volume
            }
            
        # Return open position info if found
        return position[0]._asdict() if position else None

    def close_partial_position(self, ticket: int, volume: float) -> bool:
        """
        Close a portion of an existing position
        
        Parameters:
        -----------
        ticket : int
            Position ticket number
        volume : float
            Volume to close (in lots)
            
        Returns:
        --------
        bool
            True if partial position was closed successfully
        """
        if not self.connected:
            if not self.connect():
                return False
                
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position {ticket} not found")
            return False
            
        position = position[0]._asdict()
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position['symbol'],
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if position['type'] == 0 else mt5.ORDER_TYPE_BUY,  # Opposite of position type
            "price": mt5.symbol_info_tick(position['symbol']).bid if position['type'] == 0 else mt5.symbol_info_tick(position['symbol']).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "partial close python script",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send order
        logger.info(f"Closing {volume} lots of position {ticket}...")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close partial position, retcode={result.retcode}")
            logger.error(f"   Result: {result}")
            return False
            
        logger.info(f"Successfully closed {volume} lots of position {ticket}")
        return True 