import telegram
import asyncio
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Parameters:
        -----------
        bot_token : str
            Telegram bot token
        chat_id : str
            Telegram chat ID or channel name
        """
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id
        
    def _send_async(self, message: str, reply_to: Optional[int] = None) -> Optional[int]:
        """Send message asynchronously"""
        try:
            # Create event loop if it doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Send message
            result = loop.run_until_complete(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='HTML',
                    reply_to_message_id=reply_to
                )
            )
            return result.message_id
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return None
            
    def send_message(self, message: str, reply_to: Optional[int] = None) -> Optional[int]:
        """
        Send a message to Telegram
        
        Parameters:
        -----------
        message : str
            The message to send
        reply_to : Optional[int]
            Message ID to reply to
            
        Returns:
        --------
        Optional[int]
            Message ID if successful, None otherwise
        """
        return self._send_async(message, reply_to)
        
    def edit_message(self, message_id: int, new_text: str) -> bool:
        """Edit an existing message"""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=message_id,
                    text=new_text,
                    parse_mode='HTML'
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to edit Telegram message: {e}")
            return False
            
    def delete_message(self, message_id: int) -> bool:
        """Delete a message"""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                self.bot.delete_message(
                    chat_id=self.chat_id,
                    message_id=message_id
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete Telegram message: {e}")
            return False
            
    def send_daily_summary(self, summary: Dict) -> Optional[int]:
        """Send daily trading summary"""
        message = (
            "📊 Daily Trading Summary\n\n"
            f"Total Trades: {summary['total_trades']}\n"
            f"Winning Trades: {summary['winning_trades']}\n"
            f"Win Rate: {summary['win_rate']:.1%}\n\n"
            f"Total Profit: ${summary['total_profit']:.2f}\n"
            f"Total Loss: ${summary['total_loss']:.2f}\n"
            f"Net Profit: ${summary['net_profit']:.2f}\n"
            f"Average Profit: ${summary['avg_profit']:.2f}\n"
            f"Max Drawdown: ${summary['max_drawdown']:.2f}"
        )
        return self.send_message(message)

    def send_signal_notification(self, signal_type, entry_price, stop_loss, take_profit, position_size):
        # Calculate pip distances
        sl_pips = abs(entry_price - stop_loss) * 10
        tp_pips = abs(take_profit - entry_price) * 10
        
        message = (
            f"🚨 <b>New {signal_type.upper()} Signal</b> 🚨\n"
            f"📈 Entry Price: {entry_price:.2f}\n"
            f"🛑 Stop Loss: {stop_loss:.2f} ({sl_pips:.1f} pips)\n"
            f"🎯 Take Profit: {take_profit:.2f} ({tp_pips:.1f} pips)\n"
            f"📊 Position Size: {position_size:.2f} lots\n"
            f"📐 Risk:Reward = 1:{tp_pips/sl_pips:.1f}"
        )
        return self._send_async(message)

    def send_position_update(self, positions):
        if not positions:
            return self.send_message("📭 No open positions")
        
        message = "📊 <b>Position Updates</b>\n\n"
        for pos in positions:
            profit = float(pos['profit'])
            # Convert position type to string
            position_type = 'buy' if pos['type'] == 0 else 'sell'
            
            message += (
                f"⚡ {position_type.capitalize()} {pos['volume']:.2f} lots\n"
                f"🏷 Ticket: {pos['ticket']}\n"
                f"💰 Profit: ${profit:.2f}\n"
                f"📈 Current Price: {pos['price_current']:.2f}\n"
                f"🔻 SL: {pos['sl']:.2f} | 🔺 TP: {pos['tp']:.2f}\n"
                "────────────────────\n"
            )
        return self.send_message(message)

    def send_tp_sl_notification(self, position_type, entry_price, exit_price, profit, duration, closure_type, reply_to=None, additional_info=""):
        # Calculate pip distance
        pip_distance = abs(exit_price - entry_price) * 10
        
        # Determine emoji based on closure type and profit
        if closure_type == "TP":
            emoji = "🎯"
            title = "Take Profit Hit"
        else:  # SL
            emoji = "🛑"
            title = "Stop Loss Hit"
            
        message = (
            f"{emoji} <b>{title}</b> {emoji}\n"
            f"Type: {position_type.upper()}\n"
            f"Entry: ${entry_price:.2f}\n"
            f"Exit: ${exit_price:.2f}\n"
            f"Distance: {pip_distance:.1f} pips\n"
            f"💰 Profit: ${profit:.2f}\n"
            f"⏱ Duration: {duration}"
            f"{additional_info}"  # Add the additional info if any
        )
        
        return self.send_message(message, reply_to)

    def send_sl_breakeven_notification(self, ticket, entry_price, current_price, profit_distance):
        message = (
            f"🔄 <b>SL Moved to Breakeven</b>\n"
            f"🎫 Ticket: {ticket}\n"
            f"📥 Entry: {entry_price:.2f}\n"
            f"📈 Current Price: {current_price:.2f}\n"
            f"📏 Profit Distance: {profit_distance:.1f} pips"
        )
        return self.send_message(message)

    def update_position_message(self, position_info):
        """Only send position updates for new trades, TP, or SL"""
        # Skip regular position updates
        if 'is_new' not in position_info:  # Skip if not a new trade
            return
        
        message = (
            f"New Trade Opened:\n"
            f"Symbol: {position_info['symbol']}\n"
            f"Type: {position_info['type']}\n"
            f"Volume: {position_info['volume']:.2f}\n"
            f"Open Price: {position_info['open_price']:.2f}\n"
            f"SL: {position_info['sl']:.2f}\n"
            f"TP: {position_info['tp']:.2f}"
        )
        
        self.send_message(message)

    def send_breakeven_notification(self, ticket, half_volume, partial_close_profit, breakeven_sl, current_price, entry_price, reply_to=None):
        message = (
            f"🔄 <b>Position Partially Closed at Breakeven</b>\n"
            f"🎫 Ticket: {ticket}\n"
            f"📉 Closed Amount: {half_volume} lots\n"
            f"💰 Partial Profit: ${partial_close_profit:.2f}\n"
            f"📈 Remaining Position: {half_volume} lots\n"
            f"🎯 New SL: {breakeven_sl:.2f} (Breakeven + {(breakeven_sl - entry_price)*10:.1f} pips buffer)\n"
            f"📊 Current Price: {current_price:.2f}\n"
            f"📏 Profit Distance: {(current_price - entry_price)*10:.1f} pips"
        )
        return self.send_message(message, reply_to) 