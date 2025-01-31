from dataclasses import dataclass
import MetaTrader5 as mt5
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    # Strategy parameters
    STRATEGY_NAME: str = "MA_CROSS"
    FAST_PERIOD: int = 19
    SLOW_PERIOD: int = 20
    MIN_STOP_PIPS: float = 10
    MAX_STOP_PIPS: float = 80
    
    # Trading parameters
    SYMBOL: str = os.getenv('SYMBOL', 'XAUUSDc')
    TIMEFRAME: int = mt5.TIMEFRAME_M5
    RISK_PER_TRADE: float = float(os.getenv('RISK_PER_TRADE', '200'))
    CHECK_INTERVAL: int = 1
    
    # Telegram settings
    TELEGRAM_ENABLED: bool = True
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')

    @property
    def telegram_config(self):
        return {
            'enabled': self.TELEGRAM_ENABLED,
            'bot_token': self.TELEGRAM_BOT_TOKEN,
            'chat_id': self.TELEGRAM_CHAT_ID
        } 