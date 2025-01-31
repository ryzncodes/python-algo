from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
from connectors.base_connector import BaseConnector
from strategies.base_strategy import BaseStrategy
from notifications.telegram_notifier import TelegramNotifier

class BaseTrader(ABC):
    """Abstract base class for all traders"""
    
    def __init__(self,
                 strategy: BaseStrategy,
                 connector: BaseConnector,
                 symbol: str,
                 timeframe: int,
                 risk_per_trade: float,
                 min_lot: float,
                 max_lot: float,
                 telegram_config: Optional[dict] = None):
        
        self.strategy = strategy
        self.connector = connector
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.min_lot = min_lot
        self.max_lot = max_lot
        
        # Initialize telegram notifier if config provided
        self.telegram = None
        if telegram_config and telegram_config.get('enabled', False):
            self.telegram = TelegramNotifier(
                bot_token=telegram_config['bot_token'],
                chat_id=telegram_config['chat_id']
            )
    
    @abstractmethod
    def check_for_signals(self):
        """Check for new trading signals"""
        pass
    
    @abstractmethod
    def update_positions(self):
        """Update and monitor open positions"""
        pass
    
    @abstractmethod
    def run(self, check_interval: int = 5):
        """Run the trading strategy"""
        pass 