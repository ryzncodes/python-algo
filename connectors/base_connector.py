from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
from datetime import datetime

class BaseConnector(ABC):
    """Abstract base class for broker connections"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection to broker"""
        pass
        
    @abstractmethod
    def disconnect(self):
        """Close broker connection"""
        pass
        
    @abstractmethod
    def get_historical_data(self, 
                          symbol: str, 
                          timeframe: int,
                          start_date: datetime,
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        pass
        
    @abstractmethod
    def execute_order(self, 
                     symbol: str, 
                     order_type: str, 
                     volume: float,
                     price: float = 0.0,
                     stop_loss: float = 0.0,
                     take_profit: float = 0.0) -> bool:
        """Execute trading order"""
        pass
        
    @abstractmethod
    def get_positions(self) -> List[dict]:
        """Get current open positions"""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Get account balance"""
        pass

    @abstractmethod
    def get_equity(self) -> float:
        """Get account equity"""
        pass

    @abstractmethod
    def get_margin(self) -> float:
        """Get used margin"""
        pass 