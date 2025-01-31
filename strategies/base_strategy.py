from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize strategy parameters"""
        pass
        
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        pass
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        pass
        
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Return strategy parameters"""
        pass
        
    @parameters.setter
    @abstractmethod
    def parameters(self, params: Dict[str, Any]):
        """Update strategy parameters"""
        pass 