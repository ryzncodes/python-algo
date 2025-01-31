from typing import Dict, Type
from .base_strategy import BaseStrategy
from .ma_cross import MACrossStrategy

class StrategyFactory:
    _strategies: Dict[str, Type[BaseStrategy]] = {
        "MA_CROSS": MACrossStrategy,
        # Add more strategies here
    }
    
    @staticmethod
    def create_strategy(name: str, **kwargs):
        if name == "MA_CROSS":
            return MACrossStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {name}")
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]):
        """Register a new strategy"""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def available_strategies(cls) -> list:
        """Get list of available strategies"""
        return list(cls._strategies.keys()) 