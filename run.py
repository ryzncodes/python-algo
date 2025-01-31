import logging
from config.trading_config import TradingConfig
from traders.live_trader import LiveTrader
from strategies.factory import StrategyFactory
from connectors.mt5_connector import MT5Connector

def main():
    # Load configuration
    config = TradingConfig()
    
    # Create strategy using factory
    strategy = StrategyFactory.create_strategy(
        config.STRATEGY_NAME,
        fast_period=config.FAST_PERIOD,
        slow_period=config.SLOW_PERIOD,
        min_stop_pips=config.MIN_STOP_PIPS,
        max_stop_pips=config.MAX_STOP_PIPS
    )
    
    # Create connector
    connector = MT5Connector()
    
    # Create and run trader
    trader = LiveTrader(
        strategy=strategy,
        connector=connector,
        symbol=config.SYMBOL,
        timeframe=config.TIMEFRAME,
        risk_per_trade=config.RISK_PER_TRADE,
        telegram_config=config.telegram_config,
        test_mode=True
    )
    
    trader.run(check_interval=config.CHECK_INTERVAL)

if __name__ == "__main__":
    main() 