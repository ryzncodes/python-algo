# XAUUSD Trading System

A professional Python-based trading system specifically designed for XAUUSD (Gold) trading, featuring both live trading and backtesting capabilities with MetaTrader 5 integration.

## Features

- Live trading with MetaTrader 5 integration
- Real-time Telegram notifications for trades and updates
- Test mode for safe strategy validation
- Moving Average Crossover strategy optimization
- Dynamic position sizing based on risk amount
- Advanced risk management features
- Comprehensive backtesting engine
- Multi-timeframe analysis (M5 to H4)
- Daily performance summaries
- Detailed trade analytics

## Core Components

- Live trading system with position management
- Real-time price monitoring and signal generation
- Telegram integration for trade notifications
- Automated stop-loss and take-profit management
- Risk-based position sizing calculator
- Performance analytics and reporting

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd xauusd-trading-system
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Configure your settings:
   - Copy `config/trading_config_example.py` to `config/trading_config.py`
   - Update MetaTrader 5 and Telegram credentials
   - Adjust trading parameters as needed

## Usage

### Live Trading

```python
from run import main

# Start live trading with test mode enabled
main()
```

### Backtesting

```python
from test_xauusd import test_strategy_all_timeframes

# Run backtesting across all timeframes
test_strategy_all_timeframes()
```

## Project Structure

- `run.py` - Main script for live trading
- `test_xauusd.py` - Backtesting and strategy optimization
- `config/` - Trading configuration and parameters
- `traders/` - Live trading implementation
- `strategies/` - Trading strategy implementations
- `notifications/` - Telegram notification system
- `connectors/` - MT5 connection handling
- `logs/` - Trading logs and performance records

## Configuration

Edit `config/trading_config.py` to set:

- Strategy parameters (MA periods, stop loss ranges)
- Trading parameters (symbol, timeframe, risk per trade)
- Telegram settings (bot token, chat ID)
- Test mode settings

## Risk Management

- Dynamic position sizing based on risk amount
- Automatic stop-loss calculation
- Take-profit optimization
- Maximum drawdown protection
- Spread monitoring
- Test mode for strategy validation

## Notifications

The system sends real-time Telegram notifications for:

- Trade entries and exits
- Stop-loss and take-profit hits
- Position updates
- Daily trading summaries
- Performance metrics

## Performance Analytics

Detailed analytics including:

- Win rate and profit metrics
- Drawdown analysis
- Risk-adjusted returns
- Trade distribution
- Weekly performance breakdown
- Equity curve visualization

## Safety Features

- Test mode for safe strategy validation
- Maximum position size limits
- Spread monitoring
- Error handling and logging
- Automatic MT5 reconnection

## Requirements

- Python 3.8+
- MetaTrader 5 terminal
- Active Telegram bot for notifications
- Required Python packages (see requirements.txt)

## License

MIT License
