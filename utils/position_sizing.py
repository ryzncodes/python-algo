def calculate_position_size(price: float,
                          stop_loss: float,
                          risk_amount: float,
                          min_lot: float = 0.01,
                          max_lot: float = 100.0) -> float:
    """
    Calculate position size based on risk parameters
    
    Parameters:
    -----------
    price : float
        Entry price
    stop_loss : float
        Stop loss price
    risk_amount : float
        Amount to risk in account currency
    min_lot : float
        Minimum lot size
    max_lot : float
        Maximum lot size
    
    Returns:
    --------
    float
        Position size in lots
    """
    
    # Calculate pip value and risk in pips
    pip_value = 0.1  # For gold
    risk_pips = abs(price - stop_loss) / pip_value
    
    # Calculate position size
    position_size = risk_amount / risk_pips
    
    # Round to 2 decimal places and constrain to min/max
    position_size = round(position_size, 2)
    position_size = max(min_lot, min(position_size, max_lot))
    
    return position_size 