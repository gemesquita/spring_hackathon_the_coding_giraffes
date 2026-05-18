"""Agent Configuration - Tweak these values to tune performance.

Usage:
    from agents.config import ORDERING, STAFFING, PRICING, MARKETING, DETECTION

Test after changes:
    python -m agents.my_agent 42 baseline
    python -m agents.my_agent 42 tourist_season
    python -m agents.my_agent 42 supply_crisis
    python -m agents.my_agent 42 renovation
"""

# =============================================================================
# ORDERING PARAMETERS
# =============================================================================

ORDERING = {
    # Day 1 bulk orders
    "day1_bulk_normal": 25,      # kg per ingredient on Day 1
    "day1_bulk_reduced": 12,     # kg when reduced capacity detected
    
    # Early week boost (Days 2-3)
    "early_week_days": [2, 3],
    "early_week_threshold": 35,  # Order if total < this
    "early_week_qty": 15,
    
    # Stockout recovery
    "stockout_qty": 15,
    
    # Smart restock thresholds
    "base_threshold": 15.0,
    "day_multipliers": {
        "Monday": 2.5, "Tuesday": 2.0, "Wednesday": 1.8,
        "Thursday": 1.5, "Friday": 1.2, "Saturday": 1.0, "Sunday": 1.0
    },
    "default_daily_usage": 2.5,
    "restock_days_supply": 6,
    "min_order_qty": 15,
    
    # Expiring stock
    "expiry_days_threshold": 2,
    "expiry_remaining_threshold": 5,
    "expiry_order_qty": 15,
    
    # Weekend prep (Wednesday)
    "weekend_multiplier": 4.5,
    "weekend_min_order": 20,
    "weekend_buffer": 10,
}

# =============================================================================
# STAFFING PARAMETERS
# =============================================================================

STAFFING = {
    "base_levels": {
        "Monday": 7, "Tuesday": 7, "Wednesday": 8,
        "Thursday": 9, "Friday": 12, "Saturday": 13, "Sunday": 5
    },
    "covers_per_staff": 15,
    "day_weight": 0.6,
    "demand_weight": 0.4,
    "reduced_capacity_cut": 3,
    
    # Weather adjustments
    "bad_weather_cut": 1,
    "sunny_weekend_boost": 1,
    
    # Walkout response
    "walkout_many_boost": 2,
    "walkout_some_boost": 1,
    
    # Cash safety
    "cash_critical": 5000,
    "cash_warning": 8000,
    "critical_cut": 2,
    "warning_cut": 1,
    
    # Limits
    "min_staff": 5,
    "max_staff": 14,
}

# =============================================================================
# PRICING PARAMETERS
# =============================================================================

PRICING = {
    "weekend_premium": 1.1,
    "poor_rep_discount": 0.9,
    "good_reputations": ["Good", "Very Good", "Excellent"],
    "poor_reputations": ["Poor", "Fair"],
}

# =============================================================================
# MARKETING PARAMETERS
# =============================================================================

MARKETING = {
    "min_cash": 5000,
    "min_cash_for_promo": 12000,
    "declining_amount": 200,
    "slow_day_amount": 100,
    "slow_days": ["Monday", "Tuesday"],
}

# =============================================================================
# SCENARIO DETECTION
# =============================================================================

DETECTION = {
    "capacity_keywords": ["renovation", "capacity", "table"],
    "low_covers_threshold": 80,
    "low_covers_after_day": 5,
}
