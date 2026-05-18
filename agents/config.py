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
    # Day 1 bulk orders (must be >= supplier minimums, typically 10kg)
    "day1_bulk_normal": 20,      # kg per ingredient on Day 1 (reduced to avoid over-ordering)
    "day1_bulk_reduced": 12,     # kg when reduced capacity detected
    
    # Early week boost (Days 2-3)
    "early_week_days": [2, 3],
    "early_week_threshold": 30,  # Order if total < this (lowered)
    "early_week_qty": 12,        # Smaller orders (was 15)
    
    # Stockout recovery
    "stockout_qty": 12,          # Emergency order (was 15)
    
    # Smart restock thresholds
    "base_threshold": 12.0,      # Lower threshold (was 15)
    "day_multipliers": {
        "Monday": 2.0, "Tuesday": 1.8, "Wednesday": 1.5,
        "Thursday": 1.3, "Friday": 1.0, "Saturday": 1.0, "Sunday": 1.0
    },
    "default_daily_usage": 2.0,  # Lower estimate
    "restock_days_supply": 5,    # Less buffer (was 6)
    "min_order_qty": 10,         # MUST match supplier minimum (Italian Imports = 10kg)
    "supplier_min_order": 10,    # Global supplier minimum to respect
    
    # Expiring stock
    "expiry_days_threshold": 2,
    "expiry_remaining_threshold": 5,
    "expiry_order_qty": 12,      # Smaller (was 15)
    
    # Weekend prep (Wednesday)
    "weekend_multiplier": 3.5,   # Lower (was 4.5)
    "weekend_min_order": 15,     # Lower (was 20)
    "weekend_buffer": 5,         # Lower (was 10)
}

# =============================================================================
# STAFFING PARAMETERS
# =============================================================================

STAFFING = {
    "base_levels": {
        "Monday": 7, "Tuesday": 7, "Wednesday": 8,
        "Thursday": 9, "Friday": 13, "Saturday": 14, "Sunday": 5
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
    "capacity_keywords": ["renovation", "capacity", "table", "construction"],
    "supply_crisis_keywords": ["halted", "outage", "unavailable", "crisis", "stopped"],
    "low_covers_threshold": 80,
    "low_covers_after_day": 5,
}

# =============================================================================
# TOURIST SEASON PARAMETERS
# =============================================================================

TOURIST_SEASON = {
    # Surge detection
    "surge_start_day": 9,
    "surge_end_day": 14,
    "surge_covers_threshold": 250,  # Covers indicating surge
    
    # Pre-stocking (days 5-8)
    "prestock_start_day": 5,
    "prestock_end_day": 8,
    "prestock_multiplier": 1.8,  # Order 80% more
    
    # During surge (days 9-14)
    "surge_order_multiplier": 1.5,
    "surge_staff_boost": 3,  # Add 3 staff during surge
    "surge_price_premium": 1.12,  # 12% premium during surge
    
    # Post-surge (days 15+)
    "postsurge_order_multiplier": 0.7,
    "postsurge_staff_reduction": 2,
    
    # Expected demand by phase
    "normal_covers_range": (100, 200),
    "surge_covers_range": (300, 470),
    "drop_covers_range": (50, 180),
}

# =============================================================================
# SUPPLY CRISIS PARAMETERS
# =============================================================================

SUPPLY_CRISIS = {
    # Detection
    "crisis_keywords": ["halted", "outage", "unavailable", "stopped", "suspended"],
    
    # Response
    "emergency_order_multiplier": 1.5,  # Order extra from remaining suppliers
    "buffer_days": 4,  # Days of extra stock to maintain
    
    # Alternative supplier priority (lower = higher priority)
    "supplier_priority": {
        "Fresh Farms NL": 1,
        "Italian Imports": 2,
        "Local Produce": 3,
        "Premium Goods": 4,
    },
}

# =============================================================================
# RENOVATION PARAMETERS
# =============================================================================

RENOVATION = {
    "duration_days": 14,
    "capacity_reduction": 0.5,  # 50% capacity
    "order_multiplier": 0.6,  # 60% of normal orders
    "staff_reduction": 3,
    "starting_cash": 17000,  # Extra cash for renovation scenario
}

# =============================================================================
# PROBABILITY & RISK PARAMETERS
# =============================================================================

RISK = {
    # Cash safety thresholds
    "bankruptcy_critical": 3000,
    "bankruptcy_danger": 5000,
    "bankruptcy_caution": 8000,
    "healthy_cash": 12000,
    
    # Stockout risk
    "stockout_buffer_multiplier": 1.3,  # 30% buffer above expected usage
    "emergency_restock_threshold": 3.0,  # kg below which is critical
    
    # Walkout cost estimates
    "walkout_direct_cost": 20,  # Lost revenue per walkout
    "walkout_review_cost": 15,  # Reputation damage
    
    # Waste cost
    "waste_cost_per_kg": 2.0,
    
    # Staff efficiency
    "covers_per_staff": 15,
    "diminishing_returns_threshold": 12,  # Staff above this has reduced efficiency
}

# =============================================================================
# DEMAND EXPECTATIONS BY DAY
# =============================================================================

DEMAND = {
    "Sunday": {"min": 0, "expected": 0, "max": 0},
    "Monday": {"min": 70, "expected": 85, "max": 110},
    "Tuesday": {"min": 75, "expected": 90, "max": 115},
    "Wednesday": {"min": 90, "expected": 120, "max": 150},
    "Thursday": {"min": 100, "expected": 135, "max": 165},
    "Friday": {"min": 140, "expected": 200, "max": 260},
    "Saturday": {"min": 150, "expected": 220, "max": 280},
}

# =============================================================================
# STATE MACHINE SCENARIO MULTIPLIERS
# =============================================================================
# These are the EXACT multipliers for each scenario phase.
# Key insight: During SURGE (days 9-14), order_multiplier is 1.0 NOT 2.5
# because we already pre-stocked on days 5-8. Ordering 2.5x during surge
# causes waste when the crash hits on day 15.

SCENARIO_MULTIPLIERS = {
    "baseline": {"order": 1.0, "staff": 0, "price": 1.0},
    "renovation": {"order": 0.5, "staff": -3, "price": 0.95},
    "supply_crisis": {"order": 1.5, "staff": 0, "price": 1.0},
    "tourist_normal": {"order": 1.0, "staff": 0, "price": 1.0},
    "tourist_prestock": {"order": 2.5, "staff": 0, "price": 1.0},   # Days 5-8: PRE-STOCK
    "tourist_surge": {"order": 1.0, "staff": 3, "price": 1.15},     # Days 9-14: Already stocked!
    "tourist_crash": {"order": 0.4, "staff": -2, "price": 0.9},     # Days 15+: Scale down
}

# =============================================================================
# DAY-OF-WEEK MODIFIERS (applied ON TOP of scenario multipliers)
# =============================================================================
# These boost revenue on high-demand days without changing the scenario phase.
# Price premiums require reputation check before applying.

DAY_OF_WEEK_MODIFIERS = {
    "Sunday": {"order": 0.3, "staff": 0, "price": 1.0},      # Closed - minimal
    "Monday": {"order": 1.0, "staff": 0, "price": 1.0},      # Low demand
    "Tuesday": {"order": 1.0, "staff": 0, "price": 1.0},     # Low demand
    "Wednesday": {"order": 1.2, "staff": 0, "price": 1.0},   # Prep for weekend
    "Thursday": {"order": 1.1, "staff": 0, "price": 1.0},    # Building up
    "Friday": {"order": 1.0, "staff": 1, "price": 1.08},     # HIGH demand - premium
    "Saturday": {"order": 1.0, "staff": 2, "price": 1.08},   # HIGHEST demand - premium
}
