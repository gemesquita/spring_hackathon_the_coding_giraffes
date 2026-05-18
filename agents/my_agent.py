"""Hybrid agent: Code handles reliable operations, LLM handles advanced strategy.

Key improvements over pure-LLM approach:
1. Code-based supplier matching (eliminates "wrong supplier" errors)
2. Code-based inventory management (prevents stockouts)
3. Code-based staff scheduling (deterministic, weather-aware)
4. Code-based daily specials (picks well-stocked dishes)
5. LLM for pricing/marketing/scenario detection (strategic judgment)
"""

from __future__ import annotations

import json
import os
import sys

import openai

from agents.runner import run_game
from agents.config import (
    ORDERING, STAFFING, PRICING, MARKETING, DETECTION,
    TOURIST_SEASON, SUPPLY_CRISIS, RENOVATION, RISK, DEMAND
)

# =============================================================================
# LLM CONFIGURATION (LiteLLM Proxy)
# =============================================================================

LLM_CLIENT = openai.OpenAI(
    api_key="sk-Q1QGmsIdw1v33-edjyEH-Q",
    base_url="http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com",
)

# Model to use - upgraded for better reasoning
LLM_MODEL = os.getenv("AGENT_MODEL", "gpt-4o")

# Token limits - increased for comprehensive context
MAX_TOKENS = 4000
LLM_TIMEOUT = 25  # Leave 5s buffer from 30s tick budget

# Enable/disable LLM (fallback to pure rules if False or on error)
USE_LLM = True

STRATEGY_PROMPT = """\
You are an EXPERT restaurant manager AI running a 30-day Italian restaurant simulation.
Your PRIMARY GOAL: Maximize final score = net_profit - penalties while NEVER going bankrupt.

═══════════════════════════════════════════════════════════════════════════════
SCORING SYSTEM (CRITICAL - memorize these weights)
═══════════════════════════════════════════════════════════════════════════════

BANKRUPTCY = -100,000 score (INSTANT GAME OVER - protect cash at ALL costs)

Penalty Formula (approximate weights):
  Final Score = Net Profit - Walkout Penalty - Reputation Penalty - Waste Penalty

  • Walkout Penalty: ~1.0 per walkout customer (linear, compounds via bad reviews)
  • Reputation Penalty: QUADRATIC when below threshold (small dip OK, big dip DEVASTATING)
    - "Poor" reputation = 100+ penalty per day
    - "Fair" reputation = 25-50 penalty per day  
    - "Good"+ reputation = 0 penalty
  • Waste Penalty: ~0.5 per kg wasted (only matters if excessive >15% waste rate)

PRIORITY ORDER:
  1. NEVER go bankrupt (keep cash > 3000 minimum, prefer > 8000)
  2. Protect reputation (quadratic penalty is severe)
  3. Minimize walkouts (direct + causes bad reviews)
  4. Control waste (moderate)
  5. Maximize revenue

═══════════════════════════════════════════════════════════════════════════════
DEMAND MATRIX (Expected covers by day)
═══════════════════════════════════════════════════════════════════════════════

Day of Week   | Min | Expected | Max  | Revenue Potential | Action Priority
--------------|-----|----------|------|-------------------|------------------
Sunday        |   0 |     0    |   0  | CLOSED            | Minimize staff (5)
Monday        |  70 |    85    | 110  | Low               | Marketing helps
Tuesday       |  75 |    90    | 115  | Low               | Marketing helps
Wednesday     |  90 |   120    | 150  | Medium            | Prep for weekend
Thursday      | 100 |   135    | 165  | Medium-High       | Order for weekend
Friday        | 140 |   200    | 260  | HIGH              | Max staff, premium prices
Saturday      | 150 |   220    | 280  | HIGHEST           | Max staff, premium prices

Staff Calculation: covers / 15 + 2 = base staff needed
Revenue per cover: ~€18-22 average

═══════════════════════════════════════════════════════════════════════════════
SCENARIO DETECTION (Critical for strategy adaptation)
═══════════════════════════════════════════════════════════════════════════════

1. BASELINE (default):
   - Day 1 cash: €15,000
   - No unusual alerts
   - Normal demand patterns
   - Strategy: Standard optimization

2. RENOVATION:
   - Day 1 cash: €17,000 (extra €2000)
   - Alert contains: "renovation", "capacity", "tables", "construction"
   - Effect: ~50% capacity for 14 days (only ~11 tables usable)
   - Strategy: 
     • CUT orders by 50% (less inventory needed)
     • REDUCE staff by 2-3 (fewer covers possible)
     • LOWER prices slightly (attract customers despite chaos)
     • After day 14: RAMP UP for normal operations

3. SUPPLY_CRISIS:
   - Alert contains: supplier name + "halted", "outage", "unavailable", "crisis"
   - Effect: One supplier stops delivering
   - Strategy:
     • IMMEDIATELY switch to alternative suppliers
     • Order EXTRA from remaining suppliers
     • May need to adjust menu if ingredients unavailable
     • Monitor for supplier recovery

4. TOURIST_SEASON:
   - NO special Day 1 signal (hidden scenario!)
   - Detection: Days 9-14 shows MASSIVE surge (300-470 covers!)
   - Pattern:
     • Days 1-8: Normal (100-200 covers)
     • Days 9-14: SURGE (300-470 covers) ← Critical period!
     • Days 15-30: DROP (50-180 covers)
   - Strategy:
     • Days 5-8: PRE-STOCK heavily (order 2x normal)
     • Days 9-14: MAX staff (14), premium prices (1.15x), no marketing needed
     • Days 15+: Scale DOWN, clear excess inventory

═══════════════════════════════════════════════════════════════════════════════
DECISION FRAMEWORK (Expected Value Calculations)
═══════════════════════════════════════════════════════════════════════════════

For each decision, calculate expected value:
  EV = P(success) × benefit - P(failure) × cost

ORDERING DECISIONS:
  • Stockout cost: ~€20 per customer who can't order (walkout + bad review)
  • Waste cost: ~€2-5 per kg spoiled
  • Decision: Order enough to have P(stockout) < 5%
  • Buffer formula: expected_demand × 1.3 = safe inventory level

STAFFING DECISIONS:
  • Under-staffed: Walkouts from slow service (€20 per walkout)
  • Over-staffed: €120 wasted per extra person
  • Sweet spot: (expected_covers / 15) + 2, max 14

PRICING DECISIONS:
  • Demand elasticity: ~10% price change = ~5% demand change
  • Premium pricing (1.1-1.15x) works when: reputation Good+, high demand day
  • Discount pricing (0.85-0.9x) works when: reputation Poor/Fair, need customers

MARKETING DECISIONS:
  • €100 marketing ≈ +10-15 covers on slow days
  • Diminishing returns: 2nd consecutive day = 50% effect
  • Skip if: cash < €8000, weekend (already busy), declining trend

═══════════════════════════════════════════════════════════════════════════════
CASH SAFETY RULES (Non-negotiable)
═══════════════════════════════════════════════════════════════════════════════

Cash Level    | Action
--------------|--------------------------------------------------
< €5,000      | EMERGENCY: Cut all non-essential spending, min staff
< €8,000      | CAUTION: No marketing, conservative orders
€8,000-12,000 | NORMAL: Standard operations
> €12,000     | HEALTHY: Can invest in marketing, buffer stock

Daily costs: ~€1,260-1,980 (staff €600-1680 + fixed €300 + orders)
Daily revenue: ~€1,500-4,500 depending on covers

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Respond with ONLY this JSON object (no markdown, no explanation):
{
  "scenario_detected": "baseline|renovation|supply_crisis|tourist_season",
  "confidence": 0.0 to 1.0,
  "scenario_reasoning": "Brief explanation of detection signals",
  
  "price_multiplier": 0.8 to 1.2,
  "price_reasoning": "Why this price adjustment",
  
  "marketing_spend": 0 to 500,
  "marketing_reasoning": "Why this spend level",
  
  "staff_adjustment": -3 to +3,
  "staff_reasoning": "Why adjust staff from baseline",
  
  "run_happy_hour": true or false,
  "happy_hour_reasoning": "Why or why not",
  
  "order_multiplier": 0.5 to 2.0,
  "order_reasoning": "Scale factor for inventory orders",
  
  "risk_assessment": "low|medium|high",
  "bankruptcy_risk": 0.0 to 1.0,
  
  "strategic_notes": "Key insights for future days (max 300 chars)"
}"""


# =============================================================================
# SUPPLIER & INVENTORY HELPERS (Code-based, no LLM errors)
# =============================================================================

def build_supplier_map(obs: dict) -> dict[str, str]:
    """Returns {ingredient: best_supplier} for guaranteed-correct ordering.
    
    Prefers suppliers with:
    1. Not mentioned in alerts (outage/halted)
    2. Lower price
    3. Shorter lead time (as tiebreaker)
    """
    alerts = obs.get("alerts", [])
    alert_text = " ".join(str(a).lower() for a in alerts)
    
    ingredient_to_supplier: dict[str, tuple[str, float, int]] = {}
    
    for supplier in obs.get("supplier_catalog", []):
        supplier_name = supplier["name"]
        lead_time = supplier.get("lead_time_days", 1)
        
        # Skip suppliers mentioned in alerts (outage, halted, etc.)
        if supplier_name.lower() in alert_text:
            continue
        
        for ingredient, price in supplier.get("ingredients", {}).items():
            if ingredient not in ingredient_to_supplier:
                ingredient_to_supplier[ingredient] = (supplier_name, price, lead_time)
            else:
                _, old_price, old_lead = ingredient_to_supplier[ingredient]
                if price < old_price or (price == old_price and lead_time < old_lead):
                    ingredient_to_supplier[ingredient] = (supplier_name, price, lead_time)
    
    return {ing: data[0] for ing, data in ingredient_to_supplier.items()}


def get_inventory_levels(obs: dict) -> dict[str, float]:
    """Returns {ingredient: total_kg} from inventory."""
    return {item["ingredient"]: item["total_kg"] for item in obs.get("inventory", [])}


def get_pending_ingredients(obs: dict) -> set[str]:
    """Returns set of ingredients already on order."""
    return {o["ingredient"] for o in obs.get("pending_orders", [])}


def get_menu_ingredients(obs: dict) -> dict[str, float]:
    """Returns {ingredient: daily_usage_estimate} for active menu dishes.
    
    Estimates usage based on yesterday's dishes_sold or defaults to 1 order per dish.
    """
    menu_book = {d["name"]: d for d in obs.get("menu_book", [])}
    ss = obs.get("service_summary") or {}
    dishes_sold = ss.get("dishes_sold", {})
    
    usage: dict[str, float] = {}
    
    for dish in obs.get("menu_book", []):
        if not dish.get("is_active"):
            continue
        dish_name = dish["name"]
        # Use yesterday's sales or estimate 5 orders per dish
        qty_sold = dishes_sold.get(dish_name, 5)
        
        for ing in dish.get("ingredients", []):
            ing_name = ing["ingredient"]
            ing_per_dish = ing["quantity_kg"]
            usage[ing_name] = usage.get(ing_name, 0) + (ing_per_dish * qty_sold)
    
    return usage


# =============================================================================
# ORDERING STRATEGIES (Code-based, reliable)
# =============================================================================

def day_one_bulk_order(obs: dict) -> list[dict]:
    """Day 1 (Monday): Order 25kg of every ingredient needed for active menu.
    
    Day 1 orders arrive around Day 3-5 due to lead time and delivery days.
    Need enough starting inventory + these orders to last through first weekend.
    """
    supplier_map = build_supplier_map(obs)
    orders = []
    
    # Get all ingredients from active menu
    needed = set()
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            for ing in dish.get("ingredients", []):
                needed.add(ing["ingredient"])
    
    for ing in sorted(needed):
        if ing in supplier_map:
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier_map[ing], "ingredient": ing, 
                         "quantity_kg": ORDERING["day1_bulk_normal"]}
            })
    
    return orders


def early_week_boost(obs: dict, day: int) -> list[dict]:
    """Days 2-3: Extra orders to ensure first weekend has stock.
    
    The first weekend (Days 6-7) is often a stockout because:
    - Day 1 orders might not arrive until Day 3-5
    - Starting inventory is limited
    """
    if day not in ORDERING["early_week_days"]:
        return []
    
    supplier_map = build_supplier_map(obs)
    inventory = get_inventory_levels(obs)
    
    orders = []
    
    # Get all ingredients from active menu
    needed = set()
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            for ing in dish.get("ingredients", []):
                needed.add(ing["ingredient"])
    
    # Order if below threshold
    for ing in sorted(needed):
        current = inventory.get(ing, 0)
        pending = sum(
            o["quantity_kg"] for o in obs.get("pending_orders", []) 
            if o["ingredient"] == ing
        )
        total = current + pending
        
        if total < ORDERING["early_week_threshold"] and ing in supplier_map:
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier_map[ing], "ingredient": ing, 
                         "quantity_kg": ORDERING["early_week_qty"]}
            })
    
    return orders


def stockout_recovery(obs: dict) -> list[dict]:
    """Emergency orders for dishes that ran out yesterday."""
    ss = obs.get("service_summary") or {}
    stockouts = ss.get("dishes_unavailable_at", {})
    
    if not stockouts:
        return []
    
    supplier_map = build_supplier_map(obs)
    pending = get_pending_ingredients(obs)
    menu_book = {d["name"]: d for d in obs.get("menu_book", [])}
    orders = []
    ordered_ingredients = set()
    
    for dish_name in stockouts:
        if dish_name not in menu_book:
            continue
        for ing in menu_book[dish_name].get("ingredients", []):
            ing_name = ing["ingredient"]
            if ing_name in supplier_map and ing_name not in pending and ing_name not in ordered_ingredients:
                orders.append({
                    "tool": "place_order",
                    "args": {"supplier": supplier_map[ing_name], "ingredient": ing_name, 
                             "quantity_kg": ORDERING["stockout_qty"]}
                })
                ordered_ingredients.add(ing_name)
    
    return orders


def smart_restock(obs: dict, base_threshold: float = None) -> list[dict]:
    """Order ingredients when total available (stock + pending) falls below threshold.
    
    Consider TOTAL available (current + pending), not just current.
    This prevents over-ordering while ensuring we always have stock.
    """
    if base_threshold is None:
        base_threshold = ORDERING["base_threshold"]
    
    supplier_map = build_supplier_map(obs)
    inventory = get_inventory_levels(obs)
    usage = get_menu_ingredients(obs)
    day_of_week = obs.get("day_of_week", "Monday")
    
    orders = []
    
    # Adjust threshold based on day - HIGHER early week for weekend prep
    multiplier = ORDERING["day_multipliers"].get(day_of_week, 1.0)
    effective_threshold = base_threshold * multiplier
    
    # Check all ingredients from menu
    menu_ingredients = set()
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            for ing in dish.get("ingredients", []):
                menu_ingredients.add(ing["ingredient"])
    
    for ing in sorted(menu_ingredients):
        current_stock = inventory.get(ing, 0)
        pending_qty = sum(
            o["quantity_kg"] for o in obs.get("pending_orders", []) 
            if o["ingredient"] == ing
        )
        total_available = current_stock + pending_qty
        
        # Use actual usage if known, else default
        daily_use = usage.get(ing, ORDERING["default_daily_usage"])
        
        # Order if TOTAL available is below threshold
        if total_available < effective_threshold and ing in supplier_map:
            # Order enough for configured days of supply
            order_qty = max(ORDERING["min_order_qty"], 
                          round(daily_use * ORDERING["restock_days_supply"]))
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier_map[ing], "ingredient": ing, "quantity_kg": order_qty}
            })
    
    return orders


def expiring_stock_orders(obs: dict) -> list[dict]:
    """Order replacements for stock expiring in 1-2 days."""
    supplier_map = build_supplier_map(obs)
    pending = get_pending_ingredients(obs)
    orders = []
    
    for item in obs.get("inventory", []):
        ing = item["ingredient"]
        batches = item.get("batches", [])
        
        # Check if significant stock is expiring soon
        expiring_soon = sum(
            b["quantity_kg"] for b in batches 
            if b.get("expires_in_days", 99) <= ORDERING["expiry_days_threshold"]
        )
        remaining = item["total_kg"] - expiring_soon
        
        if remaining < ORDERING["expiry_remaining_threshold"] and ing not in pending and ing in supplier_map:
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier_map[ing], "ingredient": ing, 
                         "quantity_kg": ORDERING["expiry_order_qty"]}
            })
    
    return orders


def weekend_prep_orders(obs: dict) -> list[dict]:
    """Wednesday special: Order extra stock to survive Sat/Sun when no deliveries happen.
    
    This is critical because:
    - Suppliers typically don't deliver Sat/Sun
    - Wednesday orders with 1-day lead arrive Friday
    - Friday stock must last Fri + Sat + Sun (3 high-demand days)
    """
    day_of_week = obs.get("day_of_week", "Monday")
    
    # Only run on Wednesday
    if day_of_week != "Wednesday":
        return []
    
    supplier_map = build_supplier_map(obs)
    inventory = get_inventory_levels(obs)
    pending = get_pending_ingredients(obs)
    usage = get_menu_ingredients(obs)
    
    orders = []
    
    # For weekend, need stock for Fri + Sat + Sun = 3 days of high demand
    for ing, daily_use in usage.items():
        current = inventory.get(ing, 0)
        pending_qty = sum(
            o["quantity_kg"] for o in obs.get("pending_orders", [])
            if o["ingredient"] == ing
        )
        total_available = current + pending_qty
        
        weekend_need = daily_use * ORDERING["weekend_multiplier"]
        
        if total_available < weekend_need and ing in supplier_map and ing not in pending:
            order_qty = max(ORDERING["weekend_min_order"], 
                          round(weekend_need - total_available + ORDERING["weekend_buffer"]))
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier_map[ing], "ingredient": ing, "quantity_kg": order_qty}
            })
    
    return orders


# =============================================================================
# STAFF SCHEDULING (Code-based, deterministic)
# =============================================================================

def get_staff_action(obs: dict) -> dict:
    """Adaptive staff scheduling based on actual performance + day patterns.
    
    Key insight from business sim: avoid over-staffing (diminishing returns).
    Adapt to scenarios like renovation with reduced capacity.
    """
    day_of_week = obs.get("day_of_week", "Monday")
    weather = obs.get("weather_today", "sunny")
    ss = obs.get("service_summary") or {}
    walkout_band = ss.get("walkout_band", "None")
    cash = obs.get("cash", 10000)
    yesterday_covers = ss.get("total_covers", 100)
    alerts = obs.get("alerts", [])
    
    # Check for capacity-reducing alerts (renovation scenario)
    reduced_capacity = any(kw in str(a).lower() for a in alerts 
                          for kw in DETECTION["capacity_keywords"])
    
    # Base levels by day of week
    level = STAFFING["base_levels"].get(day_of_week, 8)
    
    # ADAPTIVE: Scale staff based on yesterday's actual covers
    if yesterday_covers > 0:
        suggested_staff = max(STAFFING["min_staff"], 
                            min(STAFFING["max_staff"], 
                                (yesterday_covers // STAFFING["covers_per_staff"]) + 2))
        # Blend day-based and performance-based
        level = round(level * STAFFING["day_weight"] + suggested_staff * STAFFING["demand_weight"])
    
    # If reduced capacity (renovation), cut staff
    if reduced_capacity:
        level = max(STAFFING["min_staff"], level - STAFFING["reduced_capacity_cut"])
    
    # Weather adjustments
    if weather in ("rainy", "stormy"):
        level -= STAFFING["bad_weather_cut"]
    elif weather == "sunny" and day_of_week in ("Friday", "Saturday"):
        level += STAFFING["sunny_weekend_boost"]
    
    # Walkout recovery
    if walkout_band == "Many":
        level += STAFFING["walkout_many_boost"]
    elif walkout_band == "Some":
        level += STAFFING["walkout_some_boost"]
    
    # Cash safety
    if cash < STAFFING["cash_critical"]:
        level = max(level - STAFFING["critical_cut"], STAFFING["min_staff"])
    elif cash < STAFFING["cash_warning"]:
        level = max(level - STAFFING["warning_cut"], STAFFING["min_staff"] + 1)
    
    # Clamp to valid range
    level = max(STAFFING["min_staff"], min(STAFFING["max_staff"], level))
    
    return {"tool": "set_staff_level", "args": {"level": level}}


# =============================================================================
# DAILY SPECIAL (Code-based, picks well-stocked dish)
# =============================================================================

def get_daily_special(obs: dict) -> dict | None:
    """Pick a daily special from well-stocked dishes."""
    inventory = get_inventory_levels(obs)
    active_menu = obs.get("active_menu", [])
    menu_book = {d["name"]: d for d in obs.get("menu_book", [])}
    
    # Score each dish by how well-stocked its ingredients are
    dish_scores = []
    for dish_name in active_menu:
        if dish_name not in menu_book:
            continue
        dish = menu_book[dish_name]
        ingredients = dish.get("ingredients", [])
        if not ingredients:
            continue
        
        # Calculate minimum ingredient availability (in portions)
        min_portions = float('inf')
        for ing in ingredients:
            ing_name = ing["ingredient"]
            ing_qty = ing["quantity_kg"]
            stock = inventory.get(ing_name, 0)
            if ing_qty > 0:
                portions = stock / ing_qty
                min_portions = min(min_portions, portions)
        
        if min_portions > 10:  # At least 10 portions available
            dish_scores.append((dish_name, min_portions))
    
    if dish_scores:
        # Pick the dish with most portions available
        dish_scores.sort(key=lambda x: -x[1])
        return {"tool": "offer_daily_special", "args": {"dish": dish_scores[0][0]}}
    
    # Fallback: just pick first active dish
    if active_menu:
        return {"tool": "offer_daily_special", "args": {"dish": active_menu[0]}}
    
    return None


# =============================================================================
# PRICING STRATEGY (Code-based for reliability)
# =============================================================================

def get_pricing_actions(obs: dict) -> list[dict]:
    """Dynamic pricing based on day of week and reputation."""
    day_of_week = obs.get("day_of_week", "Monday")
    reputation = obs.get("reputation_band", "Good")
    
    actions = []
    menu_book = {d["name"]: d for d in obs.get("menu_book", [])}
    
    # Only adjust prices if reputation is good enough
    if reputation in PRICING["good_reputations"]:
        # Weekend premium
        if day_of_week in ("Friday", "Saturday"):
            for dish_name, dish in menu_book.items():
                if dish.get("is_active"):
                    base_price = dish["base_price"]
                    new_price = round(base_price * PRICING["weekend_premium"], 2)
                    actions.append({
                        "tool": "set_price",
                        "args": {"dish": dish_name, "price": new_price}
                    })
    elif reputation in PRICING["poor_reputations"]:
        # Discount to attract customers when reputation is low
        for dish_name, dish in menu_book.items():
            if dish.get("is_active"):
                base_price = dish["base_price"]
                new_price = round(base_price * PRICING["poor_rep_discount"], 2)
                actions.append({
                    "tool": "set_price",
                    "args": {"dish": dish_name, "price": new_price}
                })
    
    return actions


# =============================================================================
# MARKETING STRATEGY (Code-based)
# =============================================================================

def get_marketing_action(obs: dict) -> dict | None:
    """Strategic marketing spend based on conditions."""
    day_of_week = obs.get("day_of_week", "Monday")
    cash = obs.get("cash", 0)
    reputation = obs.get("reputation_band", "Good")
    customer_trend = obs.get("customer_trend", "Stable")
    
    # Don't spend if cash is tight
    if cash < MARKETING["min_cash"]:
        return None
    
    # Boost marketing on slower days or when trend is declining
    if customer_trend == "Declining":
        return {"tool": "set_marketing_spend", "args": {"amount": MARKETING["declining_amount"]}}
    
    if day_of_week in MARKETING["slow_days"] and reputation not in PRICING["poor_reputations"]:
        return {"tool": "set_marketing_spend", "args": {"amount": MARKETING["slow_day_amount"]}}
    
    return None


# =============================================================================
# NOTES SYSTEM - Persist state across days
# =============================================================================

def parse_notes(notes: str) -> dict:
    """Parse notes string into key-value pairs. Format: KEY:VALUE;KEY2:VALUE2"""
    result = {}
    if not notes:
        return result
    for part in notes.split(";"):
        if ":" in part:
            key, val = part.split(":", 1)
            result[key.strip()] = val.strip()
    return result


def build_notes(data: dict) -> str:
    """Build notes string from key-value pairs."""
    return ";".join(f"{k}:{v}" for k, v in data.items())


# =============================================================================
# LLM STRATEGIC ADVISOR
# =============================================================================

def calculate_bankruptcy_risk(obs: dict) -> float:
    """Calculate probability of going bankrupt based on current state."""
    cash = obs.get("cash", 15000)
    
    # Estimate daily burn rate
    staff_level = obs.get("staff_level", 8)
    daily_cost = (staff_level * 120) + 300  # staff + fixed
    
    # Estimate daily revenue based on reputation and trend
    reputation = obs.get("reputation_band", "Good")
    rep_factor = {"Poor": 0.6, "Fair": 0.8, "Good": 1.0, "Very Good": 1.1, "Excellent": 1.2}.get(reputation, 1.0)
    
    ss = obs.get("service_summary") or {}
    yesterday_revenue = obs.get("yesterday_revenue", 2000)
    expected_revenue = yesterday_revenue * rep_factor
    
    # Calculate days until bankruptcy
    net_daily = expected_revenue - daily_cost
    if net_daily >= 0:
        return 0.0
    
    days_until_zero = cash / abs(net_daily) if net_daily < 0 else float('inf')
    days_remaining = obs.get("days_remaining", 30)
    
    if days_until_zero > days_remaining:
        return 0.0
    elif days_until_zero < 3:
        return 0.9
    elif days_until_zero < 7:
        return 0.5
    elif days_until_zero < 14:
        return 0.2
    return 0.05


def calculate_walkout_risk(obs: dict) -> str:
    """Assess walkout risk based on current conditions."""
    ss = obs.get("service_summary") or {}
    
    # Check yesterday's walkouts
    walkout_band = ss.get("walkout_band", "None")
    if walkout_band == "Many":
        return "HIGH"
    if walkout_band == "Some":
        return "MEDIUM"
    
    # Check for stockouts
    stockouts = ss.get("dishes_unavailable_at", {})
    if len(stockouts) > 2:
        return "HIGH"
    if len(stockouts) > 0:
        return "MEDIUM"
    
    # Check wait times
    peak_wait = ss.get("peak_wait_minutes", 0)
    if peak_wait > 20:
        return "HIGH"
    if peak_wait > 12:
        return "MEDIUM"
    
    return "LOW"


def get_consumption_rates(obs: dict) -> dict[str, float]:
    """Calculate ingredient consumption rates from yesterday's sales."""
    ss = obs.get("service_summary") or {}
    dishes_sold = ss.get("dishes_sold", {})
    menu_book = {d["name"]: d for d in obs.get("menu_book", [])}
    
    consumption = {}
    for dish_name, qty in dishes_sold.items():
        if dish_name in menu_book:
            for ing in menu_book[dish_name].get("ingredients", []):
                ing_name = ing["ingredient"]
                used = ing["quantity_kg"] * qty
                consumption[ing_name] = consumption.get(ing_name, 0) + used
    
    return consumption


def get_supplier_reliability(obs: dict) -> dict[str, float]:
    """Calculate supplier reliability scores from delivery history."""
    history = obs.get("delivery_history", [])
    supplier_stats = {}
    
    for delivery in history:
        supplier = delivery.get("supplier", "")
        ordered = delivery.get("ordered_kg", 0)
        delivered = delivery.get("delivered_kg", 0)
        on_time = delivery.get("on_time", True)
        
        if supplier not in supplier_stats:
            supplier_stats[supplier] = {"total": 0, "delivered": 0, "on_time": 0, "count": 0}
        
        supplier_stats[supplier]["total"] += ordered
        supplier_stats[supplier]["delivered"] += delivered
        supplier_stats[supplier]["count"] += 1
        if on_time:
            supplier_stats[supplier]["on_time"] += 1
    
    reliability = {}
    for supplier, stats in supplier_stats.items():
        if stats["count"] > 0:
            fill_rate = stats["delivered"] / stats["total"] if stats["total"] > 0 else 1.0
            on_time_rate = stats["on_time"] / stats["count"]
            reliability[supplier] = round((fill_rate * 0.7 + on_time_rate * 0.3), 2)
    
    return reliability


def summarize_observation(obs: dict, day: int) -> str:
    """Create a comprehensive summary for the LLM to analyze."""
    ss = obs.get("service_summary") or {}
    
    # Key metrics
    cash = obs.get("cash", 0)
    day_of_week = obs.get("day_of_week", "Monday")
    days_remaining = obs.get("days_remaining", 30 - day)
    reputation = obs.get("reputation_band", "Good")
    customer_trend = obs.get("customer_trend", "Stable")
    weather = obs.get("weather_today", "sunny")
    weather_forecast = obs.get("weather_forecast", [])
    
    # Yesterday's performance
    covers = ss.get("total_covers", 0)
    revenue = obs.get("yesterday_revenue", 0)
    costs = obs.get("yesterday_total_costs", 0)
    cost_breakdown = obs.get("cost_breakdown", {})
    walkout_band = ss.get("walkout_band", "None")
    stockouts = list(ss.get("dishes_unavailable_at", {}).keys())
    avg_wait = ss.get("avg_wait_minutes", 0)
    peak_wait = ss.get("peak_wait_minutes", 0)
    
    # Calculate profit/loss
    profit = revenue - costs if costs > 0 else 0
    
    # Risk assessments
    bankruptcy_risk = calculate_bankruptcy_risk(obs)
    walkout_risk = calculate_walkout_risk(obs)
    
    # Alerts (critical for scenario detection)
    alerts = obs.get("alerts", [])
    
    # Inventory analysis
    inventory = obs.get("inventory", [])
    low_stock = []
    critical_stock = []
    expiring = []
    total_inventory_value = 0
    
    for item in inventory:
        ing = item["ingredient"]
        total_kg = item["total_kg"]
        
        if total_kg < 3:
            critical_stock.append(f"{ing}({total_kg:.1f}kg)")
        elif total_kg < 8:
            low_stock.append(f"{ing}({total_kg:.1f}kg)")
        
        for b in item.get("batches", []):
            if b.get("expires_in_days", 99) <= 2:
                expiring.append(f"{ing}({b['quantity_kg']:.1f}kg in {b['expires_in_days']}d)")
    
    # Consumption rates
    consumption = get_consumption_rates(obs)
    top_consumed = sorted(consumption.items(), key=lambda x: -x[1])[:5]
    
    # Supplier reliability
    reliability = get_supplier_reliability(obs)
    
    # Pending orders
    pending = obs.get("pending_orders", [])
    pending_summary = []
    for order in pending:
        pending_summary.append(f"{order['ingredient']}:{order['quantity_kg']}kg→Day{order['delivery_day']}")
    
    # Staff info
    staff_level = obs.get("staff_level", 8)
    staff_cost = staff_level * 120
    
    # Notes from previous day
    notes = obs.get("notes", "")
    
    # Dishes sold summary
    dishes_sold = ss.get("dishes_sold", {})
    top_dishes = sorted(dishes_sold.items(), key=lambda x: -x[1])[:5]
    
    summary = f"""═══════════════════════════════════════════════════════════════
DAY {day}/30 ({day_of_week}) — {days_remaining} days remaining
═══════════════════════════════════════════════════════════════

FINANCIAL STATUS:
  Cash: €{cash:,.0f}
  Yesterday: Revenue €{revenue:,.0f} - Costs €{costs:,.0f} = Profit €{profit:,.0f}
  Cost breakdown: Staff €{cost_breakdown.get('staff', 0):.0f}, Fixed €{cost_breakdown.get('fixed', 0):.0f}, Waste €{cost_breakdown.get('waste', 0):.1f}
  
RISK ASSESSMENT:
  Bankruptcy Risk: {bankruptcy_risk:.0%} {'⚠️ DANGER' if bankruptcy_risk > 0.3 else '✓ OK'}
  Walkout Risk: {walkout_risk}
  
REPUTATION & DEMAND:
  Reputation: {reputation}
  Customer Trend: {customer_trend}
  Weather: {weather} → Forecast: {', '.join(weather_forecast[:3]) if weather_forecast else 'N/A'}

YESTERDAY'S SERVICE:
  Covers: {covers}
  Walkouts: {walkout_band}
  Wait times: avg {avg_wait:.1f}min, peak {peak_wait:.1f}min
  Top dishes: {', '.join(f'{d}({q})' for d, q in top_dishes) if top_dishes else 'N/A'}
  Stockouts: {', '.join(stockouts) if stockouts else 'None'}

STAFFING:
  Current level: {staff_level} (cost: €{staff_cost}/day)
  Covers per staff yesterday: {covers / staff_level:.1f} if staff_level > 0 else 'N/A'

INVENTORY STATUS:
  CRITICAL (<3kg): {', '.join(critical_stock) if critical_stock else 'None'}
  LOW (<8kg): {', '.join(low_stock) if low_stock else 'None'}
  EXPIRING (≤2 days): {', '.join(expiring) if expiring else 'None'}
  
CONSUMPTION (yesterday):
  {', '.join(f'{ing}:{qty:.1f}kg' for ing, qty in top_consumed) if top_consumed else 'N/A'}

PENDING ORDERS:
  {', '.join(pending_summary) if pending_summary else 'None'}

SUPPLIER RELIABILITY:
  {', '.join(f'{s}:{r:.0%}' for s, r in reliability.items()) if reliability else 'No history yet'}

ALERTS: {'; '.join(str(a) for a in alerts) if alerts else 'None'}

PREVIOUS STRATEGIC NOTES: {notes if notes else 'None'}
═══════════════════════════════════════════════════════════════"""
    
    return summary


def get_llm_advice(obs: dict, day: int) -> dict | None:
    """Get strategic advice from LLM. Returns None on error (fallback to rules)."""
    if not USE_LLM:
        return None
    
    try:
        summary = summarize_observation(obs, day)
        
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": STRATEGY_PROMPT},
                {"role": "user", "content": summary},
            ],
            temperature=0.2,  # Lower for more consistent decisions
            max_tokens=MAX_TOKENS,
            timeout=LLM_TIMEOUT,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up markdown if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        # Handle potential JSON in markdown json block
        if content.startswith("json"):
            content = content[4:].strip()
        
        advice = json.loads(content)
        
        # Validate and clamp all values with safe defaults
        advice["price_multiplier"] = max(0.8, min(1.2, float(advice.get("price_multiplier", 1.0))))
        advice["marketing_spend"] = max(0, min(500, int(advice.get("marketing_spend", 0))))
        advice["staff_adjustment"] = max(-3, min(3, int(advice.get("staff_adjustment", 0))))
        advice["run_happy_hour"] = bool(advice.get("run_happy_hour", False))
        advice["order_multiplier"] = max(0.5, min(2.0, float(advice.get("order_multiplier", 1.0))))
        advice["confidence"] = max(0.0, min(1.0, float(advice.get("confidence", 0.5))))
        advice["bankruptcy_risk"] = max(0.0, min(1.0, float(advice.get("bankruptcy_risk", 0.0))))
        
        return advice
        
    except Exception as e:
        print(f"  LLM error on day {day}: {e}")
        return None


# =============================================================================
# RULE-BASED FALLBACK SYSTEM
# =============================================================================

def detect_scenario_rules(obs: dict, day: int) -> str:
    """Detect scenario using pure rule-based logic with config parameters."""
    cash = obs.get("cash", 15000)
    alerts = obs.get("alerts", [])
    alert_text = " ".join(str(a).lower() for a in alerts)
    ss = obs.get("service_summary") or {}
    covers = ss.get("total_covers", 0)
    notes = parse_notes(obs.get("notes", ""))
    
    # Check for renovation (Day 1 cash = 17000 + renovation alert)
    if day == 1 and cash >= RENOVATION["starting_cash"] - 500:
        if any(kw in alert_text for kw in DETECTION["capacity_keywords"]):
            return "renovation"
    
    # Check persisted scenario from notes
    if "SCENARIO" in notes:
        return notes["SCENARIO"]
    
    # Check for supply crisis (supplier outage alert)
    if any(kw in alert_text for kw in SUPPLY_CRISIS["crisis_keywords"]):
        return "supply_crisis"
    
    # Check for tourist season (surge detection days 9-14)
    surge_start = TOURIST_SEASON["surge_start_day"]
    surge_end = TOURIST_SEASON["surge_end_day"]
    surge_threshold = TOURIST_SEASON["surge_covers_threshold"]
    
    if surge_start <= day <= surge_end and covers > surge_threshold:
        return "tourist_season"
    
    # Check historical pattern for tourist season
    if "TOURIST" in notes or notes.get("SCENARIO") == "tourist_season":
        return "tourist_season"
    
    return "baseline"


def rule_based_pricing(obs: dict, scenario: str) -> float:
    """Get price multiplier using rules and config parameters."""
    day_of_week = obs.get("day_of_week", "Monday")
    reputation = obs.get("reputation_band", "Good")
    day = obs.get("day", 1)
    
    # Base multiplier
    multiplier = 1.0
    
    # Weekend premium if reputation allows (use PRICING config)
    if day_of_week in ("Friday", "Saturday"):
        if reputation in PRICING["good_reputations"]:
            multiplier = PRICING["weekend_premium"]
        elif reputation == "Fair":
            multiplier = 1.0
        else:  # Poor
            multiplier = PRICING["poor_rep_discount"]
    
    # Recovery discount for poor reputation
    if reputation == "Poor":
        multiplier = min(multiplier, 0.88)
    elif reputation == "Fair":
        multiplier = min(multiplier, 0.95)
    
    # Scenario adjustments using config
    if scenario == "renovation":
        multiplier *= 0.95  # Slight discount during renovation
    elif scenario == "tourist_season":
        surge_start = TOURIST_SEASON["surge_start_day"]
        surge_end = TOURIST_SEASON["surge_end_day"]
        surge_threshold = TOURIST_SEASON["surge_covers_threshold"]
        
        ss = obs.get("service_summary") or {}
        covers = ss.get("total_covers", 0)
        
        if (surge_start <= day <= surge_end) or covers > surge_threshold:
            multiplier = min(1.18, multiplier * TOURIST_SEASON["surge_price_premium"])
    
    return max(0.8, min(1.2, multiplier))


def rule_based_marketing(obs: dict, scenario: str) -> int:
    """Get marketing spend using rules only."""
    cash = obs.get("cash", 15000)
    day_of_week = obs.get("day_of_week", "Monday")
    customer_trend = obs.get("customer_trend", "Stable")
    reputation = obs.get("reputation_band", "Good")
    
    # No marketing if cash is low
    if cash < 8000:
        return 0
    
    # No marketing on weekends (already busy)
    if day_of_week in ("Friday", "Saturday", "Sunday"):
        return 0
    
    # No marketing during tourist season surge
    if scenario == "tourist_season":
        return 0
    
    # Boost marketing if trend is declining
    if customer_trend == "Declining":
        return 200 if cash > 10000 else 100
    
    # Light marketing on slow days
    if day_of_week in ("Monday", "Tuesday"):
        if reputation in ("Poor", "Fair"):
            return 150 if cash > 10000 else 0
        return 100 if cash > 12000 else 0
    
    return 0


def rule_based_staff_adjustment(obs: dict, scenario: str) -> int:
    """Get staff adjustment using rules and config parameters."""
    ss = obs.get("service_summary") or {}
    walkout_band = ss.get("walkout_band", "None")
    covers = ss.get("total_covers", 100)
    cash = obs.get("cash", 15000)
    day = obs.get("day", 1)
    
    adjustment = 0
    
    # Increase if walkouts detected
    if walkout_band == "Many":
        adjustment += 2
    elif walkout_band == "Some":
        adjustment += 1
    
    # Scenario-specific adjustments using config
    if scenario == "renovation":
        adjustment -= RENOVATION["staff_reduction"]
    elif scenario == "tourist_season":
        surge_start = TOURIST_SEASON["surge_start_day"]
        surge_end = TOURIST_SEASON["surge_end_day"]
        surge_threshold = TOURIST_SEASON["surge_covers_threshold"]
        
        if surge_start <= day <= surge_end or covers > surge_threshold:
            adjustment += TOURIST_SEASON["surge_staff_boost"]
        elif day > surge_end:
            adjustment -= TOURIST_SEASON["postsurge_staff_reduction"]
    
    # Cash safety using RISK config
    if cash < RISK["bankruptcy_danger"]:
        adjustment -= 2
    elif cash < RISK["bankruptcy_caution"]:
        adjustment -= 1
    
    return max(-3, min(3, adjustment))


def rule_based_happy_hour(obs: dict, scenario: str) -> bool:
    """Decide on happy hour using rules only."""
    cash = obs.get("cash", 15000)
    day_of_week = obs.get("day_of_week", "Monday")
    customer_trend = obs.get("customer_trend", "Stable")
    
    # No happy hour if cash is low
    if cash < 10000:
        return False
    
    # No happy hour on already-busy days
    if day_of_week in ("Friday", "Saturday"):
        return False
    
    # Happy hour on slow days with declining trend
    if day_of_week in ("Monday", "Tuesday", "Wednesday") and customer_trend == "Declining":
        return True
    
    return False


def rule_based_order_multiplier(obs: dict, scenario: str, day: int) -> float:
    """Get order multiplier using rules and config parameters."""
    day_of_week = obs.get("day_of_week", "Monday")
    ss = obs.get("service_summary") or {}
    stockouts = ss.get("dishes_unavailable_at", {})
    
    multiplier = 1.0
    
    # Increase if stockouts detected
    if len(stockouts) > 0:
        multiplier *= RISK["stockout_buffer_multiplier"]
    
    # Scenario adjustments using config
    if scenario == "renovation":
        multiplier *= RENOVATION["order_multiplier"]
    elif scenario == "tourist_season":
        prestock_start = TOURIST_SEASON["prestock_start_day"]
        prestock_end = TOURIST_SEASON["prestock_end_day"]
        surge_start = TOURIST_SEASON["surge_start_day"]
        surge_end = TOURIST_SEASON["surge_end_day"]
        
        if prestock_start <= day <= prestock_end:
            multiplier *= TOURIST_SEASON["prestock_multiplier"]
        elif surge_start <= day <= surge_end:
            multiplier *= TOURIST_SEASON["surge_order_multiplier"]
        elif day > surge_end:
            multiplier *= TOURIST_SEASON["postsurge_order_multiplier"]
    elif scenario == "supply_crisis":
        multiplier *= SUPPLY_CRISIS["emergency_order_multiplier"]
    
    # Wednesday prep for weekend
    if day_of_week == "Wednesday":
        multiplier *= 1.2
    
    return max(0.5, min(2.0, multiplier))


def generate_rule_based_advice(obs: dict, day: int) -> dict:
    """Generate complete advice using only rule-based logic (no LLM)."""
    scenario = detect_scenario_rules(obs, day)
    
    return {
        "scenario_detected": scenario,
        "confidence": 0.7,  # Lower confidence for rule-based
        "scenario_reasoning": f"Rule-based detection: {scenario}",
        "price_multiplier": rule_based_pricing(obs, scenario),
        "price_reasoning": "Rule-based pricing",
        "marketing_spend": rule_based_marketing(obs, scenario),
        "marketing_reasoning": "Rule-based marketing",
        "staff_adjustment": rule_based_staff_adjustment(obs, scenario),
        "staff_reasoning": "Rule-based staffing",
        "run_happy_hour": rule_based_happy_hour(obs, scenario),
        "happy_hour_reasoning": "Rule-based decision",
        "order_multiplier": rule_based_order_multiplier(obs, scenario, day),
        "order_reasoning": "Rule-based ordering",
        "risk_assessment": "medium",
        "bankruptcy_risk": calculate_bankruptcy_risk(obs),
        "strategic_notes": f"FALLBACK|{scenario}|day{day}",
        "_is_fallback": True,
    }


# =============================================================================
# SAFETY VALIDATION LAYER (Bankruptcy Protection)
# =============================================================================

def validate_and_adjust_advice(advice: dict, obs: dict, day: int) -> dict:
    """Validate LLM/rule-based advice against safety rules.
    
    Overrides dangerous decisions to protect against bankruptcy.
    Uses RISK config for thresholds.
    """
    cash = obs.get("cash", 15000)
    days_remaining = obs.get("days_remaining", 30 - day)
    reputation = obs.get("reputation_band", "Good")
    
    validated = dict(advice)  # Copy to avoid mutation
    
    # ==========================================================================
    # BANKRUPTCY PROTECTION (Highest priority) - Using RISK config
    # ==========================================================================
    
    if cash < RISK["bankruptcy_critical"]:
        # CRITICAL: Emergency mode
        validated["marketing_spend"] = 0
        validated["run_happy_hour"] = False
        validated["staff_adjustment"] = max(-3, validated.get("staff_adjustment", 0) - 2)
        validated["order_multiplier"] = max(0.5, validated.get("order_multiplier", 1.0) * 0.5)
        validated["_safety_override"] = "CRITICAL_CASH"
        
    elif cash < RISK["bankruptcy_danger"]:
        # DANGER: Conservative mode
        validated["marketing_spend"] = 0
        validated["run_happy_hour"] = False
        validated["staff_adjustment"] = max(-2, validated.get("staff_adjustment", 0) - 1)
        validated["order_multiplier"] = max(0.6, validated.get("order_multiplier", 1.0) * 0.7)
        validated["_safety_override"] = "LOW_CASH"
        
    elif cash < RISK["bankruptcy_caution"]:
        # CAUTION: Reduced spending
        validated["marketing_spend"] = min(100, validated.get("marketing_spend", 0))
        validated["order_multiplier"] = min(1.0, validated.get("order_multiplier", 1.0))
        
    # ==========================================================================
    # REPUTATION PROTECTION (Quadratic penalty avoidance)
    # ==========================================================================
    
    if reputation in ("Poor", "Fair"):
        # Don't raise prices when reputation is low
        if validated.get("price_multiplier", 1.0) > 1.0:
            validated["price_multiplier"] = min(1.0, validated.get("price_multiplier", 1.0))
        
        # Boost staff slightly to reduce walkouts
        validated["staff_adjustment"] = max(validated.get("staff_adjustment", 0), 1)
        
    # ==========================================================================
    # END-GAME PROTECTION
    # ==========================================================================
    
    if days_remaining <= 5:
        # Final stretch - prioritize survival
        if cash < 10000:
            validated["marketing_spend"] = 0
            validated["order_multiplier"] = max(0.5, validated.get("order_multiplier", 1.0) * 0.7)
    
    # ==========================================================================
    # PROBABILITY-BASED ADJUSTMENTS
    # ==========================================================================
    
    # Calculate expected cost of advice
    expected_cost = calculate_expected_daily_cost(validated, obs)
    
    # If expected cost would put us at risk, scale back
    if cash - expected_cost < 3000:
        # Scale back to ensure survival
        scale_factor = max(0.3, (cash - 3000) / expected_cost) if expected_cost > 0 else 1.0
        validated["marketing_spend"] = int(validated.get("marketing_spend", 0) * scale_factor)
        validated["order_multiplier"] = validated.get("order_multiplier", 1.0) * max(0.5, scale_factor)
    
    return validated


def calculate_expected_daily_cost(advice: dict, obs: dict) -> float:
    """Calculate expected total daily cost based on advice."""
    staff_level = obs.get("staff_level", 8)
    adjustment = advice.get("staff_adjustment", 0)
    new_staff = max(STAFFING["min_staff"], min(STAFFING["max_staff"], staff_level + adjustment))
    
    staff_cost = new_staff * 120
    fixed_cost = 300
    marketing_cost = advice.get("marketing_spend", 0)
    
    # Estimate order costs (rough approximation)
    order_multiplier = advice.get("order_multiplier", 1.0)
    base_order_cost = 500  # Average daily ingredient cost
    order_cost = base_order_cost * order_multiplier
    
    return staff_cost + fixed_cost + marketing_cost + order_cost


# =============================================================================
# PROBABILITY-BASED DECISION HELPERS
# =============================================================================

def calculate_stockout_probability(obs: dict, ingredient: str, days_ahead: int = 2) -> float:
    """Calculate probability of stockout for an ingredient in next N days."""
    inventory = {item["ingredient"]: item["total_kg"] for item in obs.get("inventory", [])}
    current_stock = inventory.get(ingredient, 0)
    
    # Get pending orders for this ingredient
    pending = sum(
        o["quantity_kg"] for o in obs.get("pending_orders", [])
        if o["ingredient"] == ingredient and o.get("delivery_day", 99) <= obs.get("day", 1) + days_ahead
    )
    
    total_available = current_stock + pending
    
    # Estimate daily usage
    consumption = get_consumption_rates(obs)
    daily_usage = consumption.get(ingredient, 2.0)  # Default 2kg/day
    
    expected_usage = daily_usage * days_ahead * 1.3  # 30% buffer
    
    if total_available >= expected_usage:
        return 0.0
    elif total_available <= 0:
        return 1.0
    else:
        return min(1.0, 1.0 - (total_available / expected_usage))


def calculate_expected_walkouts(obs: dict, staff_level: int) -> float:
    """Estimate expected walkouts based on staff level and demand."""
    ss = obs.get("service_summary") or {}
    yesterday_covers = ss.get("total_covers", 100)
    peak_wait = ss.get("peak_wait_minutes", 5)
    
    # Estimate capacity
    capacity_per_staff = 15
    capacity = staff_level * capacity_per_staff
    
    # If demand exceeds capacity, expect walkouts
    expected_demand = yesterday_covers * 1.1  # Assume slight growth
    
    if expected_demand <= capacity:
        base_walkouts = 0
    else:
        overflow = expected_demand - capacity
        base_walkouts = overflow * 0.5  # 50% of overflow walks out
    
    # Adjust for wait times
    if peak_wait > 15:
        base_walkouts *= 1.5
    elif peak_wait > 20:
        base_walkouts *= 2.0
    
    return base_walkouts


def calculate_optimal_staff_ev(obs: dict) -> int:
    """Calculate optimal staff level using expected value."""
    ss = obs.get("service_summary") or {}
    yesterday_covers = ss.get("total_covers", 100)
    day_of_week = obs.get("day_of_week", "Monday")
    
    # Expected demand by day
    demand_multipliers = {
        "Sunday": 0.0, "Monday": 0.9, "Tuesday": 0.95,
        "Wednesday": 1.1, "Thursday": 1.2,
        "Friday": 1.6, "Saturday": 1.8
    }
    
    expected_covers = yesterday_covers * demand_multipliers.get(day_of_week, 1.0)
    
    best_staff = 5
    best_ev = float('-inf')
    
    for staff in range(STAFFING["min_staff"], STAFFING["max_staff"] + 1):
        # Revenue potential
        capacity = staff * 15
        served = min(expected_covers, capacity)
        revenue = served * 20  # ~€20 per cover
        
        # Costs
        staff_cost = staff * 120
        
        # Walkout penalty
        walkouts = max(0, expected_covers - capacity)
        walkout_penalty = walkouts * 20  # €20 per walkout (direct + review damage)
        
        # Expected value
        ev = revenue - staff_cost - walkout_penalty
        
        if ev > best_ev:
            best_ev = ev
            best_staff = staff
    
    return best_staff


def apply_llm_pricing(obs: dict, advice: dict) -> list[dict]:
    """Apply LLM-recommended price multiplier to all active dishes."""
    multiplier = advice.get("price_multiplier", 1.0)
    
    # Skip if multiplier is neutral
    if 0.98 <= multiplier <= 1.02:
        return []
    
    actions = []
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            base_price = dish["base_price"]
            new_price = round(base_price * multiplier, 2)
            # Ensure within allowed range
            min_price = base_price * 0.8
            max_price = base_price * 1.2
            new_price = max(min_price, min(max_price, new_price))
            actions.append({
                "tool": "set_price",
                "args": {"dish": dish["name"], "price": new_price}
            })
    
    return actions


# =============================================================================
# MAIN STRATEGY FUNCTION
# =============================================================================

def is_reduced_capacity(observation: dict, day: int) -> tuple[bool, int]:
    """Detect if we're in a reduced capacity scenario (renovation, etc.).
    
    Returns: (is_reduced, remaining_days)
    """
    alerts = observation.get("alerts", [])
    alert_text = " ".join(str(a).lower() for a in alerts)
    notes = parse_notes(observation.get("notes", ""))
    
    # Check for NEW renovation alert (Day 1)
    if any(kw in alert_text for kw in DETECTION["capacity_keywords"]):
        # "two weeks" = 14 days of renovation
        if "two week" in alert_text or "2 week" in alert_text:
            return True, 14
        return True, 7  # Default to 1 week if unclear
    
    # Check persisted renovation state from notes
    if "RENO" in notes:
        remaining = int(notes.get("RENO", 0))
        if remaining > 0:
            return True, remaining
    
    # Fallback: detect by low covers on busy days
    ss = observation.get("service_summary") or {}
    covers = ss.get("total_covers", 100)
    dow = observation.get("day_of_week", "Monday")
    
    if dow in ("Friday", "Saturday") and covers < DETECTION["low_covers_threshold"] and day > DETECTION["low_covers_after_day"]:
        return True, 7  # Assume 1 week remaining
    
    return False, 0


def strategy(observation: dict, day: int) -> list[dict]:
    """Hybrid strategy: Code for reliability, LLM for strategic judgment.
    
    Architecture:
    1. Try LLM for strategic advice
    2. Fall back to rule-based system if LLM fails
    3. Validate all decisions against safety rules
    4. Code handles supplier/inventory (never let LLM touch supplier names)
    
    Code handles (must be exact):
    - Supplier selection and ordering
    - Inventory management
    - Basic staffing baseline
    
    LLM/Fallback handles (judgment calls):
    - Scenario detection and adaptation
    - Pricing optimization
    - Marketing decisions
    - Staff adjustments
    - Happy hour timing
    """
    actions = []
    cash = observation.get("cash", 15000)
    day_of_week = observation.get("day_of_week", "Monday")
    reputation = observation.get("reputation_band", "Good")
    
    # Get LLM strategic advice with fallback to rule-based system
    llm_advice = get_llm_advice(observation, day)
    
    if llm_advice is None:
        # LLM failed - use comprehensive rule-based fallback
        print(f"  Day {day}: Using rule-based fallback")
        llm_advice = generate_rule_based_advice(observation, day)
    
    # Apply safety validation to override dangerous decisions
    llm_advice = validate_and_adjust_advice(llm_advice, observation, day)
    
    # Extract scenario info
    scenario = llm_advice.get("scenario_detected", "baseline")
    order_multiplier = llm_advice.get("order_multiplier", 1.0)
    
    # Detect reduced capacity scenarios
    reduced = scenario == "renovation"
    reno_remaining = 14 if reduced else 0
    
    # Parse existing notes and update state
    notes_data = parse_notes(observation.get("notes", ""))
    
    if reduced:
        # Track renovation countdown
        existing_reno = int(notes_data.get("RENO", reno_remaining))
        notes_data["RENO"] = str(max(0, existing_reno - 1))
    elif "RENO" in notes_data and int(notes_data.get("RENO", 0)) <= 0:
        del notes_data["RENO"]
    
    # Track scenario and strategic notes
    notes_data["SCENARIO"] = scenario
    if llm_advice.get("strategic_notes"):
        notes_data["LLM"] = str(llm_advice["strategic_notes"])[:200]
    
    # Track tourist season detection
    if scenario == "tourist_season":
        notes_data["TOURIST"] = "1"
    
    # ==========================================================================
    # 1. ORDERING (Code-based with LLM multiplier adjustment)
    # ==========================================================================
    order_actions = []
    
    if day == 1:
        if reduced:
            order_actions = day_one_bulk_order_reduced(observation)
        else:
            order_actions = day_one_bulk_order(observation)
    else:
        if not reduced:
            order_actions.extend(early_week_boost(observation, day))
        
        order_actions.extend(stockout_recovery(observation))
        
        # Use order_multiplier to adjust restock threshold
        adjusted_threshold = ORDERING["base_threshold"] * order_multiplier
        order_actions.extend(smart_restock(observation, adjusted_threshold))
        order_actions.extend(expiring_stock_orders(observation))
        
        # Wednesday weekend prep (boosted for tourist season)
        if day_of_week == "Wednesday":
            order_actions.extend(weekend_prep_orders(observation))
    
    # Apply order multiplier to quantities (for surge preparation)
    if order_multiplier != 1.0 and order_multiplier > 0:
        for action in order_actions:
            if action.get("tool") == "place_order":
                orig_qty = action["args"]["quantity_kg"]
                action["args"]["quantity_kg"] = round(orig_qty * order_multiplier)
    
    actions.extend(order_actions)
    
    # ==========================================================================
    # 2. STAFFING (Code baseline + adjustment from advice)
    # ==========================================================================
    staff_action = get_staff_action(observation)
    base_level = staff_action["args"]["level"]
    
    adjustment = llm_advice.get("staff_adjustment", 0)
    new_level = max(STAFFING["min_staff"], min(STAFFING["max_staff"], base_level + adjustment))
    staff_action["args"]["level"] = new_level
    
    actions.append(staff_action)
    
    # ==========================================================================
    # 3. DAILY SPECIAL (Code-based, picks well-stocked dish)
    # ==========================================================================
    special = get_daily_special(observation)
    if special:
        actions.append(special)
    
    # ==========================================================================
    # 4. PRICING (From advice)
    # ==========================================================================
    pricing_actions = apply_llm_pricing(observation, llm_advice)
    actions.extend(pricing_actions)
    
    # ==========================================================================
    # 5. MARKETING (From advice with cash safety check)
    # ==========================================================================
    marketing_amount = llm_advice.get("marketing_spend", 0)
    if marketing_amount > 0 and cash > MARKETING["min_cash"]:
        # Additional safety: reduce marketing if cash is tight
        if cash < 8000:
            marketing_amount = min(marketing_amount, 100)
        actions.append({"tool": "set_marketing_spend", "args": {"amount": marketing_amount}})
    
    # ==========================================================================
    # 6. HAPPY HOUR (From advice with cash safety check)
    # ==========================================================================
    if llm_advice.get("run_happy_hour") and cash > MARKETING["min_cash"]:
        actions.append({"tool": "run_happy_hour", "args": {}})
    
    # ==========================================================================
    # CLEANUP & DEDUPLICATION
    # ==========================================================================
    actions = [a for a in actions if a is not None]
    
    # Deduplicate orders for same ingredient
    seen_ingredients = set()
    final_actions = []
    for action in actions:
        if action.get("tool") == "place_order":
            ing = action["args"]["ingredient"]
            if ing not in seen_ingredients:
                seen_ingredients.add(ing)
                final_actions.append(action)
        else:
            final_actions.append(action)
    
    # Save notes for next day
    if notes_data:
        final_actions.append({"tool": "save_notes", "args": {"text": build_notes(notes_data)}})
    
    return final_actions


def day_one_bulk_order_reduced(obs: dict) -> list[dict]:
    """Day 1 for reduced capacity: Order smaller quantities."""
    supplier_map = build_supplier_map(obs)
    orders = []
    
    needed = set()
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            for ing in dish.get("ingredients", []):
                needed.add(ing["ingredient"])
    
    for ing in sorted(needed):
        if ing in supplier_map:
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier_map[ing], "ingredient": ing, 
                         "quantity_kg": ORDERING["day1_bulk_reduced"]}
            })
    
    return orders


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    scenario = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    print(f"Running Hybrid Agent (seed={seed}, scenario={scenario})")
    print("=" * 60)
    result = run_game(strategy, team_name="The_Coding_Giraffes", 
                      scenario=scenario, seed=seed)
