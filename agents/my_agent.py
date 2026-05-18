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
import math
import os
import re
import sys

import openai

from agents.runner import run_game
from agents.config import (
    ORDERING, STAFFING, PRICING, MARKETING, DETECTION,
    TOURIST_SEASON, SUPPLY_CRISIS, RENOVATION, RISK, DEMAND,
    SCENARIO_MULTIPLIERS, DAY_OF_WEEK_MODIFIERS
)

# =============================================================================
# LLM CONFIGURATION (LiteLLM Proxy)
# =============================================================================

LLM_CLIENT = openai.OpenAI(
    api_key="sk-Q1QGmsIdw1v33-edjyEH-Q",
    base_url="http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com",
)

# Model to use - BEST model for maximum accuracy
LLM_MODEL = os.getenv("AGENT_MODEL", "gpt-4o")

# Token limits - MAXIMIZED for comprehensive context and reasoning
MAX_TOKENS = 10000      # Output tokens for detailed response
MAX_RETRIES = 3        # Retry on failure to avoid fallbacks
LLM_TIMEOUT = 28       # Maximize time budget (30s tick - 2s buffer)

# Enable/disable LLM (fallback to pure rules if False or on error)
USE_LLM = True

STRATEGY_PROMPT = """\
You are an ELITE restaurant profit optimizer AI for a 30-day Italian restaurant
simulation. Goal: MAXIMIZE total score (net profit minus penalties).

SCORE = Net Profit - Penalties
BANKRUPTCY (cash < 0) = -100,000 → GAME OVER
PRIORITY: Cash Safety > Reputation > Walkouts > Waste > Revenue

═══════════════════════════════════════════════════════════════════════════════
⚠️  MOST CRITICAL FIELDS — READ THESE EVERY SINGLE TURN  ⚠️
═══════════════════════════════════════════════════════════════════════════════
Most teams fail because they miss these two fields. They are non-negotiable:

  1. service_summary.dishes_unavailable_at
     → tells you which dish ran out and at what hour
     → back-solve the ingredient and reorder BEFORE next service
     → a 14:00 stockout = ~8 hours of lost sales for that dish

  2. pending_orders
     → tells you what is already on the way and when it arrives
     → NEVER recommend a higher order_multiplier without checking this first
     → double-ordering creates waste penalties AND spends cash you need

If your strategic_notes don't reference at least one of these when the data
warrants it, your recommendation is incomplete. Treat them as mandatory inputs.

═══════════════════════════════════════════════════════════════════════════════
PRIMARY SIGNALS (in priority order — read these first every turn)
═══════════════════════════════════════════════════════════════════════════════
1. service_summary.dishes_unavailable_at — the #1 signal (see above).
2. pending_orders — the #2 signal (see above). Sum quantity_kg and check
   delivery_day BEFORE adjusting order_multiplier.
3. service_summary.walkout_band — "None" / "Few" / "Some" / "Many". "Some" or
   "Many" two days in a row means staff up or fix wait times now.
4. reputation_band — movement matters more than the level. A drop from
   "Very Good" → "Good" is an early warning of a spiral.
5. customer_trend — "Declining" predicts walkouts BEFORE they show in the band.
6. weather_forecast — rainy/stormy = fewer covers; sunny weekend = more.

═══════════════════════════════════════════════════════════════════════════════
WHAT YOU CANNOT SEE (decide under uncertainty)
═══════════════════════════════════════════════════════════════════════════════
• Exact satisfaction score (only the reputation band)
• Exact walkout count (only the band)
• Supplier reliability ratings (only the delivery_history you can mine)
• Customer cohort sizes
• Other teams' games
Never pretend you know more than the bands say. Plan for the worst case
within each band when stakes are high (reputation, cash).

═══════════════════════════════════════════════════════════════════════════════
PENALTY WEIGHTS
═══════════════════════════════════════════════════════════════════════════════
• Walkout: ~1 point per customer (+ compounding bad reviews)
• Reputation below "Good": QUADRATIC penalty (Poor=100+/day, Fair=25-50/day)
• Waste: ~0.5 per kg

═══════════════════════════════════════════════════════════════════════════════
DEMAND BY DAY
═══════════════════════════════════════════════════════════════════════════════
Sunday: 0 CLOSED → Staff: 5 (HARD RULE)
Monday: 70-110 → Staff: 7-8
Tuesday: 75-115 → Staff: 7-8
Wednesday: 90-150 → Staff: 8-9
Thursday: 100-165 → Staff: 9-10
Friday: 140-260 → Staff: 11-13, PREMIUM PRICES
Saturday: 150-280 → Staff: 12-14, PREMIUM PRICES

Formula: staff = (expected_covers / 15) + 2, capped at 14

HOURLY DEMAND PATTERN: covers cluster around lunch (12-14) and dinner (19-22).
hourly_covers in service_summary shows where pressure was. If peak-hour
walkouts are high, the answer is MORE STAFF AT THE PEAK, not more all day.

═══════════════════════════════════════════════════════════════════════════════
SUPPLY-CHAIN TIMING (critical — most stockouts come from this)
═══════════════════════════════════════════════════════════════════════════════
• Lead time is 1-2 days AND suppliers only deliver on SPECIFIC days.
• Example: a Wed-only supplier ordered Thursday with 1-day lead arrives
  NEXT WEDNESDAY — six days later, not tomorrow.
• Before ordering, mentally simulate the delivery day. Never assume
  "1-day lead = arrives tomorrow".
• For Saturday demand, the order has to be placed by Wednesday (and
  Thursday is the last realistic safety net) — Friday is too late.

═══════════════════════════════════════════════════════════════════════════════
PERISHABILITY (FIFO consumption)
═══════════════════════════════════════════════════════════════════════════════
• Shelf life is 3-14 days depending on ingredient.
• Oldest batches are consumed first.
• Over-ordering creates waste penalties; right amount = cover the gap to the
  NEXT delivery plus one day of buffer.
• Track expires_in_days in inventory; if a batch is expiring in 1-2 days,
  promote that dish as the daily special so it gets used.

═══════════════════════════════════════════════════════════════════════════════
TABLES & CAPACITY
═══════════════════════════════════════════════════════════════════════════════
• 22 tables: 4× 2-seat, 8× 4-seat, 6× 6-seat, 4× 8-seat.
• Customers take the smallest table that fits, wait briefly if none, leave.
• Walkouts come mostly from WAIT TIME (staff + table turnover), not just
  raw seat count being exceeded.

═══════════════════════════════════════════════════════════════════════════════
REPUTATION SPIRAL (sticky, asymmetric)
═══════════════════════════════════════════════════════════════════════════════
• Starts at "Very Good". Negative experiences have outsized impact.
• A single bad day takes MANY good days to recover from.
• Strategy: PREVENT bad days rather than chase great ones. When trading off,
  accept slightly higher cost (more staff, smaller order) to avoid stockouts
  or walkouts — the reputation save is worth more than the saved euros.

═══════════════════════════════════════════════════════════════════════════════
STATE MACHINE PHASES (one is active each turn — given to you in input)
═══════════════════════════════════════════════════════════════════════════════
• baseline: normal ops, optimize freely (order 1.0, staff 0, price 1.0)
• renovation: 12-14 days reduced capacity (order 0.5, staff -3, price 0.95)
• supply_crisis: supplier halted (order 1.5 from healthy ones, staff 0, price 1.0)
• tourist_normal: days 1-4 of tourist scenario (1.0, 0, 1.0)
• tourist_prestock: days 5-8, LOAD UP for surge (order 2.5, staff 0, price 1.0)
• tourist_surge: days 9-14, already stocked — DO NOT over-order (order 1.0, staff +3, price 1.15)
• tourist_crash: days 15+, demand cratered (order 0.4, staff -2, price 0.9)

The state machine recommendation is in the user message. You may FINE-TUNE
the multipliers within +/-15% but DO NOT contradict the phase.

═══════════════════════════════════════════════════════════════════════════════
HIDDEN SCENARIOS (final evaluation includes scenarios you have NOT seen)
═══════════════════════════════════════════════════════════════════════════════
• The four named scenarios are reference points, not an exhaustive list.
• Read observation.alerts EVERY turn — alerts describe events in plain English
  (e.g. "Supplier X halted operations", "Health inspection scheduled",
  "Heatwave forecast"). React to what you read.
• When the scenario is unfamiliar, the state machine's recommendation is your
  FLOOR — fine-tune within +/-15% rather than override it.
• If alerts mention a supply problem, ORDER EARLIER from a healthy supplier.
  If alerts mention demand uncertainty, KEEP MORE BUFFER STOCK and AVOID
  PRICE HIKES.

═══════════════════════════════════════════════════════════════════════════════
CASH SAFETY (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════════════════════════
Cash < €5,000: EMERGENCY → marketing_spend=0, staff_adjustment=-2, order=0.5
Cash < €8,000: CAUTION → marketing_spend=0
Cash > €12,000: Can invest in marketing

═══════════════════════════════════════════════════════════════════════════════
ORDERING RULES (and the ANTI-DOUBLE-ORDER rule)
═══════════════════════════════════════════════════════════════════════════════
• Check pending_orders BEFORE recommending a higher order_multiplier — if
  ingredients are already inbound, more orders means waste.
• Most suppliers have 10kg MINIMUM. Don't suggest tiny tweaks the code
  will round up anyway.
• order_multiplier scales the code's calculated quantities (1.0 = normal).
• When in doubt on a high-demand day, lean toward MORE inventory — a
  stockout costs ~€5,000 in lost Saturday revenue plus reputation damage.

═══════════════════════════════════════════════════════════════════════════════
PRICING RULES
═══════════════════════════════════════════════════════════════════════════════
• price_multiplier > 1.10 requires reputation Very Good or better. If the
  band is "Good" or below, KEEP PRICE ≤ 1.08. The safety layer will cap
  you anyway — don't waste the recommendation on something that gets
  clamped.
• During recovery (Fair / Poor reputation): price ≤ 1.00.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (VALID JSON ONLY — no markdown, no explanation)
═══════════════════════════════════════════════════════════════════════════════

{"scenario_detected":"baseline","price_multiplier":1.0,"marketing_spend":0,"staff_adjustment":0,"run_happy_hour":false,"order_multiplier":1.0,"strategic_notes":"brief"}

FIELDS:
• scenario_detected: one of the 7 phases above (use the one given in input)
• price_multiplier: 0.8 to 1.2 (within +/-0.05 of state machine recommendation,
  AND respect reputation cap above)
• marketing_spend: 0 to 500 (0 if cash<8000, weekend, or during surge)
• staff_adjustment: -3 to +3 (FINE-TUNE on top of state machine — usually 0)
• run_happy_hour: true | false (slow days only, cash>10000)
• order_multiplier: 0.5 to 2.5 (within +/-0.15 of state machine recommendation)
• strategic_notes: max 200 chars — what to remember for tomorrow

RESPOND WITH JSON ONLY. NO OTHER TEXT."""


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


# Day-of-week ordering for forecast helpers
_DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def pending_arriving_by(obs: dict, ingredient: str, max_delivery_day: int) -> float:
    """Sum pending quantities for `ingredient` that will arrive on or before `max_delivery_day`.
    
    Orders with no `delivery_day` field are conservatively excluded (treated as far-future).
    Use this whenever the question is "will this be here in time?" — for example when
    checking Wednesday's prep for Saturday, max_delivery_day = today + days_to_saturday.
    """
    total = 0.0
    for order in obs.get("pending_orders", []):
        if order.get("ingredient") != ingredient:
            continue
        dd = order.get("delivery_day")
        if dd is None:
            continue
        try:
            if int(dd) <= max_delivery_day:
                total += float(order.get("quantity_kg", 0))
        except (ValueError, TypeError):
            continue
    return total


def forecast_demand_through(obs: dict, target_dow: str, use_max: bool = False,
                            safety_factor: float = 1.0) -> dict[str, float]:
    """Estimate ingredient consumption (kg) from today through `target_dow` inclusive.
    
    Uses DEMAND["<dow>"]["expected"] (or "max" if use_max=True) from config — the
    source of truth for covers per day — and distributes covers evenly across active
    dishes. `safety_factor` multiplies the result for over-provisioning.
    Returns {ingredient: total_kg_needed_through_target_dow}.
    
    Why not yesterday's dishes_sold? Because yesterday might have been Sunday (0 covers)
    or a slow Tuesday, and we'd massively under-order for the upcoming Saturday surge.
    """
    today_dow = obs.get("day_of_week", "Monday")
    if today_dow not in _DOW_ORDER or target_dow not in _DOW_ORDER:
        return {}
    today_idx = _DOW_ORDER.index(today_dow)
    target_idx = _DOW_ORDER.index(target_dow)
    if today_idx > target_idx:
        return {}
    
    active_dishes = [d for d in obs.get("menu_book", []) if d.get("is_active")]
    if not active_dishes:
        return {}
    n_dishes = len(active_dishes)
    
    covers_key = "max" if use_max else "expected"
    needed: dict[str, float] = {}
    for i in range(today_idx, target_idx + 1):
        dow = _DOW_ORDER[i]
        day_data = DEMAND.get(dow, {})
        # Fall back to expected if max isn't set
        covers = day_data.get(covers_key, day_data.get("expected", 0))
        if covers <= 0:
            continue
        covers_per_dish = (covers * safety_factor) / n_dishes
        for dish in active_dishes:
            for ing in dish.get("ingredients", []):
                ing_name = ing["ingredient"]
                qty_per_portion = ing.get("quantity_kg", 0)
                needed[ing_name] = needed.get(ing_name, 0) + (qty_per_portion * covers_per_dish)
    return needed


def days_until_dow(today_dow: str, target_dow: str) -> int:
    """Number of calendar days from `today_dow` to `target_dow` (0 if same day, negative if past)."""
    if today_dow not in _DOW_ORDER or target_dow not in _DOW_ORDER:
        return 0
    return _DOW_ORDER.index(target_dow) - _DOW_ORDER.index(today_dow)


def reputation_price_cap(reputation: str) -> float:
    """Maximum price multiplier the reputation band can absorb without driving walkouts."""
    return {
        "Excellent": 1.20,
        "Very Good": 1.15,
        "Good": 1.08,
        "Fair": 1.00,
        "Poor": 0.95,
    }.get(reputation, 1.08)


# =============================================================================
# ORDERING STRATEGIES (Code-based, reliable)
# =============================================================================

def get_supplier_min_order(obs: dict, supplier_name: str) -> float:
    """Get minimum order quantity for a supplier."""
    for supplier in obs.get("supplier_catalog", []):
        if supplier["name"] == supplier_name:
            return supplier.get("min_order_kg", 10.0)
    return 10.0  # Default minimum


def enforce_min_order(obs: dict, supplier_name: str, quantity: float) -> float:
    """Ensure order quantity meets supplier minimum."""
    min_order = get_supplier_min_order(obs, supplier_name)
    return max(quantity, min_order)


def day_one_bulk_order(obs: dict, scale: float = 1.0) -> list[dict]:
    """Day 1 (Monday): Order base quantity of every ingredient needed for active menu.
    
    The `scale` parameter is the resolved final order_multiplier applied ONCE
    at the source. Do not multiply elsewhere.
    """
    supplier_map = build_supplier_map(obs)
    orders = []
    
    needed = set()
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            for ing in dish.get("ingredients", []):
                needed.add(ing["ingredient"])
    
    base = ORDERING["day1_bulk_normal"]
    for ing in sorted(needed):
        if ing in supplier_map:
            supplier = supplier_map[ing]
            qty = enforce_min_order(obs, supplier, round(base * scale))
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier, "ingredient": ing, "quantity_kg": qty}
            })
    
    return orders


def early_week_boost(obs: dict, day: int, scale: float = 1.0) -> list[dict]:
    """Days 2-3: Extra orders to ensure first weekend has stock.
    
    `scale` is the resolved final order_multiplier applied ONCE at source.
    """
    if day not in ORDERING["early_week_days"]:
        return []
    
    supplier_map = build_supplier_map(obs)
    inventory = get_inventory_levels(obs)
    
    orders = []
    
    needed = set()
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            for ing in dish.get("ingredients", []):
                needed.add(ing["ingredient"])
    
    base = ORDERING["early_week_qty"]
    for ing in sorted(needed):
        current = inventory.get(ing, 0)
        pending = sum(
            o["quantity_kg"] for o in obs.get("pending_orders", []) 
            if o["ingredient"] == ing
        )
        total = current + pending
        
        if total < ORDERING["early_week_threshold"] and ing in supplier_map:
            supplier = supplier_map[ing]
            qty = enforce_min_order(obs, supplier, round(base * scale))
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier, "ingredient": ing, "quantity_kg": qty}
            })
    
    return orders


def stockout_recovery(obs: dict, scale: float = 1.0) -> list[dict]:
    """Emergency orders for dishes that ran out yesterday.
    
    `scale` is the resolved final order_multiplier applied ONCE at source.
    Stockout recovery uses MAX(scale, 1.0) — never reduce emergency orders.
    
    Pending orders are no longer a blanket skip — only skip if pending inbound
    in the next 1 day already covers the stockout_qty. Otherwise emergency
    means emergency and we order again.
    """
    ss = obs.get("service_summary") or {}
    stockouts = ss.get("dishes_unavailable_at", {})
    
    if not stockouts:
        return []
    
    supplier_map = build_supplier_map(obs)
    menu_book = {d["name"]: d for d in obs.get("menu_book", [])}
    today_day = obs.get("day", 1)
    orders = []
    ordered_ingredients = set()
    
    # Emergency: never scale BELOW 1.0 for stockout recovery
    effective_scale = max(scale, 1.0)
    base = ORDERING["stockout_qty"]
    threshold_for_skip = base * effective_scale
    
    for dish_name in stockouts:
        if dish_name not in menu_book:
            continue
        for ing in menu_book[dish_name].get("ingredients", []):
            ing_name = ing["ingredient"]
            if ing_name not in supplier_map or ing_name in ordered_ingredients:
                continue
            # Only skip if enough is already arriving by tomorrow
            inbound_soon = pending_arriving_by(obs, ing_name, today_day + 1)
            if inbound_soon >= threshold_for_skip:
                continue
            supplier = supplier_map[ing_name]
            qty = enforce_min_order(obs, supplier, round(base * effective_scale))
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier, "ingredient": ing_name, "quantity_kg": qty}
            })
            ordered_ingredients.add(ing_name)
    
    return orders


def smart_restock(obs: dict, base_threshold: float = None, scale: float = 1.0) -> list[dict]:
    """Order ingredients when total near-term available falls below threshold.
    
    `scale` is the resolved final order_multiplier applied ONCE at source.
    The day_multipliers still adjust the THRESHOLD (when to trigger) but not
    the order quantity — quantity is scaled by `scale` only.
    
    Time-aware pending: only count pending orders arriving within 2 days when
    checking the threshold. A delivery scheduled six days out doesn't help us
    avoid a stockout today.
    """
    if base_threshold is None:
        base_threshold = ORDERING["base_threshold"]
    
    supplier_map = build_supplier_map(obs)
    inventory = get_inventory_levels(obs)
    usage = get_menu_ingredients(obs)
    day_of_week = obs.get("day_of_week", "Monday")
    today_day = obs.get("day", 1)
    
    orders = []
    
    # Day multiplier adjusts the THRESHOLD (when to order) not the QUANTITY
    threshold_multiplier = ORDERING["day_multipliers"].get(day_of_week, 1.0)
    effective_threshold = base_threshold * threshold_multiplier
    
    menu_ingredients = set()
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            for ing in dish.get("ingredients", []):
                menu_ingredients.add(ing["ingredient"])
    
    for ing in sorted(menu_ingredients):
        current_stock = inventory.get(ing, 0)
        # Only count pending that arrives within 2 days for threshold check
        pending_qty = pending_arriving_by(obs, ing, today_day + 2)
        total_available = current_stock + pending_qty
        
        daily_use = usage.get(ing, ORDERING["default_daily_usage"])
        
        if total_available < effective_threshold and ing in supplier_map:
            supplier = supplier_map[ing]
            base_qty = max(ORDERING["min_order_qty"], round(daily_use * ORDERING["restock_days_supply"]))
            scaled_qty = round(base_qty * scale)
            order_qty = enforce_min_order(obs, supplier, scaled_qty)
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier, "ingredient": ing, "quantity_kg": order_qty}
            })
    
    return orders


def expiring_stock_orders(obs: dict, scale: float = 1.0) -> list[dict]:
    """Order replacements for stock expiring in 1-2 days.
    
    `scale` applied ONCE at source. Capped at 1.0 — during crash/renovation
    we still need replacements when stock expires, but never order MORE than
    base for expiry recovery.
    """
    supplier_map = build_supplier_map(obs)
    pending = get_pending_ingredients(obs)
    orders = []
    
    # Don't aggressively over-stock on expiry replacement
    effective_scale = min(scale, 1.0)
    base = ORDERING["expiry_order_qty"]
    
    for item in obs.get("inventory", []):
        ing = item["ingredient"]
        batches = item.get("batches", [])
        
        expiring_soon = sum(
            b["quantity_kg"] for b in batches 
            if b.get("expires_in_days", 99) <= ORDERING["expiry_days_threshold"]
        )
        remaining = item["total_kg"] - expiring_soon
        
        if remaining < ORDERING["expiry_remaining_threshold"] and ing not in pending and ing in supplier_map:
            supplier = supplier_map[ing]
            qty = enforce_min_order(obs, supplier, round(base * effective_scale))
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier, "ingredient": ing, "quantity_kg": qty}
            })
    
    return orders


def weekend_prep_orders(obs: dict, scale: float = 1.0) -> list[dict]:
    """Order enough stock to survive from today through Saturday.
    
    Caller decides when to invoke (typically Wednesday for primary prep and
    Thursday as a defensive top-up). Uses DEMAND config for expected covers
    per day-of-week — NOT yesterday's sales, which mis-estimate if yesterday
    was a slow Tuesday or a Sunday.
    
    Time-aware pending: only counts pending orders that will arrive by
    Saturday. A six-day-out delivery doesn't save Saturday service.
    
    `scale` is the resolved final order_multiplier applied ONCE at source.
    """
    day_of_week = obs.get("day_of_week", "Monday")
    
    if day_of_week not in ("Wednesday", "Thursday"):
        return []
    
    # Use MAX demand (not expected) and a 1.15 safety factor — Saturday is the
    # highest-revenue day; running out is catastrophic (~€5k of lost revenue plus
    # multi-day reputation damage). Waste from a 15% overshoot is far cheaper.
    needed_by_sat = forecast_demand_through(obs, "Saturday", use_max=True, safety_factor=1.15)
    if not needed_by_sat:
        return []
    
    supplier_map = build_supplier_map(obs)
    inventory = get_inventory_levels(obs)
    today_day = obs.get("day", 1)
    days_to_sat = days_until_dow(day_of_week, "Saturday")
    sat_day_number = today_day + days_to_sat
    # Per-ingredient flat buffer on top of forecasted demand
    buffer_kg = max(ORDERING.get("weekend_buffer", 5), 15)
    min_weekend_order = max(ORDERING.get("weekend_min_order", 15), 20)
    
    orders = []
    for ing, demand_kg in needed_by_sat.items():
        if ing not in supplier_map:
            continue
        current = inventory.get(ing, 0)
        inbound_by_sat = pending_arriving_by(obs, ing, sat_day_number)
        total_available_by_sat = current + inbound_by_sat
        gap = demand_kg + buffer_kg - total_available_by_sat
        if gap <= 0:
            continue
        supplier = supplier_map[ing]
        base_qty = max(min_weekend_order, math.ceil(gap))
        scaled_qty = round(base_qty * scale)
        order_qty = enforce_min_order(obs, supplier, scaled_qty)
        orders.append({
            "tool": "place_order",
            "args": {"supplier": supplier, "ingredient": ing, "quantity_kg": order_qty}
        })
    
    return orders


# =============================================================================
# STAFF SCHEDULING (Code-based, deterministic)
# =============================================================================

def get_staff_action(
    obs: dict,
    scenario_phase: str = "baseline",
    extra_adjustment: int = 0,
) -> dict:
    """Adaptive staff scheduling with single-source-of-truth scenario logic.
    
    The state machine passes the resolved scenario_phase and a final
    extra_adjustment. This function handles base level, weather, walkouts,
    and cash safety. Scenario-specific cuts (renovation) are applied based on
    `scenario_phase` ONLY, not via alert detection — preventing double-application.
    """
    day_of_week = obs.get("day_of_week", "Monday")
    
    # HARD OVERRIDE: Sunday is ALWAYS 5 staff (closed)
    if day_of_week == "Sunday":
        return {"tool": "set_staff_level", "args": {"level": 5}}
    
    weather = obs.get("weather_today", "sunny")
    ss = obs.get("service_summary") or {}
    walkout_band = ss.get("walkout_band", "None")
    cash = obs.get("cash", 10000)
    yesterday_covers = ss.get("total_covers", 100)
    
    # Base levels by day of week
    level = STAFFING["base_levels"].get(day_of_week, 8)
    
    # Adaptive: scale staff based on yesterday's actual covers
    if yesterday_covers > 0:
        suggested_staff = max(STAFFING["min_staff"], 
                            min(STAFFING["max_staff"], 
                                (yesterday_covers // STAFFING["covers_per_staff"]) + 2))
        level = round(level * STAFFING["day_weight"] + suggested_staff * STAFFING["demand_weight"])
    
    # Scenario phase: only renovation needs a base cut (other phases use extra_adjustment)
    if scenario_phase == "renovation":
        level = max(STAFFING["min_staff"], level - STAFFING["reduced_capacity_cut"])
    
    # Weather adjustments (always apply)
    if weather in ("rainy", "stormy"):
        level -= STAFFING["bad_weather_cut"]
    elif weather == "sunny" and day_of_week in ("Friday", "Saturday"):
        level += STAFFING["sunny_weekend_boost"]
    
    # Walkout recovery (always apply - this responds to actual observed pain)
    if walkout_band == "Many":
        level += STAFFING["walkout_many_boost"]
    elif walkout_band == "Some":
        level += STAFFING["walkout_some_boost"]
    
    # Cash safety (always apply - survival overrides scenario)
    if cash < STAFFING["cash_critical"]:
        level = max(level - STAFFING["critical_cut"], STAFFING["min_staff"])
    elif cash < STAFFING["cash_warning"]:
        level = max(level - STAFFING["warning_cut"], STAFFING["min_staff"] + 1)
    
    # Apply the resolved extra_adjustment (from state machine + LLM merge)
    level += extra_adjustment
    
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


def update_historical_notes(obs: dict, day: int, notes_data: dict) -> dict:
    """Update notes with comprehensive historical tracking for LLM context.
    
    Sunday-aware: when today is Monday (so service_summary reflects Sunday's
    closed-day zeros), the cover history still gets the 0 appended for context,
    but cash trend and crash detection are skipped — Sunday data poisons
    trend comparisons and would falsely fire CRASH_DETECTED every Monday.
    """
    ss = obs.get("service_summary") or {}
    today_dow = obs.get("day_of_week", "Monday")
    yesterday_was_sunday = today_dow == "Monday"
    
    # Track recent covers (rolling 5-day for pattern detection)
    covers = ss.get("total_covers", 0)
    prev_covers = notes_data.get("COVERS", "")
    if prev_covers:
        covers_list = prev_covers.split(",")[-4:]  # Keep last 4
        covers_list.append(str(covers))
        notes_data["COVERS"] = ",".join(covers_list)
    else:
        notes_data["COVERS"] = str(covers)
    
    # Track revenue history (rolling 5-day)
    revenue = obs.get("yesterday_revenue", 0)
    prev_rev = notes_data.get("REV", "")
    if prev_rev:
        rev_list = prev_rev.split(",")[-4:]
        rev_list.append(str(int(revenue)))
        notes_data["REV"] = ",".join(rev_list)
    else:
        notes_data["REV"] = str(int(revenue))
    
    # Track walkout history with severity
    walkout = ss.get("walkout_band", "None")
    walkout_history = notes_data.get("WALK_HIST", "")
    walk_code = {"None": "0", "Few": "1", "Some": "2", "Many": "3"}.get(walkout, "0")
    if walkout_history:
        walk_list = walkout_history.split(",")[-4:]
        walk_list.append(walk_code)
        notes_data["WALK_HIST"] = ",".join(walk_list)
    else:
        notes_data["WALK_HIST"] = walk_code
    
    # Track stockout incidents with dish names
    stockouts = ss.get("dishes_unavailable_at", {})
    if stockouts:
        notes_data["LAST_STOCKOUT"] = f"D{day}:{','.join(list(stockouts.keys())[:3])}"
    
    # Track cumulative walkouts estimate (tolerate both int and float string formats)
    try:
        prev_total_walk = int(float(notes_data.get("TOTAL_WALK", "0")))
    except (ValueError, TypeError):
        prev_total_walk = 0
    walk_estimate = {"None": 0, "Few": 2, "Some": 12, "Many": 30}.get(walkout, 0)
    notes_data["TOTAL_WALK"] = str(prev_total_walk + walk_estimate)
    
    cash = obs.get("cash", 15000)
    
    # Track cash trend — skip Mondays (Sunday closed-day cash drop is noise)
    if not yesterday_was_sunday:
        prev_cash = notes_data.get("CASH_PREV", str(cash))
        try:
            cash_change = cash - float(prev_cash)
            notes_data["CASH_DELTA"] = f"{'+' if cash_change >= 0 else ''}{int(cash_change)}"
            if cash_change < -2000:
                notes_data["TREND"] = "CRITICAL_LOSS"
            elif cash_change < -500:
                notes_data["TREND"] = "DECLINING"
            elif cash_change > 2000:
                notes_data["TREND"] = "STRONG_GROWTH"
            elif cash_change > 500:
                notes_data["TREND"] = "GROWING"
            else:
                notes_data["TREND"] = "STABLE"
        except ValueError:
            notes_data["TREND"] = "UNKNOWN"
            notes_data["CASH_DELTA"] = "0"
    notes_data["CASH_PREV"] = str(int(cash))
    
    # Track peak cash and low cash (tolerate both int and float string formats)
    cash_int = int(cash)
    try:
        peak_cash = int(float(notes_data.get("PEAK_CASH", cash_int)))
    except (ValueError, TypeError):
        peak_cash = cash_int
    try:
        low_cash = int(float(notes_data.get("LOW_CASH", cash_int)))
    except (ValueError, TypeError):
        low_cash = cash_int
    notes_data["PEAK_CASH"] = str(max(peak_cash, cash_int))
    notes_data["LOW_CASH"] = str(min(low_cash, cash_int))
    
    # NOTE: sticky SURGE_DETECTED flag deliberately removed. Tourist entry is
    # now re-evaluated every turn from the cover history in detect_scenario_phase.
    # A single Friday spike no longer flips the agent for the rest of the game.
    
    # ==========================================================================
    # CRASH DETECTION: Catch surge-to-crash transition
    # ==========================================================================
    # Critical for tourist season — detecting when the surge ends so we cut
    # orders to 0.4x and reduce staff. Zero-cover days (Sunday or full-stockout
    # Saturday) are filtered out, otherwise EVERY Monday looks like a "crash".
    prev_phase = notes_data.get("PHASE", "baseline")
    if not yesterday_was_sunday and prev_phase in ("tourist_surge", "tourist_prestock"):
        covers_hist = notes_data.get("COVERS", "").split(",")
        try:
            recent = [int(c) for c in covers_hist[-3:] if c.isdigit() and int(c) > 0]
            if len(recent) >= 2 and recent[0] > 0 and recent[-1] > 0:
                if recent[-1] < recent[0] * 0.6:
                    notes_data["CRASH_DETECTED"] = "YES"
                    print(f"  CRASH DETECTED: covers dropped from {recent[0]} to {recent[-1]}")
        except (ValueError, IndexError):
            pass
    
    return notes_data


# =============================================================================
# STATE MACHINE SCENARIO DETECTION
# =============================================================================

def detect_scenario_phase(obs: dict, day: int, notes_data: dict) -> str:
    """State machine for scenario detection with explicit phase tracking.
    
    This is the FIRST computation every turn - determines which mode the agent
    operates in. All subsequent decisions (ordering, staffing, pricing) use
    the multipliers associated with this phase.
    
    Phases:
    - baseline: Normal operations
    - renovation: Reduced capacity for ~14 days
    - supply_crisis: Supplier halted, need alternatives
    - tourist_normal: Days 1-4 of tourist season
    - tourist_prestock: Days 5-8, ORDER 2.5x to prepare for surge
    - tourist_surge: Days 9-14, DON'T over-order (already stocked), max staff
    - tourist_crash: Days 15+, REDUCE orders to 0.4x, cut staff
    """
    alerts = obs.get("alerts", [])
    alert_text = " ".join(str(a).lower() for a in alerts)
    cash = obs.get("cash", 15000)
    ss = obs.get("service_summary") or {}
    covers = ss.get("total_covers", 0)
    prev_phase = notes_data.get("PHASE", "baseline")
    
    # =========================================================================
    # RENOVATION: Explicit detection (highest priority after supply crisis)
    # =========================================================================
    if day == 1 and cash >= 16500:
        if any(kw in alert_text for kw in DETECTION["capacity_keywords"]):
            return "renovation"
    
    # Continue renovation if we were in it and haven't passed day 14
    if prev_phase == "renovation":
        try:
            reno_remaining = int(float(notes_data.get("RENO", "14")))
        except (ValueError, TypeError):
            reno_remaining = 14
        if reno_remaining > 0:
            return "renovation"
        # Renovation ended, return to baseline
        return "baseline"
    
    # =========================================================================
    # SUPPLY CRISIS: Alert-based detection
    # =========================================================================
    if any(kw in alert_text for kw in SUPPLY_CRISIS["crisis_keywords"]):
        return "supply_crisis"
    
    # =========================================================================
    # TOURIST SEASON: Phase-based state machine with explicit transitions
    # =========================================================================
    
    # Check if we're already in tourist season (from previous day's phase)
    in_tourist = prev_phase.startswith("tourist_")
    
    # Tourist entry requires a SUSTAINED surge — not a single Friday spike.
    # Real tourist surge runs 300-470 covers across many days; baseline Saturday
    # tops out around 280. Require 2 of the last 3 non-zero days to exceed 280.
    surge_detected = False
    if not in_tourist and day >= 9:
        covers_hist = notes_data.get("COVERS", "").split(",")
        try:
            non_zero_hist = [int(c) for c in covers_hist if c.isdigit() and int(c) > 0]
        except ValueError:
            non_zero_hist = []
        if covers > 0:
            non_zero_hist = non_zero_hist + [covers]
        last_three = non_zero_hist[-3:]
        high_days = sum(1 for c in last_three if c > 280)
        if high_days >= 2:
            surge_detected = True
    
    # EXIT condition: if we somehow entered tourist mode but recent reality
    # looks like baseline (last 2 non-zero days both < 200 covers), return to
    # baseline immediately. Self-corrects accidental entries within ~2 days.
    if in_tourist:
        covers_hist = notes_data.get("COVERS", "").split(",")
        try:
            non_zero_hist = [int(c) for c in covers_hist if c.isdigit() and int(c) > 0]
        except ValueError:
            non_zero_hist = []
        last_two = non_zero_hist[-2:]
        if len(last_two) == 2 and all(c < 200 for c in last_two):
            return "baseline"
    
    if in_tourist or surge_detected:
        # Check if crash was detected (from update_historical_notes)
        crash_detected = notes_data.get("CRASH_DETECTED") == "YES"
        
        # We're in tourist season - determine exact phase by day number
        if day <= 4:
            return "tourist_normal"
        elif day <= 8:
            return "tourist_prestock"   # PRE-STOCK phase: order 2.5x
        elif day <= 14:
            # If crash detected early (before day 15), transition to crash
            if crash_detected:
                return "tourist_crash"
            return "tourist_surge"      # SURGE phase: staff +3, DON'T over-order
        else:
            # After day 14, default to crash phase
            # But also check cover drop for early crash detection — filter zeros
            # so a Sunday or stockout doesn't poison the comparison.
            if not crash_detected:
                covers_hist = notes_data.get("COVERS", "").split(",")
                try:
                    recent = [int(c) for c in covers_hist[-3:] if c.isdigit() and int(c) > 0]
                    if len(recent) >= 2 and recent[0] > 0 and recent[-1] > 0:
                        if recent[-1] < recent[0] * 0.6:
                            crash_detected = True
                except (ValueError, IndexError):
                    pass
            
            # Default to crash after day 14
            return "tourist_crash"
    
    # =========================================================================
    # BASELINE: Normal operations
    # =========================================================================
    return "baseline"


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
    """Create MAXIMUM information summary for optimal LLM decisions."""
    ss = obs.get("service_summary") or {}
    
    # === CORE METRICS ===
    cash = obs.get("cash", 0)
    day_of_week = obs.get("day_of_week", "Monday")
    days_remaining = obs.get("days_remaining", 30 - day)
    reputation = obs.get("reputation_band", "Good")
    customer_trend = obs.get("customer_trend", "Stable")
    weather = obs.get("weather_today", "sunny")
    weather_forecast = obs.get("weather_forecast", [])
    
    # === YESTERDAY'S PERFORMANCE ===
    covers = ss.get("total_covers", 0)
    revenue = obs.get("yesterday_revenue", 0)
    costs = obs.get("yesterday_total_costs", 0)
    cost_breakdown = obs.get("cost_breakdown", {})
    walkout_band = ss.get("walkout_band", "None")
    stockouts = ss.get("dishes_unavailable_at", {})
    avg_wait = ss.get("avg_wait_minutes", 0)
    peak_wait = ss.get("peak_wait_minutes", 0)
    hourly_covers = ss.get("hourly_covers", [])
    
    profit = revenue - costs if costs > 0 else 0
    
    # === RISK CALCULATIONS ===
    bankruptcy_risk = calculate_bankruptcy_risk(obs)
    walkout_risk = calculate_walkout_risk(obs)
    
    # Daily costs projection
    staff_level = obs.get("staff_level", 8)
    daily_fixed_cost = 300 + (staff_level * 120)
    days_of_runway = int(cash / daily_fixed_cost) if daily_fixed_cost > 0 else 99
    
    # === ALERTS (CRITICAL) ===
    alerts = obs.get("alerts", [])
    alert_text = " | ".join(str(a) for a in alerts) if alerts else "NONE"
    
    # === COMPLETE INVENTORY WITH DETAILS ===
    inventory = obs.get("inventory", [])
    inv_lines = []
    total_inv_value = 0
    for item in inventory:
        ing = item["ingredient"]
        total_kg = item["total_kg"]
        batches = item.get("batches", [])
        
        # Calculate days until stockout based on usage
        consumption = get_consumption_rates(obs)
        daily_use = consumption.get(ing, 2.0)
        days_supply = total_kg / daily_use if daily_use > 0 else 99
        
        # Expiry info
        expiry_info = ""
        for b in batches:
            exp_days = b.get("expires_in_days", 99)
            if exp_days <= 3:
                expiry_info = f" [EXPIRES:{b['quantity_kg']:.1f}kg in {exp_days}d]"
                break
        
        status = "OK"
        if total_kg < 3:
            status = "CRITICAL"
        elif total_kg < 8:
            status = "LOW"
        elif days_supply < 2:
            status = "REORDER"
        
        inv_lines.append(f"  {ing}: {total_kg:.1f}kg ({days_supply:.1f}d supply) [{status}]{expiry_info}")
    
    inventory_text = "\n".join(inv_lines) if inv_lines else "  No inventory data"
    
    # === SUPPLIER CATALOG WITH MINIMUMS (CRITICAL FOR ORDERING) ===
    suppliers = obs.get("supplier_catalog", [])
    supplier_lines = []
    for sup in suppliers:
        name = sup["name"]
        lead = sup.get("lead_time_days", 1)
        min_order = sup.get("min_order_kg", 5)
        delivery_days = sup.get("delivery_days", [])
        ingredients = sup.get("ingredients", {})
        
        # Check if supplier mentioned in alerts (outage)
        is_halted = any(name.lower() in str(a).lower() for a in alerts)
        status = " [HALTED!]" if is_halted else ""
        
        ing_list = ", ".join(f"{k}:€{v:.2f}/kg" for k, v in list(ingredients.items())[:5])
        supplier_lines.append(f"  {name}{status}: min={min_order}kg, lead={lead}d, delivers={','.join(delivery_days[:3])}")
        supplier_lines.append(f"    Items: {ing_list}")
    
    supplier_text = "\n".join(supplier_lines) if supplier_lines else "  No suppliers"
    
    # === PENDING ORDERS (KNOW WHAT'S COMING) ===
    pending = obs.get("pending_orders", [])
    pending_lines = []
    pending_by_ing = {}
    for order in pending:
        ing = order["ingredient"]
        qty = order["quantity_kg"]
        delivery = order.get("delivery_day", "?")
        pending_by_ing[ing] = pending_by_ing.get(ing, 0) + qty
        pending_lines.append(f"  {ing}: {qty}kg arriving Day {delivery}")
    
    pending_text = "\n".join(pending_lines) if pending_lines else "  No pending orders"
    
    # === MENU WITH PRICES AND INGREDIENT REQUIREMENTS ===
    menu_book = obs.get("menu_book", [])
    menu_lines = []
    dishes_sold = ss.get("dishes_sold", {})
    for dish in menu_book:
        if dish.get("is_active"):
            name = dish["name"]
            base_price = dish["base_price"]
            current_price = dish["current_price"]
            sold = dishes_sold.get(name, 0)
            
            # Ingredient requirements
            ings = dish.get("ingredients", [])
            ing_text = ", ".join(f"{i['ingredient']}:{i['quantity_kg']}kg" for i in ings[:3])
            
            price_status = ""
            if current_price > base_price * 1.05:
                price_status = " [PREMIUM]"
            elif current_price < base_price * 0.95:
                price_status = " [DISCOUNT]"
            
            menu_lines.append(f"  {name}: €{current_price:.2f} (base €{base_price:.2f}){price_status} | Sold:{sold} | Uses: {ing_text}")
    
    menu_text = "\n".join(menu_lines[:10]) if menu_lines else "  No active menu"
    
    # === DELIVERY HISTORY (RELIABILITY) ===
    delivery_hist = obs.get("delivery_history", [])
    reliability = get_supplier_reliability(obs)
    reliability_text = ", ".join(f"{s}:{r:.0%}" for s, r in reliability.items()) if reliability else "No history"
    
    # === STRATEGIC NOTES (HISTORY) ===
    notes = obs.get("notes", "")
    
    # === EXPECTED DEMAND BY DAY ===
    demand_expectations = {
        "Sunday": "CLOSED (0 covers)",
        "Monday": "LOW (70-110 covers)",
        "Tuesday": "LOW (75-115 covers)",
        "Wednesday": "MEDIUM (90-150 covers)",
        "Thursday": "MEDIUM-HIGH (100-165 covers)",
        "Friday": "HIGH (140-260 covers)",
        "Saturday": "HIGHEST (150-280 covers)"
    }
    expected_today = demand_expectations.get(day_of_week, "Unknown")
    
    # === BUILD COMPREHENSIVE SUMMARY ===
    summary = f"""══════════════════════════════════════════════════════════════════════════════
DAY {day}/30 ({day_of_week}) — {days_remaining} days remaining — Expected: {expected_today}
══════════════════════════════════════════════════════════════════════════════

💰 FINANCIAL STATUS:
  Cash: €{cash:,.0f} | Daily fixed costs: €{daily_fixed_cost} | Runway: {days_of_runway} days
  Yesterday: €{revenue:,.0f} revenue - €{costs:,.0f} costs = €{profit:,.0f} profit
  Costs: Staff €{cost_breakdown.get('staff', 0):.0f} + Fixed €{cost_breakdown.get('fixed', 0):.0f} + Waste €{cost_breakdown.get('waste', 0):.1f}
  
⚠️ RISK ASSESSMENT:
  Bankruptcy Risk: {bankruptcy_risk:.0%} {'🚨 DANGER!' if bankruptcy_risk > 0.2 else '✅ OK'}
  Walkout Risk: {walkout_risk}
  Cash Safety: {'🔴 CRITICAL' if cash < 5000 else '🟡 CAUTION' if cash < 8000 else '🟢 HEALTHY'}
  
📊 REPUTATION & MARKET:
  Reputation: {reputation} | Customer Trend: {customer_trend}
  Weather: {weather} | Forecast: {', '.join(weather_forecast[:3]) if weather_forecast else 'N/A'}

📈 YESTERDAY'S PERFORMANCE:
  Covers: {covers} | Walkouts: {walkout_band}
  Wait: avg {avg_wait:.1f}min, peak {peak_wait:.1f}min
  Hourly pattern: {hourly_covers if hourly_covers else 'N/A'}
  Stockouts: {', '.join(stockouts.keys()) if stockouts else 'None'}

👥 STAFFING:
  Level: {staff_level} | Cost: €{staff_level * 120}/day | Efficiency: {covers/staff_level:.1f} covers/staff

🚨 ALERTS: {alert_text}

📦 INVENTORY (with supply days & status):
{inventory_text}

🚚 PENDING ORDERS (already ordered - DO NOT DUPLICATE):
{pending_text}

🏪 SUPPLIERS (RESPECT MINIMUM ORDERS!):
{supplier_text}
  Reliability: {reliability_text}

🍝 ACTIVE MENU (with prices & yesterday's sales):
{menu_text}

📝 HISTORICAL NOTES (patterns & context):
{notes if notes else 'Day 1 - No history yet'}

══════════════════════════════════════════════════════════════════════════════
DECISION GUIDANCE:
- Today is {day_of_week}: {expected_today}
- Minimum order: 10kg for most suppliers (check above)
- Staff sweet spot: (expected_covers / 15) + 2
- Price range: 0.8x to 1.2x base price
- Marketing: 0-500 EUR (skip if cash < 8000 or weekend)
══════════════════════════════════════════════════════════════════════════════"""
    
    return summary


def clean_json_response(content: str) -> str:
    """Aggressively clean LLM response to extract valid JSON."""
    if not content:
        return "{}"
    
    content = content.strip()
    
    # Remove markdown code blocks (various formats)
    if "```" in content:
        # Find content between ``` markers
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            # Skip the language identifier line
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and "}" in part:
                content = part
                break
    
    # Remove "json" prefix if present
    if content.lower().startswith("json"):
        content = content[4:].strip()
    
    # Find the JSON object boundaries
    start = content.find("{")
    end = content.rfind("}")
    
    if start == -1 or end == -1 or end <= start:
        # No valid JSON found, return empty object
        return "{}"
    
    content = content[start:end + 1]
    
    # Fix common JSON issues
    content = content.replace("'", '"')  # Single quotes to double quotes
    content = content.replace("True", "true").replace("False", "false")  # Python bools
    content = content.replace("None", "null")  # Python None
    
    # Fix trailing commas before closing braces (common LLM error)
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)
    
    # Fix unquoted string values (very common error)
    # This is tricky, so we'll try to parse and if it fails, attempt more fixes
    
    return content


def get_llm_advice(
    obs: dict,
    day: int,
    scenario_phase: str = "baseline",
    state_multipliers: dict | None = None,
) -> dict | None:
    """Get strategic advice from LLM with retry logic.
    
    The state machine's resolved phase + recommended multipliers are passed
    into the user message so the LLM operates coherently with the state
    machine's structural understanding rather than guessing the scenario.
    """
    if not USE_LLM:
        return None
    
    if state_multipliers is None:
        state_multipliers = SCENARIO_MULTIPLIERS.get(scenario_phase, SCENARIO_MULTIPLIERS["baseline"])
    
    summary = summarize_observation(obs, day)
    phase_context = (
        f"\n\n═══════════════════════════════════════════════════════════════════════════════\n"
        f"STATE MACHINE DECISION (use this as your baseline)\n"
        f"═══════════════════════════════════════════════════════════════════════════════\n"
        f"Active phase: {scenario_phase}\n"
        f"Recommended order_multiplier: {state_multipliers['order']}\n"
        f"Recommended staff_adjustment: {state_multipliers['staff']}\n"
        f"Recommended price_multiplier: {state_multipliers['price']}\n"
        f"\n"
        f"Your job: fine-tune these within +/-20%. Set scenario_detected={scenario_phase}.\n"
        f"DO NOT contradict the phase (e.g., do not order 2.0x during tourist_crash).\n"
    )
    user_message = summary + phase_context
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            # Adjust temperature slightly on retries to get different output
            temp = 0.1 + (attempt * 0.1)
            
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": STRATEGY_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=temp,
                max_tokens=MAX_TOKENS,
                timeout=LLM_TIMEOUT,
            )
            
            content = response.choices[0].message.content
            content = clean_json_response(content)
            
            advice = json.loads(content)
            
            # Validate and clamp all values with safe defaults
            advice["price_multiplier"] = max(0.8, min(1.2, float(advice.get("price_multiplier", 1.0))))
            advice["marketing_spend"] = max(0, min(500, int(advice.get("marketing_spend", 0))))
            advice["staff_adjustment"] = max(-3, min(3, int(advice.get("staff_adjustment", 0))))
            advice["run_happy_hour"] = bool(advice.get("run_happy_hour", False))
            advice["order_multiplier"] = max(0.5, min(2.0, float(advice.get("order_multiplier", 1.0))))
            advice["confidence"] = max(0.0, min(1.0, float(advice.get("confidence", 0.5))))
            advice["bankruptcy_risk"] = max(0.0, min(1.0, float(advice.get("bankruptcy_risk", 0.0))))
            
            # Success! Return the advice
            if attempt > 0:
                print(f"  LLM succeeded on retry {attempt + 1}")
            return advice
            
        except json.JSONDecodeError as e:
            last_error = f"JSON: {e}"
            if attempt < MAX_RETRIES - 1:
                print(f"  LLM JSON error on day {day} (attempt {attempt + 1}/{MAX_RETRIES}), retrying...")
                continue
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                print(f"  LLM error on day {day} (attempt {attempt + 1}/{MAX_RETRIES}): {e}, retrying...")
                continue
    
    # All retries failed
    print(f"  LLM FAILED all {MAX_RETRIES} attempts on day {day}: {last_error}")
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
    
    # Reputation-aware price ceiling — applied AFTER the low-rep block so it's
    # the final word. Prevents tourist_surge's 1.15x from driving walkouts when
    # reputation is merely "Good" rather than "Very Good"/"Excellent".
    price_cap = reputation_price_cap(reputation)
    current_price = validated.get("price_multiplier", 1.0)
    if current_price > price_cap:
        validated["price_multiplier"] = price_cap
        
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
    """Apply the resolved price multiplier to all active dishes.
    
    Always emits price actions every turn so prices reset from any prior
    weekend premium / discount and reflect the current phase's intent.
    The multiplier is the final merged value (state machine + LLM + safety).
    """
    multiplier = advice.get("price_multiplier", 1.0)
    multiplier = max(0.8, min(1.2, multiplier))
    
    actions = []
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            base_price = dish["base_price"]
            new_price = round(base_price * multiplier, 2)
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
        try:
            remaining = int(float(notes.get("RENO", 0)))
        except (ValueError, TypeError):
            remaining = 0
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
    """Single-purpose-layered strategy for maximum revenue.
    
    Architecture (each layer does ONE thing then hands off resolved values):
    
    1. State machine detects scenario_phase (deterministic, structural)
    2. LLM receives phase + recommended multipliers, fine-tunes within tolerance
    3. Merge: state machine dominates during active phases (85/15),
       LLM has more weight during baseline/tourist_normal (40/60)
    4. Safety layer applies cash-safety scaling to the MERGED values
    5. Code dispatches actions with the ONE final multiplier per dimension
    """
    actions = []
    cash = observation.get("cash", 15000)
    day_of_week = observation.get("day_of_week", "Monday")
    
    # ==========================================================================
    # STEP 1: STATE MACHINE SCENARIO DETECTION (FIRST)
    # ==========================================================================
    notes_data = parse_notes(observation.get("notes", ""))
    scenario_phase = detect_scenario_phase(observation, day, notes_data)
    
    state_multipliers = SCENARIO_MULTIPLIERS.get(scenario_phase, SCENARIO_MULTIPLIERS["baseline"])
    state_order_mult = state_multipliers["order"]
    state_staff_adj = state_multipliers["staff"]
    state_price_mult = state_multipliers["price"]
    
    # ==========================================================================
    # STEP 1B: APPLY DAY-OF-WEEK MODIFIERS (boost weekends)
    # ==========================================================================
    dow_mods = DAY_OF_WEEK_MODIFIERS.get(day_of_week, {"order": 1.0, "staff": 0, "price": 1.0})
    reputation = observation.get("reputation_band", "Good")
    
    # Order: multiply (e.g., Wednesday prep boost)
    state_order_mult *= dow_mods["order"]
    
    # Staff: add (e.g., +2 on Saturday)
    state_staff_adj += dow_mods["staff"]
    
    # Price: only apply premium if reputation supports it
    dow_price = dow_mods["price"]
    if dow_price > 1.0:
        price_cap = reputation_price_cap(reputation)
        # Apply the DOW premium but cap it
        state_price_mult = min(state_price_mult * dow_price, price_cap)
    else:
        state_price_mult *= dow_price
    
    print(f"  Day {day}: Phase={scenario_phase}, order={state_order_mult:.2f}, staff={state_staff_adj:+d}, price={state_price_mult:.2f}")
    
    # ==========================================================================
    # STEP 2: LLM ADVICE (phase-aware, can fine-tune within tolerance)
    # ==========================================================================
    llm_advice = get_llm_advice(observation, day, scenario_phase, state_multipliers)
    
    if llm_advice is None:
        print(f"  Day {day}: Using rule-based fallback")
        llm_advice = generate_rule_based_advice(observation, day)
    
    llm_order_mult = llm_advice.get("order_multiplier", 1.0)
    llm_staff_adj = llm_advice.get("staff_adjustment", 0)
    llm_price_mult = llm_advice.get("price_multiplier", 1.0)
    
    # ==========================================================================
    # STEP 3: MERGE WITH PHASE-AWARE WEIGHTS
    # ==========================================================================
    # Baseline and tourist_normal: trust LLM more (it knows market nuance)
    # Active scenario phases: state machine dominates (deterministic structural knowledge)
    TRUST_LLM_PHASES = {"baseline", "tourist_normal"}
    if scenario_phase in TRUST_LLM_PHASES:
        state_weight, llm_weight = 0.4, 0.6
    else:
        state_weight, llm_weight = 0.85, 0.15
    
    merged_order_mult = (state_order_mult * state_weight) + (llm_order_mult * llm_weight)
    merged_staff_adj = state_staff_adj + int(round(llm_staff_adj * llm_weight))
    merged_price_mult = (state_price_mult * state_weight) + (llm_price_mult * llm_weight)
    
    # Build the unified advice dict for safety layer
    merged_advice = {
        "order_multiplier": merged_order_mult,
        "staff_adjustment": merged_staff_adj,
        "price_multiplier": merged_price_mult,
        "marketing_spend": llm_advice.get("marketing_spend", 0),
        "run_happy_hour": llm_advice.get("run_happy_hour", False),
        "scenario_detected": scenario_phase,
        "strategic_notes": llm_advice.get("strategic_notes", ""),
    }
    
    # ==========================================================================
    # STEP 4: SAFETY LAYER (operates on resolved merged values, not LLM intermediate)
    # ==========================================================================
    final_advice = validate_and_adjust_advice(merged_advice, observation, day)
    
    # Extract resolved final values
    order_multiplier = max(0.3, min(2.5, final_advice.get("order_multiplier", 1.0)))
    staff_adjustment = max(-4, min(4, int(final_advice.get("staff_adjustment", 0))))
    price_multiplier = max(0.8, min(1.2, final_advice.get("price_multiplier", 1.0)))
    
    # ==========================================================================
    # STEP 5: UPDATE NOTES (historical tracking + state persistence)
    # ==========================================================================
    notes_data = update_historical_notes(observation, day, notes_data)
    notes_data["PHASE"] = scenario_phase
    notes_data["SCENARIO"] = scenario_phase
    
    reduced = scenario_phase == "renovation"
    if reduced:
        try:
            existing_reno = int(float(notes_data.get("RENO", "14")))
        except (ValueError, TypeError):
            existing_reno = 14
        notes_data["RENO"] = str(max(0, existing_reno - 1))
    elif "RENO" in notes_data:
        try:
            if int(float(notes_data.get("RENO", "0"))) <= 0:
                del notes_data["RENO"]
        except (ValueError, TypeError):
            del notes_data["RENO"]
    
    if final_advice.get("strategic_notes"):
        notes_data["LLM"] = str(final_advice["strategic_notes"])[:200]
    
    if scenario_phase.startswith("tourist_"):
        notes_data["TOURIST"] = "1"
    
    # ==========================================================================
    # STEP 6: DISPATCH ACTIONS (each subsystem gets ONE resolved value)
    # ==========================================================================
    
    # --- ORDERING: pass scale ONCE, applied at source in each helper ---
    order_actions = []
    if day == 1:
        if reduced:
            order_actions = day_one_bulk_order_reduced(observation, scale=order_multiplier)
        else:
            order_actions = day_one_bulk_order(observation, scale=order_multiplier)
    else:
        if not reduced:
            order_actions.extend(early_week_boost(observation, day, scale=order_multiplier))
        order_actions.extend(stockout_recovery(observation, scale=order_multiplier))
        order_actions.extend(smart_restock(observation, ORDERING["base_threshold"], scale=order_multiplier))
        order_actions.extend(expiring_stock_orders(observation, scale=order_multiplier))
        # Wednesday: primary weekend prep. Thursday: defensive top-up that
        # patches any remaining gaps while there's still time for delivery.
        if day_of_week in ("Wednesday", "Thursday"):
            order_actions.extend(weekend_prep_orders(observation, scale=order_multiplier))
    
    actions.extend(order_actions)
    
    # --- STAFFING: pass phase + resolved adjustment ONCE ---
    staff_action = get_staff_action(
        observation,
        scenario_phase=scenario_phase,
        extra_adjustment=staff_adjustment,
    )
    actions.append(staff_action)
    
    # --- DAILY SPECIAL ---
    special = get_daily_special(observation)
    if special:
        actions.append(special)
    
    # --- PRICING: always emit so prices reset every day ---
    pricing_actions = apply_llm_pricing(observation, {"price_multiplier": price_multiplier})
    actions.extend(pricing_actions)
    
    # --- MARKETING ---
    marketing_amount = final_advice.get("marketing_spend", 0)
    if marketing_amount > 0 and cash > MARKETING["min_cash"]:
        if cash < 8000:
            marketing_amount = min(marketing_amount, 100)
        actions.append({"tool": "set_marketing_spend", "args": {"amount": marketing_amount}})
    
    # --- HAPPY HOUR ---
    if final_advice.get("run_happy_hour") and cash > MARKETING["min_cash"]:
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
    
    # Build comprehensive notes for next day's LLM context
    ss = observation.get("service_summary") or {}
    notes_data["DAY"] = str(day)
    notes_data["DOW"] = observation.get("day_of_week", "?")[:3]
    notes_data["CASH"] = str(int(observation.get("cash", 0)))
    notes_data["COV"] = str(ss.get("total_covers", 0))
    notes_data["WALK"] = ss.get("walkout_band", "None")[0]
    notes_data["REP"] = observation.get("reputation_band", "Good")[0]
    
    # Scenario tracking (first 4 chars of full phase name)
    notes_data["SCEN"] = scenario_phase[:4]
    
    # Save notes for next day
    notes_text = build_notes(notes_data)
    if len(notes_text) > 3900:
        # Truncate if too long, keeping most recent data
        # PHASE is critical for state machine continuity
        important_keys = ["DAY", "DOW", "CASH", "COV", "WALK", "REP", "SCEN", "LLM", "TREND", "PHASE", "RENO", "COVERS"]
        notes_data = {k: v for k, v in notes_data.items() if k in important_keys}
        notes_text = build_notes(notes_data)
    
    final_actions.append({"tool": "save_notes", "args": {"text": notes_text}})
    
    return final_actions


def day_one_bulk_order_reduced(obs: dict, scale: float = 1.0) -> list[dict]:
    """Day 1 for reduced capacity (renovation): smaller quantities.
    
    `scale` is the resolved final order_multiplier applied ONCE at source.
    Base is already reduced via `day1_bulk_reduced`. State machine will pass
    scale=0.5 during renovation (which compounds with the reduced base for
    the desired effect).
    """
    supplier_map = build_supplier_map(obs)
    orders = []
    
    needed = set()
    for dish in obs.get("menu_book", []):
        if dish.get("is_active"):
            for ing in dish.get("ingredients", []):
                needed.add(ing["ingredient"])
    
    base = ORDERING["day1_bulk_reduced"]
    for ing in sorted(needed):
        if ing in supplier_map:
            supplier = supplier_map[ing]
            qty = enforce_min_order(obs, supplier, round(base * scale))
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier, "ingredient": ing, "quantity_kg": qty}
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
