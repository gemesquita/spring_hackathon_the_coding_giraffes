"""Hybrid agent: Code handles reliable operations, LLM handles advanced strategy.

Key improvements over pure-LLM approach:
1. Code-based supplier matching (eliminates "wrong supplier" errors)
2. Code-based inventory management (prevents stockouts)
3. Code-based staff scheduling (deterministic, weather-aware)
4. Code-based daily specials (picks well-stocked dishes)
5. Optional LLM for pricing/marketing strategy (within size limits)
"""

from __future__ import annotations

import json
import os
import sys

from agents.runner import run_game


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
                "args": {"supplier": supplier_map[ing], "ingredient": ing, "quantity_kg": 25}
            })
    
    return orders


def early_week_boost(obs: dict, day: int) -> list[dict]:
    """Days 2-3: Extra orders to ensure first weekend has stock.
    
    The first weekend (Days 6-7) is often a stockout because:
    - Day 1 orders might not arrive until Day 3-5
    - Starting inventory is limited
    """
    if day not in (2, 3):
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
    
    # Order 15kg of anything below 25kg total available
    for ing in sorted(needed):
        current = inventory.get(ing, 0)
        pending = sum(
            o["quantity_kg"] for o in obs.get("pending_orders", []) 
            if o["ingredient"] == ing
        )
        total = current + pending
        
        if total < 35 and ing in supplier_map:  # Need ~35kg to cover week 1
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier_map[ing], "ingredient": ing, "quantity_kg": 15}
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
                    "args": {"supplier": supplier_map[ing_name], "ingredient": ing_name, "quantity_kg": 15}
                })
                ordered_ingredients.add(ing_name)
    
    return orders


def smart_restock(obs: dict, base_threshold: float = 15.0) -> list[dict]:
    """Order ingredients when total available (stock + pending) falls below threshold.
    
    Consider TOTAL available (current + pending), not just current.
    This prevents over-ordering while ensuring we always have stock.
    """
    supplier_map = build_supplier_map(obs)
    inventory = get_inventory_levels(obs)
    usage = get_menu_ingredients(obs)
    day_of_week = obs.get("day_of_week", "Monday")
    
    orders = []
    
    # Adjust threshold based on day - HIGHER early week for weekend prep
    day_multipliers = {
        "Monday": 2.5, "Tuesday": 2.0, "Wednesday": 1.8,
        "Thursday": 1.5, "Friday": 1.2, "Saturday": 1.0, "Sunday": 1.0
    }
    multiplier = day_multipliers.get(day_of_week, 1.0)
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
        
        # Use actual usage if known, else default to 2.5kg/day
        daily_use = usage.get(ing, 2.5)
        
        # Order if TOTAL available is below threshold
        if total_available < effective_threshold and ing in supplier_map:
            # Order enough for ~6 days of usage, minimum 15kg
            order_qty = max(15, round(daily_use * 6))
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
            if b.get("expires_in_days", 99) <= 2
        )
        remaining = item["total_kg"] - expiring_soon
        
        if remaining < 5 and ing not in pending and ing in supplier_map:
            orders.append({
                "tool": "place_order",
                "args": {"supplier": supplier_map[ing], "ingredient": ing, "quantity_kg": 15}
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
    # Weekend demand is ~1.5x weekday
    weekend_multiplier = 4.5  # 3 days * 1.5x demand
    
    for ing, daily_use in usage.items():
        current = inventory.get(ing, 0)
        pending_qty = sum(
            o["quantity_kg"] for o in obs.get("pending_orders", [])
            if o["ingredient"] == ing
        )
        total_available = current + pending_qty
        
        weekend_need = daily_use * weekend_multiplier
        
        if total_available < weekend_need and ing in supplier_map and ing not in pending:
            order_qty = max(20, round(weekend_need - total_available + 10))  # Extra buffer
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
    reduced_capacity = any("renovation" in str(a).lower() or "capacity" in str(a).lower() 
                          for a in alerts)
    
    # Base levels by day of week
    base_levels = {
        "Monday": 7, "Tuesday": 7,
        "Wednesday": 8, "Thursday": 9,
        "Friday": 12, "Saturday": 13, "Sunday": 5
    }
    level = base_levels.get(day_of_week, 8)
    
    # ADAPTIVE: Scale staff based on yesterday's actual covers
    # ~15 covers per staff member is efficient
    if yesterday_covers > 0:
        suggested_staff = max(5, min(14, (yesterday_covers // 15) + 2))
        # Blend: 60% day-based, 40% performance-based
        level = round(level * 0.6 + suggested_staff * 0.4)
    
    # If reduced capacity (renovation), cut staff
    if reduced_capacity:
        level = max(5, level - 3)
    
    # Weather adjustments
    if weather in ("rainy", "stormy"):
        level -= 1
    elif weather == "sunny" and day_of_week in ("Friday", "Saturday"):
        level += 1
    
    # Walkout recovery
    if walkout_band == "Many":
        level += 2
    elif walkout_band == "Some":
        level += 1
    
    # Cash safety
    if cash < 5000:
        level = max(level - 2, 5)
    elif cash < 8000:
        level = max(level - 1, 6)
    
    # Clamp to valid range
    level = max(5, min(14, level))
    
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
    if reputation in ("Good", "Very Good", "Excellent"):
        # Weekend premium
        if day_of_week in ("Friday", "Saturday"):
            for dish_name, dish in menu_book.items():
                if dish.get("is_active"):
                    base_price = dish["base_price"]
                    new_price = round(base_price * 1.1, 2)  # 10% weekend premium
                    actions.append({
                        "tool": "set_price",
                        "args": {"dish": dish_name, "price": new_price}
                    })
    elif reputation in ("Poor", "Fair"):
        # Discount to attract customers when reputation is low
        for dish_name, dish in menu_book.items():
            if dish.get("is_active"):
                base_price = dish["base_price"]
                new_price = round(base_price * 0.9, 2)  # 10% discount
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
    if cash < 5000:
        return None
    
    # Boost marketing on slower days or when trend is declining
    if customer_trend == "Declining":
        return {"tool": "set_marketing_spend", "args": {"amount": 200}}
    
    if day_of_week in ("Monday", "Tuesday") and reputation not in ("Poor", "Fair"):
        return {"tool": "set_marketing_spend", "args": {"amount": 100}}
    
    return None


# =============================================================================
# MAIN STRATEGY FUNCTION
# =============================================================================

def is_reduced_capacity(observation: dict, day: int) -> bool:
    """Detect if we're in a reduced capacity scenario (renovation, etc.).
    
    Uses notes to persist detection across days since alerts are transient.
    """
    notes = observation.get("notes", "") or ""
    
    # Check if we already detected reduced capacity (saved in notes)
    if "REDUCED_CAPACITY" in notes:
        return True
    
    alerts = observation.get("alerts", [])
    alert_text = " ".join(str(a).lower() for a in alerts)
    
    # Check for renovation/capacity keywords in current alerts
    if "renovation" in alert_text or "capacity" in alert_text or "table" in alert_text:
        return True
    
    # Also detect by checking yesterday's covers vs expected
    ss = observation.get("service_summary") or {}
    covers = ss.get("total_covers", 100)
    dow = observation.get("day_of_week", "Monday")
    
    if dow in ("Friday", "Saturday") and covers < 80 and day > 5:
        return True
    
    return False


def get_notes_action(observation: dict, day: int) -> dict | None:
    """Save important state to notes for persistence across days."""
    notes = observation.get("notes", "") or ""
    alerts = observation.get("alerts", [])
    alert_text = " ".join(str(a).lower() for a in alerts)
    
    new_notes_parts = []
    
    # Detect and save reduced capacity state
    if "renovation" in alert_text or "capacity" in alert_text:
        if "REDUCED_CAPACITY" not in notes:
            new_notes_parts.append("REDUCED_CAPACITY")
    elif "REDUCED_CAPACITY" in notes:
        new_notes_parts.append("REDUCED_CAPACITY")  # Keep it
    
    # Detect and save supply crisis state
    if "halted" in alert_text or "outage" in alert_text or "supplier" in alert_text:
        if "SUPPLY_CRISIS" not in notes:
            new_notes_parts.append("SUPPLY_CRISIS")
    elif "SUPPLY_CRISIS" in notes:
        new_notes_parts.append("SUPPLY_CRISIS")  # Keep it
    
    if new_notes_parts:
        new_notes = " | ".join(new_notes_parts) + f" | Day {day}"
        return {"tool": "save_notes", "args": {"text": new_notes}}
    
    return None


def strategy(observation: dict, day: int) -> list[dict]:
    """Hybrid strategy with multi-dimensional optimization.
    
    Inspired by business simulator breakthroughs:
    1. Staff cap - avoid over-staffing (diminishing returns)
    2. No-overlap - don't drain cash on multiple fronts simultaneously
    3. Broad prestige - optimize pricing + marketing + specials together
    4. Adapt to scenarios - detect reduced capacity and cut costs
    """
    actions = []
    cash = observation.get("cash", 15000)
    reduced = is_reduced_capacity(observation, day)
    
    # 1. ORDERING (Code-based, reliable)
    # Reduce quantities if in reduced capacity mode
    if day == 1:
        if reduced:
            # Smaller orders for reduced capacity
            actions.extend(day_one_bulk_order_reduced(observation))
        else:
            actions.extend(day_one_bulk_order(observation))
    else:
        if not reduced:
            actions.extend(early_week_boost(observation, day))
        
        actions.extend(stockout_recovery(observation))
        actions.extend(smart_restock(observation))
        actions.extend(expiring_stock_orders(observation))
    
    # 2. STAFFING (adaptive to actual demand)
    actions.append(get_staff_action(observation))
    
    # 3. DAILY SPECIAL - always free bonus
    special = get_daily_special(observation)
    if special:
        actions.append(special)
    
    # 4. PRICING - Weekend premium when reputation allows
    day_of_week = observation.get("day_of_week", "Monday")
    reputation = observation.get("reputation_band", "Good")
    if day_of_week in ("Friday", "Saturday") and reputation in ("Good", "Very Good", "Excellent"):
        actions.extend(get_pricing_actions(observation))
    
    # 5. MARKETING - Only when cash is healthy and not in reduced capacity
    if day_of_week in ("Monday", "Tuesday") and cash > 12000 and not reduced:
        marketing = get_marketing_action(observation)
        if marketing:
            actions.append(marketing)
    
    # 6. SAVE NOTES - Persist important state
    notes_action = get_notes_action(observation, day)
    if notes_action:
        actions.append(notes_action)
    
    # Remove any None values
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
    
    return final_actions


def day_one_bulk_order_reduced(obs: dict) -> list[dict]:
    """Day 1 for reduced capacity: Order only 12kg each (half normal)."""
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
                "args": {"supplier": supplier_map[ing], "ingredient": ing, "quantity_kg": 12}
            })
    
    return orders


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print(f"Running Hybrid Agent (seed={seed})")
    print("=" * 60)
    result = run_game(strategy, team_name="The_Coding_Giraffes", seed=seed)
