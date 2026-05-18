# RestBench Restaurant Simulation - Key Learnings

## What We Learned (Ranked by Impact)

### Critical Success Factors

1. **Never run out of ingredients** - Single most important rule
   - Stockout = lost revenue + reputation damage that compounds for days
   - Order 20-25kg of every ingredient on Day 1
   - Check `dishes_unavailable_at` every turn - it's the #1 signal

2. **Code-based supplier matching eliminates errors**
   - LLM frequently orders from wrong suppliers ("Fresh Farms doesn't carry Flour")
   - Build `{ingredient: best_supplier}` map in code
   - Skip suppliers mentioned in alerts (outages)

3. **Sunday has 0 covers in baseline scenario (seed 42)**
   - Set Sunday staff to minimum (5) to save ~360 EUR/week
   - This alone turned our score from -1,176 to +264

4. **Weekend pricing premium works (+10% Fri/Sat)**
   - Only when reputation is Good/Very Good/Excellent
   - Boosted score from +264 to +22,653 on baseline

5. **Delivery timing is brutal**
   - Wed-only supplier + 1-day lead + order Thursday = arrives NEXT Wednesday (6 days!)
   - Order early in week (Mon/Tue/Wed) to ensure weekend stock
   - Day multipliers: Mon=2.5x, Tue=2.0x, Wed=1.8x threshold

### Scenario-Specific Adaptations

6. **Renovation scenario (reduced capacity)**
   - Alert on Day 1: "Half your tables are unavailable for two weeks"
   - Use `save_notes` to persist "REDUCED_CAPACITY" flag
   - Reduce staff (5-8 vs 7-13)
   - Reduce Day 1 orders (12kg vs 25kg)
   - Skip marketing (fewer customers to attract anyway)

7. **Supply crisis scenario**
   - Check alerts for "halted" or "outage"
   - Skip affected suppliers in supplier map
   - Order from alternatives immediately

8. **Tourist season scenario**
   - Has surge (200-400 covers/day) then drop (50-80 covers/day)
   - Our simple strategy scored +23k but top teams hit +66k
   - Gap likely from: max staff (15) during surge, aggressive cost cuts during drop

### What Didn't Work

9. **Complex demand mode detection**
   - Tried to detect "surge" vs "drop" vs "normal" modes
   - Caused misclassification (Sunday 0 covers triggered "drop" mode)
   - Misfired staffing on normal Fridays
   - **Lesson**: Simple rules beat clever detection that can misfire

10. **Over-ordering during "detected surge"**
    - Increased waste, hurt cash flow
    - Better to have consistent ordering

### Scoring Formula Insights

11. **Penalties breakdown (typical)**
    - Walkout penalty: 700-1,500 (from understaffing)
    - Waste penalty: 150-300 (from over-ordering)
    - Reputation penalty: 0-1,600 (from stockouts)
    - Satisfaction penalty: usually 0 if consistent

12. **Leaderboard observations**
    - Top teams score 50k-66k on tourist_season seed=88
    - Top baseline score ~45k (seed 7, 88)
    - We're rank ~29 with 22,653 on baseline seed=42

## Key Code Patterns

### Supplier Map (eliminates wrong-supplier errors)
```python
def build_supplier_map(obs):
    """Returns {ingredient: best_supplier}"""
    # Skip suppliers in alerts (outages)
    # Prefer lower price, then shorter lead time
```

### Staffing by Day (with Sunday optimization)
```python
base_levels = {
    "Monday": 7, "Tuesday": 7, "Wednesday": 8,
    "Thursday": 9, "Friday": 12, "Saturday": 13, 
    "Sunday": 5  # Always low - 0 covers in baseline
}
```

### Ordering Thresholds by Day
```python
day_multipliers = {
    "Monday": 2.5, "Tuesday": 2.0, "Wednesday": 1.8,
    "Thursday": 1.5, "Friday": 1.2, "Saturday": 1.0, "Sunday": 1.0
}
# Higher early week ensures weekend stock
```

### Notes Persistence (for multi-turn state)
```python
# Day 1: Detect alert, save to notes
if "renovation" in alert_text:
    save_notes("REDUCED_CAPACITY | Day 1")

# Day 2+: Check notes since alerts are transient
if "REDUCED_CAPACITY" in notes:
    use_reduced_capacity_strategy()
```

## Performance Summary

| Version | Baseline | Tourist | Renovation | Supply | Total |
|---------|----------|---------|------------|--------|-------|
| Original LLM | -11k to -15k | - | - | - | - |
| Hybrid code-based | +264 | - | - | - | +264 |
| +Weekend pricing | +22,653 | +20,764 | -5,340 | +18,343 | +56,420 |
| +Notes persistence | +22,653 | +23,145 | +1,710 | +21,614 | **+69,122** |

## Future Improvements to Try

1. **Tourist surge optimization** - Top teams 3x our score
   - Max staff (15) when covers > 250
   - Max prices (1.2x) during surge
   - Cut to minimum (5 staff, 0.9x prices) during drop

2. **Better weekend prep**
   - Current: Day multipliers increase early-week ordering
   - Try: Explicit "weekend buffer" orders on Wednesday

3. **Happy hour on slow days**
   - Currently not using `run_happy_hour`
   - Could boost Monday/Tuesday demand

4. **Menu optimization**
   - Currently keeping default menu
   - Could disable low-margin or ingredient-heavy dishes

5. **Daily special selection**
   - Currently picking highest-stock dish
   - Could pick highest-margin dish with sufficient stock
