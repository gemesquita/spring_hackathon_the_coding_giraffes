# RestBench Simulation Facts

## Game Mechanics
- 30-day simulation
- Starting cash: 15,000 EUR (17,000 in renovation)
- Staff cost: 120 EUR/person/day
- Fixed cost: 300 EUR/day
- Service hours: 11:00-22:00
- Tables: 22 (4x2-seat, 8x4-seat, 6x6-seat, 4x8-seat)

## Demand Patterns
- Sunday: 0 covers
- Monday-Tuesday: Low demand (~85-100)
- Wednesday-Thursday: Medium (~100-150)
- Friday-Saturday: High (~150-260)

## Scoring Penalties
- Walkout: Linear per walkout + negative reviews
- Reputation: Quadratic below threshold
- Waste: Penalty for excessive waste rate
- Bankruptcy: -100,000

## Scenario Signatures

### BASELINE
- Cash Day 1: 15,000
- Normal demand curve

### RENOVATION
- Cash Day 1: 17,000
- ~50% capacity for ~14 days
- Alert contains "renovation" or "capacity"

### SUPPLY_CRISIS
- Supplier halted mid-game
- Alert contains supplier name + "halted/outage"

### TOURIST_SEASON
- Days 1-8: Normal (100-200)
- Days 9-14: Surge (300-470!)
- Days 15-30: Drop (50-180)

## Supplier Behavior
- Lead time: 1-2 days typically
- Delivery windows vary by supplier
- Some don't deliver weekends

## Key Observations
- dishes_unavailable_at = #1 stockout signal
- Reputation is sticky (bad days hurt 2-3x more)
- Sunday revenue = 0
