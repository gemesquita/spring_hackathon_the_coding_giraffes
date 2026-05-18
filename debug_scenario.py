"""Debug scenario detection."""
from agents.my_agent import strategy, detect_scenario, build_notes

# Simulate Day 1 renovation observation
obs_day1 = {
    "cash": 17000,
    "day_of_week": "Monday",
    "alerts": [],
    "notes": "",
    "service_summary": {}
}

scenario = detect_scenario(obs_day1, 1)
print(f"Day 1: detected scenario = '{scenario}' (expected: renovation)")

# Simulate Day 2 after Day 1
obs_day2 = {
    "cash": 16986,
    "day_of_week": "Tuesday", 
    "alerts": [],
    "notes": build_notes(obs_day1, 1, scenario),  # Notes from Day 1
    "service_summary": {"total_covers": 92}
}

print(f"Day 1 notes: '{obs_day2['notes']}'")
scenario2 = detect_scenario(obs_day2, 2)
print(f"Day 2: detected scenario = '{scenario2}' (expected: renovation)")
