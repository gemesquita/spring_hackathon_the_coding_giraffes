"""Debug renovation scenario."""
from agents.my_agent import strategy, get_game_mode
from agents.runner import run_game

# Mock a Day 1 observation to test detection
test_obs = {
    "cash": 17000,
    "day_of_week": "Monday",
    "alerts": [],
    "notes": "",
    "service_summary": {"total_covers": 0}
}

mode = get_game_mode(test_obs, 1)
print(f"Day 1 detection test: mode = {mode}, cash = {test_obs['cash']}")

# Now run actual game
print("\nRunning renovation game...")
result = run_game(strategy, team_name='RenovDebug', scenario='renovation', seed=42, verbose=True)
