"""Quick test for optimized agent."""
import time
from agents.my_agent import strategy
from agents.runner import run_game

scenarios = [
    ('baseline', 42),
    ('tourist_season', 42),
]

total = 0
for scenario, seed in scenarios:
    print(f'\n=== {scenario} seed={seed} ===')
    try:
        result = run_game(strategy, team_name='OptTest', scenario=scenario, seed=seed, verbose=False)
        score = result.get("score", {}).get("total_score", 0)
        total += score
        print(f'Score: {score}')
        time.sleep(2)  # Rate limit protection
    except Exception as e:
        print(f'Error: {e}')

print(f'\n=== TOTAL: {total} ===')
