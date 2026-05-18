"""Quick test across scenarios."""
import time
from agents.my_agent import strategy
from agents.runner import run_game

scenarios = [
    ('baseline', 42),
    ('tourist_season', 42),
    ('supply_crisis', 42),
    ('renovation', 42),
]

results = []
for scenario, seed in scenarios:
    print(f'\n=== {scenario} seed={seed} ===')
    try:
        result = run_game(strategy, team_name='Test', scenario=scenario, seed=seed, verbose=False)
        score = result.get("score", {}).get("total_score", 0)
        results.append((scenario, score))
        print(f'Score: {score:.0f}')
        time.sleep(1)
    except Exception as e:
        print(f'Error: {e}')
        results.append((scenario, -100000))

print(f'\n{"="*40}')
print('SUMMARY:')
total = 0
for scenario, score in results:
    print(f'  {scenario}: {score:.0f}')
    total += score
print(f'  TOTAL: {total:.0f}')
print(f'  AVERAGE: {total/len(results):.0f}')
