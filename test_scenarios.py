"""Quick test across all scenarios."""
from agents.runner import run_game
from agents.my_agent import strategy

scenarios = ['baseline', 'renovation', 'supply_crisis', 'tourist_season']
results = {}

for s in scenarios:
    r = run_game(strategy, team_name='Giraffes', scenario=s, seed=42, verbose=False)
    score = r['score']['total_score']
    results[s] = score
    print(f'{s}: {score:.0f}')

print(f'\nTotal across scenarios: {sum(results.values()):.0f}')
print(f'Average: {sum(results.values())/len(results):.0f}')
