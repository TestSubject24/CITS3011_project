"""
Simple script to test our StudentAgent against baseline GreedyAgents.
"""

from game import run_one_game
from agent_baselines import GreedyAgent
from agent_12 import StudentAgent


ALL_POWERS = [
    'AUSTRIA',
    'ENGLAND',
    'FRANCE',
    'GERMANY',
    'ITALY',
    'RUSSIA',
    'TURKEY'
]

# Change this to test our agent as another country
TEST_POWER = 'FRANCE'


def scoring(centres):
    scores = {k: min(v, 18) for k, v in centres.items()}

    wins = {}

    for k, v in scores.items():
        if v == 18:
            wins[k] = 'WIN'
        elif v == 0:
            wins[k] = 'DEFEAT'
        else:
            wins[k] = 'SURVIVE'

    return scores, wins


def main():
    print(f"Testing StudentAgent as {TEST_POWER}")
    print("Other countries are baseline GreedyAgents")
    print("=" * 60)

    agents_dict = {}

    for power in ALL_POWERS:

        if power == TEST_POWER:
            agents_dict[power] = StudentAgent()

        else:
            agents_dict[power] = GreedyAgent(
                agent_name=f'GreedyAgent_{power}'
            )

    results, year = run_one_game(
        agents_dict,
        end_year=1920
    )

    scores, wins = scoring(results)

    print(f"\nGame ended at year: {year}")
    print("-" * 60)
    print("Final Results:")
    print("-" * 60)

    for power in ALL_POWERS:

        marker = ""

        if power == TEST_POWER:
            marker = " <-- OUR AGENT"

        print(
            f"{power:10s}: "
            f"{results[power]:2d} centres | "
            f"Status: {wins[power]}"
            f"{marker}"
        )

    print("-" * 60)


if __name__ == "__main__":
    main()