import argparse
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from agent_12 import StudentAgent
from agent_baselines import (
    AttitudeAgent,
    GreedyAgent,
    RandomAgent,
    StaticAgent,
)
from game import run_one_game


ALL_POWERS = [
    'AUSTRIA',
    'ENGLAND',
    'FRANCE',
    'GERMANY',
    'ITALY',
    'RUSSIA',
    'TURKEY',
]

SCENARIOS = {
    'static': [StaticAgent],
    'mixed': [
        RandomAgent,
        AttitudeAgent,
        AttitudeAgent,
        GreedyAgent,
        GreedyAgent,
    ],
}


class BasicGreedyAgent(StudentAgent):
    """Greedy movement without the later tactical improvements."""

    UNSUPPORTED_ATTACK_PENALTY = 0

    def target_value(self, centre, centre_owner, targets):
        return 12

    def add_support(self, plans, possible_orders, unit_owner, target_values):
        pass

    def add_convoys(
        self,
        plans,
        locations,
        possible_orders,
        targets,
        target_values,
        unit_owner,
        supportable_moves,
    ):
        pass

    def stage_convoys(
        self,
        plans,
        locations,
        possible_orders,
        target_values,
        unit_owner,
    ):
        pass


class ScoredGreedyAgent(BasicGreedyAgent):
    """Adds the supply-centre location score."""

    def target_value(self, centre, centre_owner, targets):
        return StudentAgent.target_value(
            self,
            centre,
            centre_owner,
            targets,
        )


class CoordinatedAgent(ScoredGreedyAgent):
    """Adds coordinated support to the scored greedy agent."""

    def add_support(self, plans, possible_orders, unit_owner, target_values):
        return StudentAgent.add_support(
            self,
            plans,
            possible_orders,
            unit_owner,
            target_values,
        )


class ConvoyAgent(CoordinatedAgent):
    """Adds convoy planning, but not the attack feasibility penalty."""

    def add_convoys(
        self,
        plans,
        locations,
        possible_orders,
        targets,
        target_values,
        unit_owner,
        supportable_moves,
    ):
        return StudentAgent.add_convoys(
            self,
            plans,
            locations,
            possible_orders,
            targets,
            target_values,
            unit_owner,
            supportable_moves,
        )

    def stage_convoys(
        self,
        plans,
        locations,
        possible_orders,
        target_values,
        unit_owner,
    ):
        return StudentAgent.stage_convoys(
            self,
            plans,
            locations,
            possible_orders,
            target_values,
            unit_owner,
        )


ABLATION_AGENTS = [
    ('basic greedy', BasicGreedyAgent),
    ('+ location score', ScoredGreedyAgent),
    ('+ coordinated support', CoordinatedAgent),
    ('+ convoy planning', ConvoyAgent),
    ('+ attack feasibility', StudentAgent),
]


def score_result(centres):
    score = min(centres, 18)

    if score == 18:
        outcome = 'WIN'
    elif score == 0:
        outcome = 'DEFEAT'
    else:
        outcome = 'SURVIVE'

    return score, outcome


def experiment(
    agent_type,
    opponent_pool,
    repeats,
    seed,
    label,
    fixed_opponent=None,
):
    matchup_random = random.Random(seed)
    scores = defaultdict(list)
    outcomes = defaultdict(list)
    total_games = repeats * len(ALL_POWERS)
    game_number = 0

    with tqdm(total=total_games, desc=label) as progress:
        for _ in range(repeats):
            for player_power in ALL_POWERS:
                agents = {}
                other_powers = [
                    power for power in ALL_POWERS
                    if power != player_power
                ]
                fixed_power = None

                if fixed_opponent is not None:
                    fixed_power = matchup_random.choice(other_powers)

                for power in ALL_POWERS:
                    if power == player_power:
                        agents[power] = agent_type()
                    elif power == fixed_power:
                        agents[power] = fixed_opponent()
                    else:
                        opponent_type = matchup_random.choice(opponent_pool)
                        agents[power] = opponent_type()

                # Each variant gets the same opponents and the same starting
                # random state for this game. Random play can still diverge
                # after the agents produce different board positions.
                random.seed(seed + game_number)
                centres, _ = run_one_game(agents)
                score, outcome = score_result(centres[player_power])
                scores[player_power].append(score)
                scores['ALL'].append(score)
                outcomes[player_power].append(outcome)
                outcomes['ALL'].append(outcome)
                progress.update(1)
                game_number += 1

    print(f'\n{label}')
    print('-' * len(label))

    for power in ALL_POWERS + ['ALL']:
        power_scores = scores[power]
        power_outcomes = outcomes[power]
        wins = 100 * power_outcomes.count('WIN') / len(power_outcomes)
        defeats = 100 * power_outcomes.count('DEFEAT') / len(power_outcomes)
        mean = np.mean(power_scores)
        deviation = np.std(power_scores)
        print(
            f'{power:8s}  SCs {mean:5.2f} ± {deviation:4.2f}  '
            f'Wins {wins:6.2f}%  Defeats {defeats:6.2f}%'
        )

    return {
        'mean': float(np.mean(scores['ALL'])),
        'wins': outcomes['ALL'].count('WIN') / len(outcomes['ALL']),
        'defeats': outcomes['ALL'].count('DEFEAT') / len(outcomes['ALL']),
    }


def run_final_tests(repeats, seed):
    for scenario, opponents in SCENARIOS.items():
        experiment(
            StudentAgent,
            opponents,
            repeats,
            seed,
            f'final agent - {scenario}',
        )


def run_ablation_tests(repeats, seed):
    results = []

    for label, agent_type in ABLATION_AGENTS:
        result = experiment(
            agent_type,
            SCENARIOS['mixed'],
            repeats,
            seed,
            label,
        )
        results.append((label, result))

    print('\nAblation summary (mixed opponents)')
    print('----------------------------------')

    for label, result in results:
        print(
            f'{label:23s}  SCs {result["mean"]:5.2f}  '
            f'Wins {100 * result["wins"]:6.2f}%  '
            f'Defeats {100 * result["defeats"]:6.2f}%'
        )


def run_hidden_proxy(repeats, seed):
    experiment(
        StudentAgent,
        SCENARIOS['mixed'],
        repeats,
        seed,
        'scenario 3 proxy - one strong opponent',
        fixed_opponent=StudentAgent,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run final-agent and technique-ablation experiments.'
    )
    parser.add_argument(
        '--suite',
        choices=['final', 'ablation', 'hidden-proxy', 'all'],
        default='final',
    )
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3011)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.repeats < 1:
        raise ValueError('repeats must be at least 1')

    if args.suite in ('final', 'all'):
        run_final_tests(args.repeats, args.seed)

    if args.suite in ('ablation', 'all'):
        run_ablation_tests(args.repeats, args.seed)

    if args.suite in ('hidden-proxy', 'all'):
        run_hidden_proxy(args.repeats, args.seed)


if __name__ == '__main__':
    main()
