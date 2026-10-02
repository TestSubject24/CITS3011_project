"""Reproducible local experiments; agent runtime itself performs no file I/O.

S3-proxy is a stress test, NOT the unavailable official Hidden evaluation.
For official S3 pass --scenario s3 --hidden module:Class.
"""
import argparse
import contextlib
import importlib
import io
import random
import statistics
import time

from diplomacy import Game
from agent_baselines import Agent, StaticAgent, RandomAgent, AttitudeAgent, GreedyAgent
from agent_group12 import StudentAgent
from game import copy_game


class HiddenProxy(Agent):
    """Per-game uniform mixture requested by the user, not official Hidden."""
    def __init__(self):
        super().__init__('S3 proxy')

    def new_game(self, game, power_name):
        super().new_game(game, power_name)
        self.mode = random.choice(('greedy', 'attitude', 'noisy_greedy'))
        self.delegate = AttitudeAgent() if self.mode == 'attitude' else GreedyAgent()
        self.random_agent = RandomAgent()
        self.delegate.new_game(game, power_name)
        self.random_agent.new_game(game, power_name)

    def get_actions(self):
        if self.mode == 'noisy_greedy' and random.random() < 0.1:
            return self.random_agent.get_actions()
        return self.delegate.get_actions()

    def update_game(self, all_power_orders):
        # Both delegates share this proxy's Game, so process exactly once.
        self.delegate.update_game(all_power_orders)


def experiment(scenario, games, seed, trace=False, hidden_class=None,
               start_index=0, quiet=False):
    powers = list(Game().powers)
    pool = [RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent]
    scores, raw_scores, durations, wins, survivors = [], [], [], [], []
    model_matches, predicted_orders, calls = 0, 0, 0
    own_voids = baseline_errors = exceptions = 0
    worst_decision = {}
    proxy_modes = {}
    per_power = {p: {'games': 0, 'wins': 0, 'scores': []} for p in powers}
    decision_routes = {}
    game_results = []
    movement_fallbacks = {'greedy': 0, 'safe': 0}
    depth2_searches = 0
    attack_evaluations = 0
    movement_decisions = 0
    convoy_attempts = convoy_successes = 0
    for local_index in range(games):
        index = start_index + local_index
        random.seed(seed + index)
        rng = random.Random(seed + index)
        mine = powers[index % len(powers)]
        other = [p for p in powers if p != mine]
        hidden = rng.choice(other) if scenario.startswith('s3') else None
        classes = {p: StudentAgent if p == mine else StaticAgent if scenario == 's1'
                   else (hidden_class or HiddenProxy) if p == hidden else rng.choice(pool)
                   for p in powers}
        agents = {p: cls() for p, cls in classes.items()}
        game = Game()
        for p, agent in agents.items():
            agent.new_game(copy_game(game), p)
        if scenario == 's3-proxy':
            mode = agents[hidden].mode
            proxy_modes[mode] = proxy_modes.get(mode, 0) + 1
        phase_count = 0
        game_durations = []
        game_routes = {}
        game_depth2 = 0
        game_convoys = game_convoy_successes = 0
        while not game.is_game_done and int(game.get_current_phase()[1:5]) < 1920:
            phase = game.get_current_phase()
            all_orders = {}
            # Preserve the project runner's power order; the predictor has its
            # own random stream and cannot see orders selected by other agents.
            for p, agent in agents.items():
                before = agent.game.get_state() if p == mine else None
                start = time.perf_counter()
                all_orders[p] = agent.get_actions()
                elapsed = time.perf_counter() - start
                if p == mine:
                    if not durations or elapsed > max(durations):
                        worst_decision = {'game': index, 'power': mine, 'phase': phase,
                                          'route': agent.last_decision['route']}
                    durations.append(elapsed)
                    game_durations.append(elapsed)
                    decision_phase = agent.last_decision.get('phase', game.phase_type)
                    decision_route = agent.last_decision.get('route', 'unknown')
                    key = decision_phase + ':' + decision_route
                    decision_routes[key] = decision_routes.get(key, 0) + 1
                    game_routes[key] = game_routes.get(key, 0) + 1
                    if decision_phase == 'M':
                        movement_decisions += 1
                        attack_evaluations += agent.search.last_stats.get('attack_evaluated', 0)
                        if decision_route in movement_fallbacks:
                            movement_fallbacks[decision_route] += 1
                        depth2 = agent.search.last_stats.get('depth2', 0)
                        depth2_searches += depth2
                        game_depth2 += depth2
                    assert elapsed < 0.9, ('deadline', phase, elapsed)
                    after = agent.game.get_state()
                    before.pop('timestamp'); after.pop('timestamp')
                    assert before == after, ('mutated game', phase)
                    assert agent._valid(all_orders[p], game.phase_type, agent._candidates())
                    calls += agent.opponent_model.calls
                    assert agent.opponent_model.calls <= 6
            predicted = agents[mine]._opponent_predictions
            if phase.endswith('M'):
                for p in other:
                    if classes[p] is GreedyAgent:
                        expected = set(predicted.get(p, []))
                        actual = set(all_orders[p])
                        model_matches += len(expected & actual)
                        predicted_orders += len(actual)
            for p in powers:
                game.set_orders(p, all_orders[p])
                if p == mine:
                    assert not game.error, ('our submission', phase, game.error)
                else:
                    baseline_errors += len(game.error)
                game.error = []  # Keep baseline input errors distinct from ours.
            with contextlib.redirect_stdout(io.StringIO()):
                result = game.process()
                for agent in agents.values():
                    agent.update_game(all_orders)
            for unit in game.state_history.last_value()['units'][mine]:
                count = sum(str(r) == 'void' for r in result.results.get(unit.replace('*', ''), []))
                own_voids += count
            for order in all_orders[mine]:
                if order.endswith(' VIA'):
                    words = order.split()
                    success = 'A ' + words[3] in game.powers[mine].units and not result.results.get(' '.join(words[:2]), [])
                    convoy_attempts += 1
                    convoy_successes += int(success)
                    game_convoys += 1
                    game_convoy_successes += int(success)
            phase_count += 1
        n = len(game.powers[mine].centers)
        scores.append(min(18, n)); raw_scores.append(n)
        wins.append(n >= 18); survivors.append(n > 0)
        power_stats = per_power[mine]
        power_stats['games'] += 1
        power_stats['wins'] += int(n >= 18)
        power_stats['scores'].append(min(18, n))
        exceptions += agents[mine].fallback_counts['exceptions']
        game_results.append({'index': index, 'seed': seed + index, 'power': mine,
                             'sc': min(18, n), 'raw_sc': n, 'win': n >= 18,
                             'survived': n > 0, 'final_phase': game.get_current_phase(),
                             'max_ms': max(game_durations, default=0.0) * 1000,
                             'decision_routes': game_routes, 'depth2_searches': game_depth2,
                             'convoy_attempts': game_convoys, 'convoy_successes': game_convoy_successes,
                             'fallbacks': dict(agents[mine].fallback_counts),
                             'opponents': {p: cls.__name__ for p, cls in classes.items() if p != mine}})
    for power_stats in per_power.values():
        values = power_stats.pop('scores')
        power_stats['win_rate'] = power_stats['wins'] / max(1, power_stats['games'])
        power_stats['mean_sc'] = statistics.mean(values) if values else None
    ordered = sorted(durations)
    summary = {'scenario': scenario, 'games': games, 'seed': seed, 'start_index': start_index,
               'mean_sc': statistics.mean(scores), 'raw_mean_sc': statistics.mean(raw_scores),
               'win_rate': statistics.mean(wins), 'survival_rate': statistics.mean(survivors),
               'max_ms': max(durations) * 1000,
               'worst_decision': worst_decision,
               'p99_ms': ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] * 1000,
               'greedy_exact_order_match': model_matches / max(1, predicted_orders),
               'prediction_calls': calls, 'agent_exceptions': exceptions,
               'agent_void_results': own_voids, 'baseline_input_errors': baseline_errors,
               'official_hidden': scenario == 's3', 'per_power': per_power,
               'decision_routes': decision_routes, 'movement_fallbacks': movement_fallbacks,
               'depth2_searches': depth2_searches, 'game_results': game_results}
    summary['attack_evaluations'] = attack_evaluations
    summary['attack_evaluations_per_movement'] = attack_evaluations / max(1, movement_decisions)
    summary['convoy_attempts'] = convoy_attempts
    summary['convoy_successes'] = convoy_successes
    if scenario == 's3-proxy':
        summary['label'] = '代理测试，非正式 S3，仅验证泛化性'
        summary['proxy_modes'] = proxy_modes
    print_summary(summary)
    return summary


def print_summary(summary):
    print(f"{summary['scenario'].upper()} | games={summary['games']} | seed={summary['seed']}")
    print(f"{'Power':<10} {'Games':>5} {'Mean SC':>8} {'Win rate':>9}")
    for power, stats in summary['per_power'].items():
        if stats['games']:
            print(f"{power:<10} {stats['games']:>5} {stats['mean_sc']:>8.2f} {stats['win_rate']:>8.1%}")
    print(f"{'ALL':<10} {summary['games']:>5} {summary['mean_sc']:>8.2f} {summary['win_rate']:>8.1%}")
    print(f"Max / P99 decision: {summary['max_ms']:.2f} / {summary['p99_ms']:.2f} ms")
    print(f"Agent exceptions / invalid submissions / void orders: {summary['agent_exceptions']} / 0 / {summary['agent_void_results']}")
    print(f"Movement fallback (greedy / safe): {summary['movement_fallbacks']['greedy']} / {summary['movement_fallbacks']['safe']}")
    print(f"Baseline input errors: {summary['baseline_input_errors']}")
    if 'label' in summary:
        print(summary['label'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['s1', 's2', 's3-proxy', 's3'], default='s2')
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2400)
    parser.add_argument('--quiet', action='store_true', help='Retained for compatibility; output is always a final statistics table')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Continue the seed and power rotation at this absolute game index')
    parser.add_argument('--hidden', help='Import path module:Class for the official Hidden')
    args = parser.parse_args()
    hidden_class = None
    if args.scenario == 's3':
        if not args.hidden:
            parser.error('Official S3 requires --hidden module:Class; otherwise use s3-proxy')
        module, name = args.hidden.split(':')
        hidden_class = getattr(importlib.import_module(module), name)
    if args.games <= 0 or args.start_index < 0:
        parser.error('--games must be positive and --start-index must be nonnegative')
    experiment(args.scenario, args.games, args.seed, hidden_class=hidden_class,
               start_index=args.start_index, quiet=args.quiet)
