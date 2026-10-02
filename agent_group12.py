"""Pruned joint-order beam search with bounded, isolated adjudication."""
import time
import random
from itertools import combinations
from collections import OrderedDict
import networkx as nx
from scipy.optimize import linear_sum_assignment
from agent_baselines import Agent
from game import copy_game  # Future simulations must use this helper.


# 1. Constants
TIME_BUDGET = 0.9
BEAM_WIDTH_INITIAL = 20
BEAM_WIDTH = 10
BEAM_WIDTH_MAX = 30
MAX_ORDERS_PER_UNIT = 5
SEARCH_SECONDS = 0.24
MAX_ROLLOUTS = 18
ATTACK_ROLLOUTS = 24
ATTACK_SEARCH_SECONDS = 0.34
MAX_ATTACK_SUPPORTERS = 2
MAX_TRANSPORT_ROLLOUTS = 18
TRANSPORT_DISTANCE = 2.0
# Geographic bonuses were disabled after regression testing. Convoy support
# and its reachability extension remain independent of these preferences.
SECTOR_BONUS = {}
WEIGHT_SC = 100.0
WEIGHT_GROWTH = 50.0
WEIGHT_THREAT = -30.0
WEIGHT_EXPANSION = 20.0
WEIGHT_BALANCE = -10.0
# Extra terms avoid abandoning spring captures and sending every unit to one SC.
WEIGHT_PENDING = 90.0
WEIGHT_COVERAGE = 80.0
WEIGHT_PRESSURE = 45.0
MEMORY_LIMIT = 96
RELATION_DECAY = 0.9
RELATION_THRESHOLD = 3.0
UNKNOWN_GREEDY = 0.5
UNKNOWN_HOLD = 0.3
FORECAST_FALLBACK_SECONDS = 0.3
PLANNING_SECONDS = 0.42


# 2. Deadline
class Deadline:
    """Use monotonic time, reserving 20 ms for safe fallback."""
    def __init__(self, budget=TIME_BUDGET):
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self):
        return max(0.0, self.budget - (time.monotonic() - self.start))

    def expired(self):
        return self.remaining() <= 0.02

    def check(self):
        if self.expired():
            raise TimeoutError('Decision deadline reached')


# 3. Evaluator
class Evaluator:
    def __init__(self, agent):
        self.agent = agent

    def features(self, game, power_name):
        power = game.powers[power_name]
        centers = set(power.centers)
        owned = {c for p in game.powers.values() for c in p.centers}
        neutral = sorted(set(game.map.scs) - owned)
        targets = sorted(set(game.map.scs) - centers)
        threat = 0
        for name, opponent in game.powers.items():
            if name != power_name:
                for unit in opponent.units:
                    kind, origin = unit.split()
                    for center in centers:
                        if game.map.abuts(kind, origin, '-', center):
                            threat += 1
        expansion = 0.0
        for unit in power.units:
            distances = [self.agent._distance(unit, c) for c in neutral]
            potential = 1.0 / (1.0 + min(distances)) if distances else 0.0
            if (power_name == 'ENGLAND' and unit.startswith('A ')
                    and self.agent.memory.history.get(power_name)
                    and not all(self.agent.opponent_model.kind(p) == 'static' for p in game.powers if p != power_name)):
                # Extend this feature only for armies isolated from foreign SCs.
                overseas = [c for c in targets if c not in power.homes]
                if overseas and all(self.agent._distance(unit, c) == float('inf') for c in overseas):
                    for center in neutral or overseas:
                        possible, _ = self.agent._can_reach_by_convoy(game, power_name, unit[2:], center)
                        if possible:
                            bonus = SECTOR_BONUS.get(power_name, {}).get(center, 1.0)
                            potential = max(potential, bonus / (1.0 + TRANSPORT_DISTANCE))
            expansion += potential
        # A distinct destination for each unit rewards dispersal. Occupied enemy
        # SCs are not immediate opportunities under the hold-opponent rollout.
        enemy_locs = {u.split()[1][:3] for n, p in game.powers.items()
                      if n != power_name for u in p.units}
        accessible = [c for c in targets if c not in enemy_locs]
        routing = self.agent._routing_distances(enemy_locs)
        coverage = 0.0
        if power.units and accessible:
            rewards = [[1.0 / (1.0 + self.agent._distance(u, c, routing))
                        for c in accessible] for u in power.units]
            rows, cols = linear_sum_assignment(rewards, maximize=True)
            coverage = sum(rewards[r][c] for r, c in zip(rows, cols))
        pending = len({u.split()[1][:3] for u in power.units}
                      & (set(game.map.scs) - centers))
        difference = len(power.units) - len(centers)
        pressure = 0.0
        for center in targets:
            if center in enemy_locs:
                distances = sorted(self.agent._distance(u, center) for u in power.units)
                if len(distances) >= 2:
                    pressure += 1.0 / (1.0 + distances[0] + distances[1])
        return {'sc_count': len(centers),
                'sc_growth': len(centers) - self.agent._last_sc_count.get(power_name, len(centers)),
                'threat_penalty': threat,
                'expansion_potential': expansion,
                'unit_sc_balance': max(0, -difference) + 0.5 * max(0, difference),
                'pending_capture': pending, 'distinct_coverage': coverage,
                'attack_pressure': pressure}

    def contributions(self, game, power_name):
        weights = {'sc_count': WEIGHT_SC, 'sc_growth': WEIGHT_GROWTH,
                   'threat_penalty': WEIGHT_THREAT,
                   'expansion_potential': WEIGHT_EXPANSION,
                   'unit_sc_balance': WEIGHT_BALANCE,
                   'pending_capture': WEIGHT_PENDING,
                   'distinct_coverage': WEIGHT_COVERAGE,
                   'attack_pressure': WEIGHT_PRESSURE}
        return {key: value * weights[key]
                for key, value in self.features(game, power_name).items()}

    def evaluate(self, game, power_name) -> float:
        """Return weighted positional utility; larger values are better."""
        return float(sum(self.contributions(game, power_name).values()))


# 4. OpponentModel (stage four)
class OpponentModel:
    """Infer behavior from public orders only; no access to opponent objects."""
    def __init__(self, memory, agent):
        self.memory, self.agent = memory, agent
        self.cache = {}
        self.greedy_reference = {}
        self.kinds = {}
        self.calls = 0
        self._phase_key = None
        self._possible_key = None
        self._possible = {}

    def begin_phase(self):
        self.cache = {}
        self.greedy_reference = {}
        self.kinds = {}
        self.calls = 0
        self._phase_key = None

    def kind(self, power):
        observed = self.memory.behavior.get(power, {})
        if observed.get('empty_streak', 0) >= 1:
            return 'static'
        if observed.get('match_weight', 0) and observed.get('match_sum', 0) / observed['match_weight'] >= 0.64:
            return 'greedy'
        if observed.get('unexplained_holds', 0) or observed.get('convoys', 0):
            return 'random'
        if observed.get('turns', 0) >= 2:
            return 'attitude'
        return 'unknown'

    def _orders(self, game, power):
        key = (id(game), game.get_current_phase(), game.get_hash())
        if self._possible_key != key:
            self._possible = (self.agent._all_orders if getattr(self.agent, '_all_orders_key', None) == key
                              else game.get_all_possible_orders())
            self._possible_key = key
        possible = self._possible
        return {loc: sorted(possible.get(loc, []))
                for loc in game.get_orderable_locations(power) if possible.get(loc)}

    def greedy(self, game, power, avoid=None):
        candidates = self._orders(game, power)
        targets = [c for c in game.map.scs if c not in game.powers[power].centers]
        selected = {}
        for loc, orders in candidates.items():
            hold = self.agent._hold(orders)
            if loc in targets:
                selected[loc] = hold
                continue
            unit = ' '.join(hold.split()[:2])
            reachable = [(self.agent._distance(unit, c), i, c) for i, c in enumerate(targets)]
            if not reachable:
                selected[loc] = hold
                continue
            distance, _, center = min(reachable)
            moves = []
            for order in orders:
                words = order.split()
                if len(words) == 4 and words[2] == '-' and (not avoid or words[3][:3] not in avoid):
                    remaining = self.agent._distance(words[0] + ' ' + words[3], center)
                    if remaining < distance:
                        moves.append((remaining, order))
            selected[loc] = min(moves)[1] if moves else hold
        # Replace same-target moves by a legal support, without fabricating text.
        attackers = {}
        for loc, order in list(selected.items()):
            words = order.split()
            if len(words) != 4 or words[2] != '-':
                continue
            target = words[3][:3]
            if target in attackers:
                leader = attackers[target].split()
                support = next((o for o in candidates[loc]
                                if len(o.split()) == 7 and o.split()[2] == 'S'
                                and o.split()[3:5] == leader[:2]
                                and o.split()[6][:3] == target), None)
                selected[loc] = support or self.agent._hold(candidates[loc])
            else:
                attackers[target] = order
        return list(selected.values())

    def predict(self, game, opponent_name, my_power_name) -> list[str]:
        key = (id(game), game.get_current_phase(), game.get_hash(), my_power_name)
        if self._phase_key != key:
            self.begin_phase()
            self._phase_key = key
        self.agent._build_distances(getattr(self.agent, '_active_deadline', None) or Deadline())
        if opponent_name in self.cache:
            return list(self.cache[opponent_name])
        self.calls += 1
        candidates = self._orders(game, opponent_name)
        kind = self.kind(opponent_name)
        self.kinds[opponent_name] = kind
        # A local, phase-specific RNG never changes a baseline's random stream.
        rng = random.Random(game.get_current_phase() + ':' + opponent_name + ':' + my_power_name)
        if game.phase_type == 'R':
            orders, destinations = [], set()
            for choices in candidates.values():
                retreat = next((o for o in choices if len(o.split()) == 4
                                and o.split()[2] == 'R' and o.split()[3][:3] not in destinations), None)
                if retreat:
                    orders.append(retreat)
                    destinations.add(retreat.split()[3][:3])
        elif game.phase_type == 'A':
            power = game.powers[opponent_name]
            diff = len(power.centers) - len(power.units)
            wanted = 'B' if diff > 0 else 'D'
            orders = []
            for choices in candidates.values():
                if len(orders) == abs(diff):
                    break
                order = next((o for o in choices if o.split()[-1] == wanted), None)
                if order:
                    orders.append(order)
            if kind in ('static', 'attitude'):
                orders = []
        else:
            greedy = self.greedy(game, opponent_name)
            self.greedy_reference[opponent_name] = greedy
            holds = [self.agent._hold(o) for o in candidates.values()]
            if kind == 'static':
                orders = holds
            elif kind == 'greedy':
                orders = greedy
            elif kind == 'attitude':
                attitude = self.memory.attitude.get(opponent_name, 'neutral')
                protected = set(game.powers[my_power_name].centers)
                protected.update(u[2:5] for u in game.powers[my_power_name].units)
                orders = self.greedy(game, opponent_name, protected if attitude == 'friendly' else None)
            elif kind == 'random':
                orders = [rng.choice(o) for o in candidates.values()]
            else:
                draw = rng.random()
                orders = (greedy if draw < UNKNOWN_GREEDY else holds
                          if draw < UNKNOWN_GREEDY + UNKNOWN_HOLD
                          else [rng.choice(o) for o in candidates.values()])
        self.cache[opponent_name] = list(orders)
        return list(orders)


# 5. OrderPruner (stage three)
class OrderPruner:
    def __init__(self, agent=None):
        self.agent = agent

    def prune(self, game, power_name, loc, orders) -> list[str]:
        if game.phase_type != 'M':
            return list(orders)
        agent = self.agent
        owned = set(game.powers[power_name].centers)
        targets = set(game.map.scs) - owned
        all_owned = {c for p in game.powers.values() for c in p.centers}
        neutral = set(game.map.scs) - all_owned
        enemy = {u.split()[1][:3] for p, state in game.powers.items()
                 if p != power_name for u in state.units}
        friends = {u for u in game.powers[power_name].units}
        ranked, holds, convoys = [], [], []
        for order in sorted(orders):
            words = order.split()
            if words[-1] == 'VIA' or (words[0] == 'F' and words[2] == 'C'):
                convoys.append(order)
                continue
            if words[-1] == 'H':
                holds.append(order)
                continue
            if len(words) == 4 and words[2] == '-':
                kind, origin, _, target = words
                if origin[:3] == target[:3]:
                    continue
                distance = min((agent._distance(kind + ' ' + target, c) for c in targets),
                               default=float('inf')) if agent else 0
                previous = min((agent._distance(kind + ' ' + origin, c) for c in neutral),
                               default=float('inf')) if agent else 0
                if target[:3] not in neutral and distance > previous + 2:
                    continue
                support = any(game.map.abuts(u[0], u[2:], 'S', target)
                              for u in friends if u[2:5] != origin[:3])
                if target[:3] in neutral:
                    priority = 100.0
                elif target[:3] in enemy:
                    priority = 90.0 if support else -10.0
                else:
                    priority = 20.0 + 20.0 / (1.0 + distance)
                ranked.append((priority, order))
            elif len(words) == 7 and words[2] == 'S' and words[5] == '-':
                # Keep cooperative support for our own units, especially attacks.
                if ' '.join(words[3:5]) in friends and words[6][:3] in targets:
                    ranked.append((85.0 if words[6][:3] in enemy else 30.0, order))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        # Usually three alternatives plus hold; MAX remains the hard ceiling.
        limit = min(3, MAX_ORDERS_PER_UNIT - bool(holds))
        kept = [order for _, order in ranked[:limit]]
        if holds:
            kept.append(holds[0])
        return kept + convoys if kept or convoys else list(orders[:1])


# 6. Search (stage three)
class Search:
    def __init__(self, evaluator, pruner):
        self.evaluator = evaluator
        self.pruner = pruner
        self.last_stats = {}
        self.top_combinations = []
        self._units = set()

    def _has_internal_conflict(self, combo):
        moves, stationary, convoyed = {}, set(), set()
        for order in combo.values():
            words = order.split()
            origin = words[1][:3]
            if len(words) in (4, 5) and words[2] == '-':
                target = words[3][:3]
                if target in moves:
                    return True
                moves[target] = origin
                if words[-1] == 'VIA':
                    convoyed.add(origin)
            else:
                stationary.add(origin)
            if words[2] in ('S', 'C') and ' '.join(words[3:5]) not in self._units:
                return True
            if words[2] in ('S', 'C'):
                beneficiary = combo.get(words[4][:3])
                if beneficiary is not None:
                    action = beneficiary.split()
                    if len(words) == 7:
                        if len(action) not in (4, 5) or action[2] != '-' or action[3][:3] != words[6][:3]:
                            return True
                        if words[2] == 'C' and action[-1] != 'VIA':
                            return True
                    elif action[2] == '-':
                        return True
        if set(moves) & stationary:
            return True
        # Non-convoy head-to-head swaps cannot succeed.
        return any(moves.get(origin) == target and origin not in convoyed and target not in convoyed
                   for target, origin in moves.items())

    def _projected_score(self, combo, game, power_name, targets, neutral, enemy, routing):
        agent = self.evaluator.agent
        supports = set()
        for order in combo.values():
            words = order.split()
            if len(words) == 7 and words[2] == 'S':
                supports.add((' '.join(words[3:5]), words[6][:3]))
        positions = []
        for unit in game.powers[power_name].units:
            kind, origin = unit.split()
            words = combo.get(origin[:3], unit + ' H').split()
            destination = origin
            if len(words) == 4 and words[2] == '-':
                if words[3][:3] not in enemy or (unit, words[3][:3]) in supports:
                    destination = words[3]
            positions.append(kind + ' ' + destination)
        occupied = {u[2:5] for u in positions}
        capture = len(occupied & targets)
        score = (WEIGHT_PENDING + (50.0 if game.get_current_phase().startswith('F') else 0)) * capture
        for unit in positions:
            d = min((agent._distance(unit, c) for c in neutral), default=float('inf'))
            score += WEIGHT_EXPANSION / (1.0 + d)
        accessible = sorted(targets - enemy)
        if accessible and positions:
            rewards = [[1.0 / (1.0 + agent._distance(u, c, routing))
                        for c in accessible] for u in positions]
            rows, cols = linear_sum_assignment(rewards, maximize=True)
            score += WEIGHT_COVERAGE * sum(rewards[r][c] for r, c in zip(rows, cols))
        for target in targets & enemy:
            ds = sorted(agent._distance(u, target) for u in positions)
            if len(ds) >= 2:
                score += WEIGHT_PRESSURE / (1.0 + ds[0] + ds[1])
        return score

    def _attack_combinations(self, incumbent, raw, game, power_name, deadline):
        """Try coordinated attacks omitted by the per-unit candidate cap."""
        centers = set(game.map.scs) - set(game.powers[power_name].centers)
        enemy = {u[2:5] for p, state in game.powers.items() if p != power_name for u in state.units}
        history = self.evaluator.agent.memory.sc_history.get(power_name, [])
        stalled = len(history) >= 3 and len(game.powers[power_name].centers) <= history[-3]
        attack_targets = (enemy - set(game.powers[power_name].centers)) if stalled else centers & enemy
        moves, helpers = [], {}
        for loc, orders in raw.items():
            deadline.check()
            for order in orders:
                w = order.split()
                if len(w) == 4 and w[2] == '-' and w[3][:3] in attack_targets:
                    moves.append(order)
                elif len(w) == 7 and w[2] == 'S' and ' '.join(w[3:5]) in self._units:
                    helpers.setdefault((' '.join(w[3:5]), w[6][:3]), []).append((loc, order))
        base = {o.split()[1][:3]: o for o in incumbent}
        # Centers first, then occupied routes such as ENG/MAO/BUR. A supported
        # breakthrough can open access to several centers on following turns.
        for move in sorted(moves, key=lambda o: (o.split()[3][:3] not in centers, o)):
            deadline.check()
            w = move.split()
            origin, target = w[1][:3], w[3][:3]
            options = sorted(helpers.get((' '.join(w[:2]), target), []))
            # More than one support can break an opponent's supported hold.
            for count in range(1, min(MAX_ATTACK_SUPPORTERS, len(options)) + 1):
                for support in combinations(options, count):
                    deadline.check()
                    trial = dict(base)
                    trial[origin] = move
                    for loc, order in support:
                        trial[loc] = order
                    # Remove obsolete supports for orders changed by this package.
                    for loc, order in list(trial.items()):
                        words = order.split()
                        if len(words) == 7 and words[2] == 'S':
                            beneficiary = trial.get(words[4][:3], '').split()
                            if len(beneficiary) != 4 or beneficiary[2] != '-' or beneficiary[3][:3] != words[6][:3]:
                                trial[loc] = self.evaluator.agent._hold(raw[loc])
                    if not self._has_internal_conflict(trial):
                        yield list(trial.values())

    def search(self, game, power_name, deadline):
        if game.phase_type != 'M' or deadline.expired():
            return None
        agent = self.evaluator.agent
        started = time.monotonic()
        self.last_stats = {'expanded': 0, 'evaluated': 0, 'raw': 0, 'pruned': 0, 'units': 0}
        self.top_combinations = []
        ranked = []
        try:
            agent._build_distances(deadline)
            raw = agent._decision_candidates
            self._units = set(game.powers[power_name].units)
            choices = {}
            for loc, orders in raw.items():
                deadline.check()
                choices[loc] = self.pruner.prune(game, power_name, loc, orders)
            self.last_stats.update(raw=sum(map(len, raw.values())),
                                   pruned=sum(map(len, choices.values())), units=len(choices))
            # Keep the ordinary beam unchanged. Protected convoy orders are
            # explored below as complete, matched army/fleet combinations.
            convoy_candidates = choices
            choices = {loc: [o for o in orders if o.split()[-1] != 'VIA' and o.split()[2] != 'C']
                       for loc, orders in choices.items()}
            holds = {loc: agent._hold(orders) for loc, orders in raw.items()}
            targets = set(game.map.scs) - set(game.powers[power_name].centers)
            neutral = set(game.map.scs) - {c for p in game.powers.values() for c in p.centers}
            enemy = {u[2:5] for p, state in game.powers.items() if p != power_name for u in state.units}
            routing = agent._routing_distances(enemy)
            # Index legal move/support pairs. They enter the beam together so a
            # temporarily unsupported attack is not discarded before its helper.
            helpers = {}
            for loc, orders in choices.items():
                for order in orders:
                    words = order.split()
                    if len(words) == 7 and words[2] == 'S':
                        key = (' '.join(words[3:5]), words[6][:3])
                        helpers.setdefault(key, []).append((loc, order))
            beam = [(0.0, {})]
            hold_prefix = {}
            for loc in choices:
                deadline.check()
                width = BEAM_WIDTH_MAX if deadline.remaining() > 0.5 else (BEAM_WIDTH if deadline.remaining() > 0.2 else 3)
                width = min(width, max(3, 100 // max(1, len(choices))))
                expanded = {}
                for _, combo in beam:
                    deadline.check()
                    alternatives = [combo] if loc in combo else []
                    if loc not in combo:
                        for order in choices[loc]:
                            deadline.check()
                            trial = dict(combo)
                            trial[loc] = order
                            alternatives.append(trial)
                            words = order.split()
                            if len(words) == 4 and words[2] == '-' and words[3][:3] in enemy:
                                for helper, support in helpers.get((' '.join(words[:2]), words[3][:3]), []):
                                    if helper not in combo:
                                        pair = dict(trial)
                                        pair[helper] = support
                                        alternatives.append(pair)
                    for trial in alternatives:
                        deadline.check()
                        self.last_stats['expanded'] += 1
                        if self._has_internal_conflict(trial):
                            continue
                        key = tuple(sorted(trial.items()))
                        if key not in expanded:
                            expanded[key] = (self._projected_score(trial, game, power_name,
                                                                  targets, neutral, enemy, routing), trial)
                # Preserve one completable legal path at every layer. Otherwise
                # incompatible support commitments can empty a narrow beam.
                hold_prefix[loc] = holds[loc]
                safe_entry = (self._projected_score(hold_prefix, game, power_name,
                                                    targets, neutral, enemy, routing), dict(hold_prefix))
                beam = sorted(expanded.values(), key=lambda item: -item[0])[:max(1, width - 1)]
                if not any(combo == hold_prefix for _, combo in beam):
                    beam.append(safe_entry)
            # Only complete combinations reach the expensive engine. All holds
            # is an explicit incumbent so heuristic errors cannot force a loss.
            finalists = [holds] + [combo for _, combo in beam[:MAX_ROLLOUTS - 1]]
            ranked, seen = [], set()
            for combo in finalists:
                deadline.check()
                if ranked and time.monotonic() - started > SEARCH_SECONDS:
                    break
                key = tuple(sorted(combo.values()))
                if key in seen:
                    continue
                seen.add(key)
                sim = agent._simulate(list(combo.values()), deadline)
                score = self.evaluator.evaluate(sim, power_name)
                self.last_stats['evaluated'] += 1
                ranked.append((score, list(combo.values())))
            ranked.sort(key=lambda item: -item[0])
            # Keep the original beam incumbent; spend spare time only on
            # additional offensive combinations. Known Static games retain
            # the already verified baseline search.
            if ranked and not all(agent.opponent_model.kind(p) == 'static' for p in game.powers if p != power_name):
                attack_evaluated = 0
                for orders in self._attack_combinations(ranked[0][1], raw, game, power_name, deadline):
                    if (attack_evaluated >= ATTACK_ROLLOUTS or deadline.remaining() < 0.5
                            or time.monotonic() - started > ATTACK_SEARCH_SECONDS):
                        break
                    key = tuple(sorted(orders))
                    if key in seen:
                        continue
                    seen.add(key)
                    sim = agent._simulate(orders, deadline)
                    score = self.evaluator.evaluate(sim, power_name)
                    ranked.append((score, orders))
                    attack_evaluated += 1
                self.last_stats['attack_evaluated'] = attack_evaluated
                self.last_stats['evaluated'] += attack_evaluated
                ranked.sort(key=lambda item: -item[0])
            if (ranked and power_name == 'ENGLAND'
                    and not all(agent.opponent_model.kind(p) == 'static' for p in game.powers if p != power_name)):
                convoy_evaluated = 0
                for orders in self._convoy_combinations(ranked[0][1], convoy_candidates, raw, game, power_name, deadline):
                    if (convoy_evaluated >= MAX_TRANSPORT_ROLLOUTS or deadline.remaining() < 0.5
                            or time.monotonic() - started > ATTACK_SEARCH_SECONDS):
                        break
                    key = tuple(sorted(orders))
                    if key in seen:
                        continue
                    seen.add(key)
                    sim = agent._simulate(orders, deadline)
                    ranked.append((self.evaluator.evaluate(sim, power_name), orders))
                    convoy_evaluated += 1
                self.last_stats['convoy_evaluated'] = convoy_evaluated
                self.last_stats['evaluated'] += convoy_evaluated
                ranked.sort(key=lambda item: -item[0])
            self.top_combinations = ranked[:3]
            return ranked[0][1] if ranked else None
        except TimeoutError:
            if ranked:
                ranked.sort(key=lambda item: -item[0])
                self.top_combinations = ranked[:3]
                return ranked[0][1]
            return None
        except Exception as exc:
            self.last_stats['error'] = type(exc).__name__
            agent._record_failure(exc, [])
            return None


    def _convoy_combinations(self, incumbent, protected, raw, game, power_name, deadline):
        agent = self.evaluator.agent
        targets = set(game.map.scs) - set(game.powers[power_name].centers)
        overseas = targets - set(game.powers[power_name].homes)
        proposals = []
        for loc, orders in protected.items():
            deadline.check()
            for order in orders:
                words = order.split()
                if words[-1] != 'VIA' or words[0] != 'A':
                    continue
                if not overseas or any(agent._distance(' '.join(words[:2]), c) < float('inf') for c in overseas):
                    continue
                destination = words[3][:3]
                # Never use a scarce convoy to return to the home landmass.
                if any(agent._distance('A ' + home, destination) < float('inf')
                       for home in game.powers[power_name].homes):
                    continue
                possible, _ = agent._can_reach_by_convoy(game, power_name, loc, destination)
                if not possible:
                    continue
                bonus = SECTOR_BONUS.get(power_name, {}).get(destination, 1.0)
                priority = (destination in targets, bonus, order)
                for water, fleet_orders in protected.items():
                    if game.map.loc_type.get(water) != 'WATER':
                        continue
                    adjacent = {p.upper().split('/')[0] for p in game.map.abut_list(water)}
                    if loc not in adjacent or destination not in adjacent:
                        continue
                    transport = 'F ' + water + ' C A ' + words[1] + ' - ' + words[3]
                    if transport in fleet_orders:
                        proposals.append((priority, loc, order, water, transport))
        base = {o.split()[1][:3]: o for o in incumbent}
        for _, army_loc, move, water, transport in sorted(proposals, reverse=True):
            deadline.check()
            trial = dict(base)
            trial[army_loc], trial[water] = move, transport
            # Changing an attack into a convoy can make an old support stale.
            for loc, order in list(trial.items()):
                w = order.split()
                if len(w) == 7 and w[2] == 'S':
                    recipient = trial.get(w[4][:3], '').split()
                    if len(recipient) not in (4, 5) or recipient[2] != '-' or recipient[3][:3] != w[6][:3]:
                        trial[loc] = agent._hold(raw[loc])
            if not self._has_internal_conflict(trial):
                yield list(trial.values())
            # A landing or its carrier may need support. All variants still
            # contain the complete convoy pair and use engine-listed orders.
            moving = move.split()
            for helper, orders in raw.items():
                deadline.check()
                if helper in (army_loc, water):
                    continue
                for support in orders:
                    w = support.split()
                    landing = (len(w) == 7 and w[2] == 'S' and w[3:5] == moving[:2]
                               and w[6][:3] == moving[3][:3])
                    escort = len(w) == 5 and w[2] == 'S' and w[3:5] == ['F', water]
                    if landing or escort:
                        reinforced = dict(trial)
                        reinforced[helper] = support
                        if not self._has_internal_conflict(reinforced):
                            yield list(reinforced.values())


# 7. Memory (stage four)
class Memory:
    def __init__(self):
        self.history = {}
        self.attitude = {}
        self.sc_history = {}
        self._sc_years = {}
        self.last_orders = {}
        self.friend_score = {}
        self.hostile_score = {}
        self.behavior = {}
        self.events = {}
        self.my_power = None
        self.previous_units = {}
        self.previous_centers = {}
        self.greedy_reference = {}

    def capture(self, game, my_power, greedy_reference=None):
        self.my_power = my_power
        self.previous_units = {p: list(s.units) for p, s in game.powers.items()}
        self.previous_centers = {p: list(s.centers) for p, s in game.powers.items()}
        self.greedy_reference = greedy_reference or {}
        for p in game.powers:
            self.attitude.setdefault(p, 'neutral')
            self.friend_score.setdefault(p, 0.0)
            self.hostile_score.setdefault(p, 0.0)

    @staticmethod
    def _canonical(order):
        return ' '.join(word.split('/')[0] for word in order.split())

    def record_turn(self, phase, all_power_orders, game):
        old_units = set(self.previous_units.get(self.my_power, []))
        old_locations = {u[2:5] for u in old_units}
        dislodged = set(game.powers[self.my_power].retreats)
        self.events = {}
        for power, state in game.powers.items():
            orders = list(all_power_orders.get(power, []))
            self.history.setdefault(power, []).append((phase, orders))
            self.history[power] = self.history[power][-MEMORY_LIMIT:]
            self.last_orders[power] = orders
            # Keep one SC count per year, replacing it as the year progresses.
            year = phase[1:5]
            yearly = self.sc_history.setdefault(power, [])
            if yearly and self._sc_years.get(power) == year:
                yearly[-1] = len(state.centers)
            else:
                yearly.append(len(state.centers))
            self._sc_years[power] = year
            self.sc_history[power] = yearly[-MEMORY_LIMIT:]
            if not phase.endswith('M') or power == self.my_power:
                continue
            self.friend_score[power] *= RELATION_DECAY
            self.hostile_score[power] *= RELATION_DECAY
            statuses = game.get_order_status(power_name=power)
            attacks = supports = successful_attacks = successful_supports = 0
            for order in orders:
                words = order.split()
                if len(words) < 3:
                    continue
                unit = ' '.join(words[:2])
                if words[2] == '-' and len(words) >= 4 and words[3][:3] in old_locations:
                    attacks += 1
                    if not statuses.get(unit, []) and any(u[2:5] == words[3][:3] for u in dislodged):
                        successful_attacks += 1
                elif words[2] == 'S' and len(words) >= 5:
                    supported = ' '.join(words[3:5])
                    actual_unit = next((u for u in old_units if self._canonical(u) == self._canonical(supported)), None)
                    if actual_unit is not None:
                        supports += 1
                        if unit in statuses and not statuses[unit] and not game.get_order_status(unit=actual_unit):
                            successful_supports += 1
            self.hostile_score[power] += 2 * attacks
            self.friend_score[power] += 2 * supports
            difference = self.hostile_score[power] - self.friend_score[power]
            self.attitude[power] = ('hostile' if difference > RELATION_THRESHOLD else
                                    'friendly' if difference < -RELATION_THRESHOLD else 'neutral')
            self.events[power] = {'attacks': attacks, 'supports': supports,
                                  'successful_attacks': successful_attacks,
                                  'successful_supports': successful_supports}
            behavior = self.behavior.setdefault(power, {})
            behavior['turns'] = behavior.get('turns', 0) + 1
            if self.previous_units.get(power):
                behavior['empty_streak'] = behavior.get('empty_streak', 0) + 1 if not orders else 0
            expected = {self._canonical(o) for o in self.greedy_reference.get(power, [])}
            actual = {self._canonical(o) for o in orders}
            if expected and actual:
                behavior['match_sum'] = 0.85 * behavior.get('match_sum', 0) + len(actual & expected) / len(actual)
                behavior['match_weight'] = 0.85 * behavior.get('match_weight', 0) + 1
                behavior['unexplained_holds'] = behavior.get('unexplained_holds', 0) + sum(
                    o.endswith(' H') and o not in expected for o in actual)
                behavior['convoys'] = behavior.get('convoys', 0) + sum(' C ' in o or ' VIA' in o for o in actual)


# 8. Agent
class GroupXAgent(Agent):
    def __init__(self, agent_name):
        super().__init__(agent_name)
        self.evaluator = Evaluator(self)
        self._last_sc_count = {}
        self.candidate_scores = {}
        self.order_pruner = OrderPruner(self)
        self.search = Search(self.evaluator, self.order_pruner)
        self.memory = Memory()
        self.opponent_model = OpponentModel(self.memory, self)
        self._dist_cache = None
        self._graphs = None
        self._routing_key = None
        self._routing_cache = None
        self._routing_pool = OrderedDict()
        self._distance_memo = {}
        self.last_decision = {}
        self.fallback_counts = dict.fromkeys(
            ('greedy', 'safe', 'exceptions', 'timeouts'), 0)

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self._last_sc_count = {p: len(state.centers) for p, state in game.powers.items()}
        self._dist_cache = None
        self._graphs = None
        self._routing_key = None
        self._routing_cache = None
        self._routing_pool = OrderedDict()
        self._distance_memo = {}
        self.memory = Memory()
        self.memory.capture(game, power_name)
        self.opponent_model = OpponentModel(self.memory, self)
        self.last_decision = {}
        self.fallback_counts = dict.fromkeys(self.fallback_counts, 0)

    def update_game(self, all_power_orders):
        phase = self.game.get_current_phase()
        self.memory.capture(self.game, self.power_name, self.opponent_model.greedy_reference)
        self._last_sc_count = {p: len(state.centers) for p, state in self.game.powers.items()}
        super().update_game(all_power_orders)
        self.memory.record_turn(phase, all_power_orders, self.game)

    def _candidates(self):
        locations = self.game.get_orderable_locations(self.power_name)
        all_orders = self.game.get_all_possible_orders()
        self._all_orders = all_orders
        self._all_orders_key = (id(self.game), self.game.get_current_phase(), self.game.get_hash())
        return {loc: sorted(all_orders[loc]) for loc in locations
                if all_orders.get(loc)}

    def _build_distances(self, deadline):
        if self._dist_cache is not None:
            return
        board = self.game.map
        graphs, distances = {}, {}
        for kind in ('A', 'F'):
            deadline.check()
            graph = nx.DiGraph()
            for raw_loc in board.locs:
                deadline.check()
                loc = raw_loc.upper()
                if board.is_valid_unit(kind + ' ' + loc):
                    graph.add_node(loc)
            for loc in list(graph):
                deadline.check()
                for adjacent in board.abut_list(loc, incl_no_coast=True):
                    deadline.check()
                    target = adjacent.upper()
                    if target in graph and board.abuts(kind, loc, '-', target):
                        graph.add_edge(loc, target)
            lengths = {}
            for source in graph:
                deadline.check()
                lengths[source] = dict(nx.single_source_shortest_path_length(graph, source))
            graphs[kind], distances[kind] = graph, lengths
        deadline.check()
        # Publish only complete caches. A timed-out build may be retried.
        self._graphs, self._dist_cache = graphs, distances

    @staticmethod
    def _hold(orders):
        return next((o for o in orders if o.split()[-1] == 'H'),
                    orders[0] if orders else None)

    def _diff(self):
        power = self.game.powers[self.power_name]
        return len(power.centers) - len(power.units)

    def _routing_distances(self, enemy_locs):
        if self._dist_cache is None:
            self._build_distances(Deadline())
        key = frozenset(enemy_locs)
        if key != self._routing_key:
            routing = self._routing_pool.get(key)
            if routing is None:
                routing = {}
                for kind, graph in self._graphs.items():
                    available = graph.copy()
                    available.remove_nodes_from([n for n in graph if n[:3] in key])
                    routing[kind] = dict(nx.all_pairs_shortest_path_length(available))
                self._routing_pool[key] = routing
                if len(self._routing_pool) > 8:
                    self._routing_pool.popitem(last=False)
            self._routing_pool.move_to_end(key)
            self._routing_key, self._routing_cache = key, routing
        return self._routing_cache

    def _distance(self, unit, center, cache=None):
        if self._dist_cache is None:
            self._build_distances(Deadline())
        key = (unit, center, None if cache is None else self._routing_key)
        if key in self._distance_memo:
            return self._distance_memo[key]
        kind, origin = unit.split()
        lengths = (self._dist_cache if cache is None else cache)[kind].get(origin, {})
        value = min((lengths.get(c, float('inf'))
                     for c in self.game.map.find_coasts(center)), default=float('inf'))
        if len(self._distance_memo) > 50000:
            self._distance_memo.clear()
        self._distance_memo[key] = value
        return value

    def _can_reach_by_convoy(self, game, power_name, unit_loc, target_sc):
        """Return one of our sea fleets directly adjacent to both coasts."""
        origin, target = unit_loc.upper().split('/')[0], target_sc.upper().split('/')[0]
        if origin == target or game.map.loc_type.get(origin) != 'COAST' or game.map.loc_type.get(target) != 'COAST':
            return False, None
        for unit in sorted(game.powers[power_name].units):
            kind, loc = unit.split()
            if kind != 'F' or game.map.loc_type.get(loc) != 'WATER':
                continue
            adjacent = {p.upper().split('/')[0] for p in game.map.abut_list(loc)}
            if origin in adjacent and target in adjacent:
                return True, loc
        return False, None

    def _simulate(self, orders, deadline):
        deadline.check()
        # Reserve enough time to finish one non-interruptible engine call.
        if deadline.remaining() < max(0.08, 2.0 * self._rollout_max):
            raise TimeoutError('Insufficient time for another rollout')
        if time.monotonic() - self._decision_started > PLANNING_SECONDS - max(0.04, 2.0 * self._rollout_max):
            raise TimeoutError('Planning work limit reached')
        started = time.monotonic()
        if self._rollout_base is None:
            self._rollout_base = copy_game(self.game)
            # Histories are not needed for one-phase order adjudication.
            # Clear them on the private copy, never on the live Game.
            self._rollout_base.set_state(self._rollout_base.get_state(), clear_history=True)
        sim = copy_game(self._rollout_base)
        deadline.check()
        sim.clear_orders()
        sim.set_orders(self.power_name, orders)
        for opponent, prediction in self._opponent_predictions.items():
            deadline.check()
            # At low time retain only confident greedy/static predictions.
            if deadline.remaining() < FORECAST_FALLBACK_SECONDS and self.opponent_model.kinds.get(opponent) not in ('greedy', 'static'):
                continue
            sim.set_orders(opponent, prediction)
        if sim.error:
            raise ValueError('Simulation rejected orders')
        sim.process()
        self._rollout_max = max(self._rollout_max, time.monotonic() - started)
        if sim.error:
            raise ValueError('Simulation rejected orders')
        deadline.check()
        return sim

    def _movement(self, candidates, deadline):
        self._build_distances(deadline)
        holds = {loc: self._hold(choices) for loc, choices in candidates.items()}
        best = dict(holds)
        self.candidate_scores = {}
        try:
            base = self._simulate(list(holds.values()), deadline)
            hold_score = self.evaluator.evaluate(base, self.power_name)
            # Required isolated rollouts: exactly one unit moves, all others hold.
            for loc, choices in candidates.items():
                deadline.check()
                best_score = hold_score
                self.candidate_scores[loc] = {holds[loc]: hold_score}
                for order in choices:
                    deadline.check()
                    words = order.split()
                    if len(words) != 4 or words[2] != '-':
                        continue
                    trial = dict(holds)
                    trial[loc] = order
                    sim = self._simulate(list(trial.values()), deadline)
                    score = self.evaluator.evaluate(sim, self.power_name)
                    self.candidate_scores[loc][order] = score
                    if score > best_score + 1e-8:
                        best_score, best[loc] = score, order
            # Reconcile independently chosen moves using actual joint outcomes.
            # This prevents self-bounces and assigns different expansion targets.
            score = self.evaluator.evaluate(
                self._simulate(list(best.values()), deadline), self.power_name)
            for loc, choices in candidates.items():
                deadline.check()
                for order in choices:
                    deadline.check()
                    words = order.split()
                    if not (words[-1] == 'H' or (len(words) == 4 and words[2] == '-')):
                        continue
                    if order == best[loc]:
                        continue
                    trial = dict(best)
                    trial[loc] = order
                    value = self.evaluator.evaluate(
                        self._simulate(list(trial.values()), deadline), self.power_name)
                    if value > score + 1e-8:
                        best, score = trial, value
        except TimeoutError:
            self.fallback_counts['timeouts'] += 1
        return list(best.values())

    def _adjust_evaluated(self, candidates, deadline):
        self._build_distances(deadline)
        fallback = self._adjust(candidates)
        best = list(fallback)
        def reach(sim):
            targets = set(sim.map.scs) - set(sim.powers[self.power_name].centers)
            return sum(1.0 / (1.0 + min((self._distance(unit, c) for c in targets),
                                       default=float('inf')))
                       for unit in sim.powers[self.power_name].units)
        try:
            initial = self._simulate(best, deadline)
            score = self.evaluator.evaluate(initial, self.power_name)
            best_reach = reach(initial)
            # Coordinate improvement keeps the number of builds/disbands fixed.
            for index in range(len(best)):
                deadline.check()
                for choices in candidates.values():
                    deadline.check()
                    for order in choices:
                        deadline.check()
                        if order == 'WAIVE' or order.split()[-1] != best[index].split()[-1]:
                            continue
                        trial = list(best)
                        trial[index] = order
                        if not self._valid(trial, 'A', candidates):
                            continue
                        sim = self._simulate(trial, deadline)
                        value = self.evaluator.evaluate(sim, self.power_name)
                        trial_reach = reach(sim)
                        if value > score + 1e-8 or (abs(value - score) <= 1e-8 and trial_reach > best_reach + 1e-8):
                            best, score = trial, value
                            best_reach = trial_reach
        except TimeoutError:
            self.fallback_counts['timeouts'] += 1
        return best

    def _retreat_evaluated(self, candidates, deadline):
        if not candidates:
            return []
        self._build_distances(deadline)
        chosen, destinations = {}, set()
        for loc, choices in candidates.items():
            for order in choices:
                words = order.split()
                if len(words) == 4 and words[2] == 'R' and words[3][:3] not in destinations:
                    chosen[loc] = order
                    destinations.add(words[3][:3])
                    break
        try:
            score = self.evaluator.evaluate(
                self._simulate(list(chosen.values()), deadline), self.power_name)
            for loc, choices in candidates.items():
                deadline.check()
                for order in choices:
                    deadline.check()
                    words = order.split()
                    if len(words) != 4 or words[2] != 'R':
                        continue
                    if any(o.split()[3][:3] == words[3][:3] for l, o in chosen.items() if l != loc):
                        continue
                    trial = dict(chosen)
                    trial[loc] = order
                    value = self.evaluator.evaluate(
                        self._simulate(list(trial.values()), deadline), self.power_name)
                    if value > score + 1e-8:
                        chosen, score = trial, value
        except TimeoutError:
            self.fallback_counts['timeouts'] += 1
        return list(chosen.values())

    def _greedy(self, phase, candidates, deadline):
        deadline.check()
        if phase == 'M':
            return self._movement(candidates, deadline)
        if phase == 'R':
            return self._retreat_evaluated(candidates, deadline)
        if phase == 'A':
            return self._adjust_evaluated(candidates, deadline)
        return []

    def _adjust(self, candidates, safe=False, deadline=None):
        diff = self._diff()
        if safe and diff > 0:
            return ['WAIVE'] * diff
        wanted = 'B' if diff > 0 else 'D'
        orders = []
        for choices in candidates.values():
            if deadline is not None:
                deadline.check()
            if len(orders) >= abs(diff):
                break
            order = next((o for o in choices if o.split()[-1] == wanted), None)
            if order is not None:
                orders.append(order)
        return orders

    def _safe(self, phase, candidates):
        if phase == 'M':
            return [self._hold(choices) for choices in candidates.values() if choices]
        if phase == 'A':
            return self._adjust(candidates, safe=True)
        return []

    def _valid(self, orders, phase, candidates):
        """Reject malformed search output and duplicate unit/build orders."""
        if not isinstance(orders, list) or any(not isinstance(o, str) for o in orders):
            return False
        legal = {o for choices in candidates.values() for o in choices}
        seen = set()
        for order in orders:
            if order == 'WAIVE':
                if phase != 'A' or self._diff() <= 0:
                    return False
                continue
            if order not in legal:
                return False
            location = order.split()[1][:3]
            if location in seen:
                return False
            seen.add(location)
        return phase != 'A' or len(orders) <= abs(self._diff())

    def get_actions(self) -> list[str]:
        deadline = Deadline()
        self._active_deadline = deadline
        self._decision_started = time.monotonic()
        self._rollout_base = None
        self._rollout_max = 0.025
        self._opponent_predictions = {}
        self.opponent_model.begin_phase()
        phase, candidates, orders = None, None, []
        route, reasons = 'safe', []
        try:
            phase = self.game.phase_type
            candidates = self._candidates()
            self._decision_candidates = candidates
            self._build_distances(deadline)
            for opponent in self.game.powers:
                deadline.check()
                if opponent != self.power_name:
                    self._opponent_predictions[opponent] = self.opponent_model.predict(self.game, opponent, self.power_name)
            try:
                deadline.check()
                result = self.search.search(self.game, self.power_name, deadline)
                if result is not None and self._valid(result, phase, candidates):
                    orders, route = result, 'search'
                else:
                    reasons.append('search_none' if result is None else 'invalid_search')
            except Exception as exc:
                self._record_failure(exc, reasons)
            if route != 'search':
                orders = self._greedy(phase, candidates, deadline)
                if not self._valid(orders, phase, candidates):
                    raise ValueError('Invalid greedy orders')
                route = 'greedy'
        except Exception as exc:
            self._record_failure(exc, reasons)
            route = 'safe'
            try:
                phase = self.game.phase_type
                if candidates is None:
                    candidates = self._candidates()
                orders = self._safe(phase, candidates)
                if not self._valid(orders, phase, candidates):
                    orders = []
            except Exception as fallback_exc:
                self._record_failure(fallback_exc, reasons)
                orders = []  # Legal omission; engine supplies phase defaults.
        if route in self.fallback_counts:
            self.fallback_counts[route] += 1
        self.last_decision = {'phase': phase, 'route': route, 'reasons': reasons,
                              'seconds': time.monotonic() - deadline.start}
        return orders

    def _record_failure(self, exc, reasons):
        key = 'timeouts' if isinstance(exc, TimeoutError) else 'exceptions'
        self.fallback_counts[key] += 1
        reasons.append(type(exc).__name__)


# test.py imports StudentAgent and constructs it without arguments.
class StudentAgent(GroupXAgent):
    def __init__(self, agent_name='Group 12 Agent'):
        super().__init__(agent_name)
