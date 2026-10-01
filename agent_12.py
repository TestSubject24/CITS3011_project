import networkx as nx
import timeout_decorator

'''
WINDOWS COMPATIBILITY NOTE:
    The timeout_decorator package may not work correctly on Windows. For local
    development on Windows, you may comment out the import and all four
    @timeout_decorator.timeout(1) lines in this file. If you do so, measure the
    running time of __init__, new_game, update_game, and get_actions yourself
    (for example, with time.perf_counter). This local workaround does not relax
    the one-second limit: it is a hard constraint and will be enforced
    independently during marking.
'''
from agent_baselines import Agent

class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from agent_baselines.py first.
    
    You can add/override attributes and methods as needed.
    '''

    @timeout_decorator.timeout(1)
    def __init__(self, agent_name='Student Agent'):
        super().__init__(agent_name)

        '''Implement your agent here.'''

        self.map_graph_army = None
        self.map_graph_navy = None
        self.army_distances = {}
        self.navy_distances = {}

    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        '''Implement your agent here.'''

        self.build_map_graphs()

        # The map never changes, so there is no point doing these searches
        # again every turn.
        self.army_distances = dict(
            nx.all_pairs_shortest_path_length(self.map_graph_army)
        )
        self.navy_distances = dict(
            nx.all_pairs_shortest_path_length(self.map_graph_navy)
        )

    def build_map_graphs(self):
        if not self.game:
            raise Exception('Game Not Initialised. Cannot Build Map Graphs.')

        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        locations = list(self.game.map.loc_type.keys())

        # Add locations that armies and fleets can move through
        for loc in locations:
            if self.game.map.loc_type[loc] in ['LAND', 'COAST']:
                self.map_graph_army.add_node(loc.upper())

            if self.game.map.loc_type[loc] in ['WATER', 'COAST']:
                self.map_graph_navy.add_node(loc.upper())

        locations = [loc.upper() for loc in locations]

        # Add connections between locations
        for loc1 in locations:
            for loc2 in locations:

                if self.game.map.abuts('A', loc1, '-', loc2):
                    self.map_graph_army.add_edge(loc1, loc2)

                if self.game.map.abuts('F', loc1, '-', loc2):
                    self.map_graph_navy.add_edge(loc1, loc2)

    @timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy stratergy here.
    def update_game(self, all_power_orders):
        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    @timeout_decorator.timeout(1)
    def get_actions(self):
        possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(
            self.power_name
        )

        if self.game.phase_type == 'R':
            return self.handle_retreats(
                orderable_locations,
                possible_orders
            )

        if self.game.phase_type == 'A':
            return self.handle_adjustments(
                orderable_locations,
                possible_orders
            )

        return self.handle_movement(
            orderable_locations,
            possible_orders
        )

    def handle_retreats(self, locations, possible_orders):
        orders = []
        destinations = set()
        centres = set(self.game.map.scs)

        for loc in locations:
            choices = possible_orders[loc]
            retreats = [order for order in choices if ' R ' in order]

            # Taking an empty supply centre is useful, but two retreating units
            # are not allowed to choose the same province.
            retreats.sort(
                key=lambda order: self.province(order.split()[-1]) in centres,
                reverse=True
            )

            chosen = None

            for order in retreats:
                destination = self.province(order.split()[-1])

                if destination not in destinations:
                    chosen = order
                    destinations.add(destination)
                    break

            if chosen is None:
                chosen = next(
                    (order for order in choices if order.endswith(' D')),
                    None
                )

            if chosen is not None:
                orders.append(chosen)

        return orders

    def handle_adjustments(self, locations, possible_orders):
        units = self.game.get_units(self.power_name)
        centres = self.game.get_centers(self.power_name)
        adjustment = len(centres) - len(units)

        if adjustment < 0:
            disbands = []

            for loc in locations:
                disband = next(
                    (
                        order for order in possible_orders[loc]
                        if order.endswith(' D')
                    ),
                    None
                )

                if disband is not None:
                    disbands.append(disband)

            return disbands[:abs(adjustment)]

        if adjustment == 0:
            return []

        build_choices = {}

        for loc in locations:
            choices = possible_orders[loc]
            builds = [order for order in choices if order.endswith(' B')]

            if builds:
                build_choices[loc] = builds

        orders = []
        fleets = sum(unit.startswith('F ') for unit in units)
        armies = len(units) - fleets

        # Roughly one fleet for every two armies works well on the standard map.
        for loc, choices in build_choices.items():
            if len(orders) == adjustment:
                break

            army = next(
                (order for order in choices if order.startswith('A ')),
                None
            )
            fleet = next(
                (order for order in choices if order.startswith('F ')),
                None
            )

            if fleet is not None and fleets * 2 < armies:
                orders.append(fleet)
                fleets += 1
            elif army is not None:
                orders.append(army)
                armies += 1
            elif fleet is not None:
                orders.append(fleet)
                fleets += 1

        return orders

    def handle_movement(self, locations, possible_orders):
        my_centres = set(self.game.get_centers(self.power_name))
        targets = [
            centre for centre in self.game.map.scs
            if centre not in my_centres
        ]

        centre_owner = {}

        for power in self.game.powers:
            for centre in self.game.get_centers(power):
                centre_owner[centre] = power

        unit_owner = {}

        for power in self.game.powers:
            for unit in self.game.get_units(power):
                unit_owner[self.province(unit.split()[1])] = power

        target_values = {
            centre: self.target_value(centre, centre_owner, targets)
            for centre in targets
        }

        choices_by_loc = {}

        for loc in locations:
            choices_by_loc[loc] = self.movement_choices(
                loc,
                possible_orders[loc],
                targets,
                target_values,
                unit_owner
            )

        # Units with one clearly good route choose first. This also makes the
        # result deterministic, which is helpful when comparing experiments.
        unit_order = sorted(
            locations,
            key=lambda loc: choices_by_loc[loc][0][0]
                if choices_by_loc[loc] else -1000,
            reverse=True
        )

        plans = {}
        destinations = {}
        is_fall = self.game.get_current_phase().startswith('F')

        for loc in unit_order:
            loc_orders = possible_orders[loc]
            unit_type = self.unit_type(loc_orders)

            if unit_type is None:
                continue

            # Ownership only changes after Fall movement, so do not walk away
            # from a centre just before it can be claimed.
            if is_fall and self.province(loc) in targets:
                hold = f'{unit_type} {loc} H'

                if hold in loc_orders:
                    plans[loc] = {
                        'order': hold,
                        'score': 50,
                        'destination': None
                    }
                    continue

            chosen = None

            for score, order, destination in choices_by_loc[loc]:
                province = self.province(destination)

                if province not in destinations:
                    chosen = {
                        'order': order,
                        'score': score,
                        'destination': province
                    }
                    destinations[province] = loc
                    break

            if chosen is None:
                hold = f'{unit_type} {loc} H'

                if hold in loc_orders:
                    chosen = {
                        'order': hold,
                        'score': -100,
                        'destination': None
                    }

            if chosen is not None:
                plans[loc] = chosen

        self.add_support(plans, possible_orders, unit_owner, target_values)

        return [plans[loc]['order'] for loc in locations if loc in plans]

    def movement_choices(
        self,
        loc,
        loc_orders,
        targets,
        target_values,
        unit_owner
    ):
        unit_type = self.unit_type(loc_orders)

        if unit_type == 'A':
            distances = self.army_distances
        elif unit_type == 'F':
            distances = self.navy_distances
        else:
            return []

        choices = []

        for order in loc_orders:
            words = order.split()

            if len(words) != 4 or words[2] != '-':
                continue

            destination = words[3]
            best_score = -1000

            for centre in targets:
                distance = self.distance_to_centre(
                    destination,
                    centre,
                    distances
                )

                if distance is None:
                    continue

                score = target_values[centre] - (2 * distance)

                if self.province(destination) == centre:
                    score += 5

                if unit_owner.get(self.province(destination)) == self.power_name:
                    score -= 6

                best_score = max(best_score, score)

            if best_score > -1000:
                choices.append((best_score, order, destination))

        choices.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return choices

    def add_support(self, plans, possible_orders, unit_owner, target_values):
        attacks = []

        for loc, plan in plans.items():
            destination = plan['destination']

            if destination is None:
                continue

            enemy_there = (
                destination in unit_owner
                and unit_owner[destination] != self.power_name
            )

            enemy_nearby = any(
                owner != self.power_name
                and any(
                    len(order.split()) == 4
                    and order.split()[2] == '-'
                    and self.province(order.split()[3]) == destination
                    for order in possible_orders.get(loc, [])
                )
                for loc, owner in unit_owner.items()
            )

            if enemy_there or (
                destination in target_values
                and enemy_nearby
            ):
                attacks.append((plan['score'], loc, destination))

        attacks.sort(reverse=True)
        used_supporters = set()

        for attack_score, attacker, destination in attacks:
            if plans[attacker]['destination'] != destination:
                continue

            attack_order = plans[attacker]['order']
            candidates = []

            for loc, plan in plans.items():
                if loc == attacker or loc in used_supporters:
                    continue

                support_order = (
                    f'{self.unit_type(possible_orders[loc])} '
                    f'{loc} S {attack_order}'
                )

                if support_order in possible_orders[loc]:
                    candidates.append((plan['score'], loc, support_order))

            if not candidates:
                continue

            # Do not abandon another direct supply-centre capture just to add
            # support to a weaker attack.
            old_score, supporter, support_order = min(candidates)
            old_destination = plans[supporter]['destination']

            if (
                old_destination in target_values
                and old_score >= attack_score
            ):
                continue

            plans[supporter] = {
                'order': support_order,
                'score': attack_score,
                'destination': None
            }
            used_supporters.add(supporter)

    def target_value(self, centre, centre_owner, targets):
        owner = centre_owner.get(centre)

        # Neutral centres are usually the safest early gains. Enemy centres are
        # still valuable, but normally need another unit available to support.
        value = 14 if owner is None else 12

        neighbours = self.game.map.dest_with_coasts.get(centre, [])
        nearby_centres = {
            self.province(loc) for loc in neighbours
            if self.province(loc) in targets
        }

        return value + (2 * len(nearby_centres))

    def distance_to_centre(self, start, centre, distances):
        if start not in distances:
            return None

        centre_locations = [centre]
        centre_locations.extend(
            loc for loc in self.game.map.loc_type
            if loc.startswith(f'{centre}/')
        )
        values = [
            distances[start][loc]
            for loc in centre_locations
            if loc in distances[start]
        ]

        return min(values) if values else None

    @staticmethod
    def province(location):
        return location.split('/')[0]

    @staticmethod
    def unit_type(orders):
        if any(order.startswith('A ') for order in orders):
            return 'A'

        if any(order.startswith('F ') for order in orders):
            return 'F'

        return None
