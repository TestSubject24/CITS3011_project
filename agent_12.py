import time
import random
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
    def __init__(self, agent_name='Greedy Agent'):
        super().__init__(agent_name)

        '''Implement your agent here.'''

        self.map_graph_army = None
        self.map_graph_navy = None

    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        '''Implement your agent here.'''

        self.build_map_graphs()

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

        '''Implement your agent here.'''

        possible_orders = self.game.get_all_possible_orders()

        orderable_locations = self.game.get_orderable_locations(
            self.power_name
        )

        # For retreat/build phases just choose a legal action
        if self.game.phase_type != 'M':
            power_orders = []

            for loc in orderable_locations:
                if possible_orders[loc]:
                    power_orders.append(
                        random.choice(possible_orders[loc])
                    )

            return power_orders

        # Find all supply centres that we do not own
        my_centres = self.game.get_centers(self.power_name)

        target_centres = []

        for centre in self.game.map.scs:
            if centre not in my_centres:
                target_centres.append(centre)

        power_orders = []

        # Decide an action for each unit
        for loc in orderable_locations:

            loc_orders = possible_orders[loc]

            # Work out if the unit is an army or fleet
            army_orders = [
                order for order in loc_orders
                if order.startswith('A ')
            ]

            fleet_orders = [
                order for order in loc_orders
                if order.startswith('F ')
            ]

            if army_orders:
                unit_type = 'A'
                graph = self.map_graph_army

            elif fleet_orders:
                unit_type = 'F'
                graph = self.map_graph_navy

            else:
                continue

            # If already on a centre that is not ours, stay there
            if loc in target_centres:
                hold_order = f'{unit_type} {loc} H'

                if hold_order in loc_orders:
                    power_orders.append(hold_order)

                continue

            # Get shortest paths from the current location
            if loc not in graph:
                continue

            paths = nx.shortest_path(
                graph,
                source=loc
            )

            closest_centre = None
            closest_distance = float('inf')

            for centre in target_centres:

                if centre not in paths:
                    continue

                distance = len(paths[centre]) - 1

                if distance < closest_distance:
                    closest_distance = distance
                    closest_centre = centre

            # If there is a reachable centre, move one step towards it
            if closest_centre is not None:

                path = paths[closest_centre]

                if len(path) > 1:
                    next_loc = path[1]

                    move_order = (
                        f'{unit_type} {loc} - {next_loc}'
                    )

                    if move_order in loc_orders:
                        power_orders.append(move_order)
                        continue

            # Otherwise just hold
            hold_order = f'{unit_type} {loc} H'

            if hold_order in loc_orders:
                power_orders.append(hold_order)

        return power_orders

        '''
        Return a list of orders. Each order is a string, with specific format. For the format, read the game rule and game engine documentation.
        
        Expected format:
        A LON H                  # Army at LON holds
        F IRI - MAO              # Fleet at IRI moves to MAO (and attack)
        A WAL S F LON            # Army at WAL supports Fleet at LON (and hold)
        F NTH S A EDI - YOR      # Fleet at NTH supports Army at EDI to move to YOR
        F NWG C A NWY - EDI      # Fleet at NWG convoys Army at NWY to EDI
        A NWY - EDI VIA          # Army at NWY moves to EDI via convoy
        A WAL R LON              # Army at WAL retreats to LON
        A LON D                  # Disband Army at LON
        A LON B                  # Build Army at LON
        F EDI B                  # Build Fleet at EDI

        Note: If an invalid order is sent to the engine, it will be accepted but with a result of 'void' (no effect).
        Note: For a 'support' action, two orders are needed, one for the supporter and one for the supportee. (Same for 'convoy')
        Note: For each unit, if no order is given, it will 'hold' by default.

        Useful Functions:
        
        # This is a dict of all the possible orders for each unit at each location (for all powers).
        possible_orders = self.game.get_all_possible_orders()

        # This is a list of all orderable locations for the power you control.
        orderable_locations = self.game.get_orderable_locations(self.power_name)
    
        # Combining these two, you can have the full action space for the power you control.

        # You can re-use the build_map_graphs function in the GreedyAgent to build the connection graph of the map if needed.
        
        '''
