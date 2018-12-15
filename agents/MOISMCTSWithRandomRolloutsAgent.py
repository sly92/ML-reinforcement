import random
from math import sqrt, log

from environments import InformationState, GameRunner, GameState
from environments.Agent import Agent


class MOISMCTSWithRandomRolloutsAgent(Agent):

    def __init__(self, iteration_count: int, runner: GameRunner,
                 reuse_tree=True,
                 k=0.2
                 ):
        self.iteration_count = iteration_count
        self.reuse_tree = reuse_tree
        self.k = k
        self.runner = runner
        self.current_trees = {}
        self.current_iteration_selected_nodes = {}

    def observe(self, reward: float, terminal: bool) -> None:
        pass

    def act(self, player_index: int, information_state: InformationState, available_actions: 'Iterable[int]') -> int:

        for i in range(self.iteration_count):
            self.current_iteration_selected_nodes = {}
            gs = information_state.create_game_state_from_information_state()

            # SELECT
            gs, info_state, current_player, terminal = self.select(gs)

            if not terminal:
                # EXPAND
                node = self.current_trees[current_player][info_state]

                available_actions = gs.get_available_actions_id_for_player(current_player)
                node['a'] = [{'n': 0, 'r': 0, 'action_id': action_id} for action_id in available_actions]
                child_action = random.choice(node['a'])
                action_to_execute = child_action['action_id']

                self.add_visited_node(node, child_action, current_player)

                gs, reward, terminal = gs.step(current_player, action_to_execute)

            # EVALUATE
            scores = self.runner.run(initial_game_state=gs, max_rounds=1)

            # BACKPROPAGATE SCORE
            for player_id in self.current_iteration_selected_nodes.keys():
                visited_nodes = self.current_iteration_selected_nodes[player_id]
                for node, child_action in reversed(visited_nodes):
                    node['nprime'] += 1
                    child_action['n'] += 1
                    child_action['r'] += scores[player_id]

        child_action = max(self.current_iteration_selected_nodes[player_index][0][0]['a'],
                           key=lambda child: child['n'])

        return child_action['action_id']

    def select(self, gs: GameState):
        terminal = False
        while True:
            current_player = gs.get_current_player_id()
            info_state = gs.get_information_state_for_player(current_player)

            if terminal:
                return gs, info_state, current_player, True

            if not current_player in self.current_trees:
                self.current_trees[current_player] = {}

            current_tree = self.current_trees[current_player]
            if not info_state in current_tree:
                current_tree[info_state] = {
                    'nprime': 0
                }
                return gs, info_state, current_player, False

            current_node = current_tree[info_state]

            child_action = max(current_node['a'],
                               key=lambda node: (
                                   (node['r'] / node['n'] + self.k * sqrt(log(current_node['nprime']) / node['n']))
                                   if node['n'] > 0
                                   else 99999999))

            action_to_execute = child_action['action_id']

            self.add_visited_node(current_node, child_action, current_player)

            gs, reward, terminal = gs.step(current_player, action_to_execute)

    def add_visited_node(self, node, selected_action, current_player):
        if not current_player in self.current_iteration_selected_nodes:
            self.current_iteration_selected_nodes[current_player] = []

        self.current_iteration_selected_nodes[current_player].append((node, selected_action))
