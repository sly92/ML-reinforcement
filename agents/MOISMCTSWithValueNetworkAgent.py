import random
from math import sqrt, log

import numpy as np
import tensorflow as tf
from keras.activations import relu, linear
from tensorflow.python.layers.core import dense

from environments import InformationState, GameState
from environments.Agent import Agent


class ValueNetworkBrain():
    def __init__(self, state_size, num_players, num_layers=2, num_neurons_per_layer=512,
                 session=None):
        self.state_size = state_size
        self.num_players = num_players
        self.num_layers = num_layers
        self.num_neurons_per_layers = num_neurons_per_layer
        self.states_ph = tf.placeholder(shape=(None, state_size), dtype=tf.float64)
        self.target_values_ph = tf.placeholder(shape=(None, num_players), dtype=tf.float64)
        self.values_op, self.train_op = self.create_network()

        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()

        if not self.session:
            self.session = tf.Session()

        self.session.run(tf.global_variables_initializer())

    def create_network(self):
        hidden = self.states_ph

        for i in range(self.num_layers):
            hidden = dense(hidden, self.num_neurons_per_layers, activation=relu)

        values_op = dense(hidden, self.num_players, activation=linear)

        loss = tf.reduce_mean(tf.square(values_op - self.target_values_ph))

        train_op = tf.train.AdamOptimizer().minimize(loss)

        return values_op, train_op

    def predict_state(self, state):
        return self.predict_states([state])[0]

    def predict_states(self, states):
        return self.session.run(self.values_op, feed_dict={
            self.states_ph: states
        })

    def train(self, states, target_values):
        return self.session.run(self.train_op, feed_dict={
            self.states_ph: states,
            self.target_values_ph: target_values
        })


class MOISMCTSWithValueNetworkAgent(Agent):

    def __init__(self, iteration_count: int,
                 state_size,
                 num_players,
                 brain=None,
                 reuse_tree=True,
                 k=0.2,
                 gamma=0.99
                 ):
        self.iteration_count = iteration_count
        self.reuse_tree = reuse_tree
        self.k = k
        self.gamma = gamma
        self.brain = brain
        self.num_players = num_players
        self.state_size = state_size
        if not brain:
            self.brain = ValueNetworkBrain(state_size, num_players)
        self.current_trees = {}
        self.current_iteration_selected_nodes = {}
        self.current_trajectory = []
        self.current_transition = None

    def observe(self, reward: float, terminal: bool) -> None:
        if not self.current_transition:
            return

        self.current_transition['r'] += reward
        self.current_transition['terminal'] |= terminal

        if terminal:
            R = 0
            self.current_trajectory.append(self.current_transition)
            self.current_transition = None
            for transition in reversed(self.current_trajectory):
                R = transition['r'] + self.gamma * R
                accumulated_rewards = np.ones(self.num_players) * R
                for i in range(self.num_players):
                    if i != transition['player_index']:
                        accumulated_rewards[i] = -R / (self.num_players - 1)
                transition['R'] = accumulated_rewards

            states = np.array([transition['s'] for transition in self.current_trajectory])
            target_values = np.array([transition['R'] for transition in self.current_trajectory])

            self.brain.train(states, target_values)
            self.current_trajectory = []

    def act(self, player_index: int, information_state: InformationState, available_actions: 'Iterable[int]') -> int:

        if self.current_transition:
            self.current_transition['terminal'] = False
            self.current_trajectory.append(self.current_transition)
            self.current_transition = None

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
            scores = self.brain.predict_state(info_state.vectorize())

            # BACKPROPAGATE SCORE
            for player_id in self.current_iteration_selected_nodes.keys():
                visited_nodes = self.current_iteration_selected_nodes[player_id]
                for node, child_action in reversed(visited_nodes):
                    node['nprime'] += 1
                    child_action['n'] += 1
                    child_action['r'] += scores[player_id]

        child_action = max(self.current_iteration_selected_nodes[player_index][0][0]['a'],
                           key=lambda child: child['n'])

        self.current_transition = {
            's': information_state.vectorize(),
            'r': 0,
            'player_index': player_index,
            'terminal': False
        }

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
