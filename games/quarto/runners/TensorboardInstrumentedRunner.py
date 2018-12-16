import csv
import os
import re
import time
from collections import OrderedDict
from pprint import pprint

import tensorflow as tf

from agents.DeepQLearningAgent import DeepQLearningAgent
from agents.CommandLineAgent import CommandLineAgent
from agents.MOISMCTSWithValueNetworkAgent import MOISMCTSWithValueNetworkAgent
from agents.MOISMCTSWithRandomRolloutsAgent import MOISMCTSWithRandomRolloutsAgent
from agents.MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent import MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent
from agents.PPOWithMultipleTrajectoriesMultiOutputsAgent import PPOWithMultipleTrajectoriesMultiOutputsAgent
from agents.RandomAgent import RandomAgent
from agents.RandomRolloutAgent import RandomRolloutAgent
from agents.RandomAgent import RandomAgent
from agents.ReinforceClassicAgent import ReinforceClassicAgent
from agents.ReinforceClassicWithMultipleTrajectoriesAgent import ReinforceClassicWithMultipleTrajectoriesAgent
from agents.TabularQLearningAgent import TabularQLearningAgent

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from environments import Agent
from environments.GameRunner import GameRunner
from games.quarto.QuartoGameState import QuartoGameState
import numpy as np


class TensorboardInstrumentedRunner(GameRunner):
    def __init__(self, agent1: Agent, agent2: Agent,
                 print_and_reset_score_history_threshold=None,
                 replace_player1_with_commandline_after_similar_results=None, log_dir_root="./logs/"):
        self.agents = (agent1, agent2)
        self.stuck_on_same_score = 0
        self.prev_history = None
        self.print_and_reset_score_history_threshold = print_and_reset_score_history_threshold
        self.replace_player1_with_commandline_after_similar_results = replace_player1_with_commandline_after_similar_results
        self.writer = tf.summary.FileWriter(log_dir_root)

    def run(self, max_rounds: int = -1,
            initial_game_state: QuartoGameState = QuartoGameState()) -> 'Tuple[float]':
        round_id = 0
        self.round_duration_sum = 0.0
        self.mean_action_duration_sum = np.array((0.0, 0.0))
        self.score_history = np.array((0, 0, 0))
        while round_id < max_rounds or round_id == -1:
            gs = initial_game_state.copy_game_state()
            terminal = False
            round_time = 0.0
            round_step = 0
            self.action_duration_sum = ([0.0, 0.0])
            self.mean_action_duration = np.array((0.0, 0.0))
            self.accumulated_reward_sum = ([0.0, 0.0])

            while not terminal:
                # print(gs)
                round_time = time.time()
                current_player = gs.get_current_player_id()
                action = 0
                if current_player != -1:
                    action_ids = gs.get_available_actions_id_for_player(current_player)
                    info_state = gs.get_information_state_for_player(current_player)
                    action_time = time.time()
                    action = self.agents[current_player].act(current_player,
                                                             info_state,
                                                             action_ids)
                    action_time = time.time() - action_time
                    self.action_duration_sum[current_player] += action_time

                # WARNING : Two Players Zero Sum Game Hypothesis
                (gs, score, terminal) = gs.step(current_player, action)
                self.agents[current_player].observe(
                    (1 if current_player == 0 else -1) * score,
                    terminal)

                round_step += 1

            self.round_duration = time.time() - round_time
            self.round_duration_sum += self.round_duration
            self.mean_action_duration = (
                self.action_duration_sum[0] / round_step, self.action_duration_sum[1] / round_step)
            self.mean_action_duration_sum += (self.mean_action_duration[0], self.mean_action_duration[1])
            self.score_history += (1 if score == 1 else 0, 1 if score == -1 else 0, 1 if score == 0 else 0)
            other_player = (current_player + 1) % 2
            self.agents[other_player].observe(
                (1 if other_player == 0 else -1) * score,
                terminal)

            self.writer.add_summary(tf.Summary(
                value=[
                    tf.Summary.Value(tag="agent1_action_mean_duration",
                                     simple_value=self.mean_action_duration[0]),

                    tf.Summary.Value(tag="agent2_action_mean_duration",
                                     simple_value=self.mean_action_duration[1]),

                    tf.Summary.Value(tag="round_duration",
                                     simple_value=self.round_duration),

                    tf.Summary.Value(tag="agent1_accumulated_reward",
                                     simple_value=self.accumulated_reward_sum[0]),

                    tf.Summary.Value(tag="agent2_accumulated_reward",
                                     simple_value=self.accumulated_reward_sum[1])

                ],
            ), round_id)

            if round_id != -1:
                round_id += 1
                if self.print_and_reset_score_history_threshold is not None and \
                        round_id % self.print_and_reset_score_history_threshold == 0:
                    # print(score_history / self.print_and_reset_score_history_threshold)
                    if self.prev_history is not None and \
                            self.score_history[0] == self.prev_history[0] and \
                            self.score_history[1] == self.prev_history[1] and \
                            self.score_history[2] == self.prev_history[2]:
                        self.stuck_on_same_score += 1
                    else:
                        self.prev_history = self.score_history
                        self.stuck_on_same_score = 0
                    if (self.replace_player1_with_commandline_after_similar_results is not None and
                            self.stuck_on_same_score >= self.replace_player1_with_commandline_after_similar_results):
                        self.agents = (CommandLineAgent(), self.agents[1])
                        self.stuck_on_same_score = 0
        return tuple(self.score_history), self.round_duration_sum, self.mean_action_duration_sum

    def createStats(self, a1, a2, score, bn, ng, mrt, mat1, mat2):
        print('Score :', score)
        game_stats = OrderedDict({
            'game': 'TicTacToe',
            'battle_name': bn,
            'num_of_games': ng,
            'agent1_name': a1,
            'agent1_wins': score[0],
            'agent1_win_rate': (score[0] / ng) * 100,
            'agent2_name': a2,
            'agent2_wins': score[1],
            'agent2_win_rate': (score[1] / ng) * 100,
            'draw_nb': score[2],
            'draw_rate': (score[2] / ng) * 100,
            'mean_round_time': mrt,
            'mean_action_time_a1': mat1,
            'mean_action_time_a2': mat2

        })

        pprint(game_stats)
        stats_file = '../../../reinforcement_stats.csv'
        exists = os.path.isfile(stats_file)
        with open(stats_file, 'a+') as stats_csv:
            dw = csv.DictWriter(stats_csv, fieldnames=game_stats.keys())
            if not exists:
                dw.writeheader()
            dw.writerow(game_stats)

if __name__ == "__main__":
    # print("Rdm vs Rdm")
    # print(BasicQuartoRunner(RandomAgent(),
    #                            RandomAgent(),

    #                            print_and_reset_score_history_threshold=100).run(1000000))

    # print("Rdm vs ReinforceMonteCarloEpisodicAgent (should be around 80% wins for player 2)")
    # print(BasicQuartoRunner(RandomRolloutAgent(3, BasicQuartoRunner(RandomAgent(), RandomAgent())),
    #                            ReinforceMonteCarloEpisodicAgent(9, 9, lr=0.001, gamma=0.9, num_layers=5,
    #                                                             num_hidden_per_layer=64),
    #                            print_and_reset_score_history_threshold=100).run(100000))

    # print("Rdm vs ReinforceMonteCarloEpisodicAgent (should be around 80% wins for player 2)")
    # print(BasicQuartoRunner(CommandLineAgent(), CommandLineAgent(),
    #                            print_and_reset_score_history_threshold=100).run(100000000))

    # print(BasicQuartoRunner(ReinforceClassicWithMultipleTrajectoriesAgent(9, 9), RandomAgent(),
    #                          print_and_reset_score_history_threshold=100).run(100000000000))

    agentList = ["RandomAgent()", "ReinforceClassicAgent(9,9)", "DeepQLearningAgent(9,9)"]

    for i in range(len(agentList)):
        for j in range(i, len(agentList)):
            if i != j:
                # for k in 1000, 10000, 100000, 1000000:
                agent1 = agentList[i]
                agent2 = agentList[j]

                num_games = 100
                a1_name = re.match("[A-Za-z]+", agent1).group()
                a2_name = re.match("[A-Za-z]+", agent2).group()

                battle_name = a1_name + ' VS ' + a2_name

                score, round_sum_time, sum_action_duration = TensorboardInstrumentedRunner(eval(agent1),
                                                                                           eval(agent2),
                                                                                           log_dir_root="./logs/" + battle_name,
                                                                                           print_and_reset_score_history_threshold=1000).run(
                    num_games)
                mean_round_time = round_sum_time / num_games
                mean_action_time_a1 = sum_action_duration[0] / num_games
                mean_action_time_a2 = sum_action_duration[1] / num_games

                TensorboardInstrumentedRunner(eval(agent1), eval(agent2)).createStats(a1_name, a2_name,
                                                                                      score, battle_name, num_games,
                                                                                      mean_round_time,
                                                                                      mean_action_time_a1, mean_action_time_a2)
        i += 1
