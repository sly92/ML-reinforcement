import csv
import re
import os
from time import time
from collections import OrderedDict
from pprint import pprint

from games.windjammers.runners.SafeWindJammersRunner import SafeWindJammersRunner
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
from agents.TabularQLearningAgent import TabularQLearningAgent

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from agents.CommandLineAgent import CommandLineAgent
from environments import Agent
from environments.GameRunner import GameRunner
import tensorflow as tf
from games.windjammers.WindJammersGameState import WindJammersGameState
import numpy as np


class TensorboardInstrumentedRunner(GameRunner):

    def __init__(self, agent1: Agent, agent2: Agent, log_dir_root="./logs/"):
        self.agents = (agent1, agent2)
        self.writer = tf.summary.FileWriter(log_dir_root)

    def run(self, max_rounds: int = -1,
            initial_game_state: WindJammersGameState = WindJammersGameState()) -> 'Tuple[float]':
        round_id = 0
        self.round_duration_sum = 0.0
        self.mean_action_duration_sum = np.array((0.0, 0.0))

        while round_id < max_rounds or round_id == -1:
            gs = initial_game_state.copy_game_state()
            terminal = False
            round_step = 0
            self.mean_action_duration = {0: 0.0, 1: 0.0}
            self.action_duration_sum = {0: 0.0, 1: 0.0}
            self.accumulated_reward_sum = {0: 0.0, 1: 0.0}

            while not terminal:
                round_time = time()
                current_player = gs.get_current_player_id()
                action = 0

                if current_player != -1:
                    action_ids = gs.get_available_actions_id_for_player(current_player)
                    info_state = gs.get_information_state_for_player(current_player)
                    action_time = time()
                    action = self.agents[current_player].act(current_player,
                                                             info_state,
                                                             action_ids)
                    action_time = time() - action_time
                    self.action_duration_sum[current_player] += action_time

                (gs, score, terminal) = gs.step(current_player, action)
                self.agents[0].observe(score, terminal)
                self.agents[1].observe(-score, terminal)

                self.accumulated_reward_sum[0] = score
                self.accumulated_reward_sum[1] = -score
                round_step += 1

            self.round_duration = time() - round_time
            self.round_duration_sum += self.round_duration
            self.mean_action_duration = (
                self.action_duration_sum[0] / round_step, self.action_duration_sum[1] / round_step)
            self.mean_action_duration_sum += (self.mean_action_duration[0], self.mean_action_duration[1])

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

        return tuple(self.accumulated_reward_sum), self.round_duration_sum, self.mean_action_duration_sum

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

    agentList2 = ["RandomAgent()",
                  "RandomRolloutAgent(3, SafeWindJammersRunner(RandomAgent(), RandomAgent()))",
                  "TabularQLearningAgent()",
                  "DeepQLearningAgent(8,12)",
                  "ReinforceClassicAgent(8,12)",
                  "ReinforceClassicWithMultipleTrajectoriesAgent(8,12)",
                  "PPOWithMultipleTrajectoriesMultiOutputsAgent(8,12)",
                  "MOISMCTSWithValueNetworkAgent(100,8,12)",
                  "MOISMCTSWithRandomRolloutsAgent(100, SafeTicTacToeRunner(RandomAgent(), RandomAgent()))",
                  "MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(100, SafeTicTacToeRunner(RandomAgent(), RandomAgent()),9,9)"
                  ]
    agentList = ["RandomAgent()",
                 "DeepQLearningAgent(8,12)",
                 ]

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
                                                                                           log_dir_root="./logs/" + battle_name).run(
                    num_games)
                mean_round_time = round_sum_time / num_games
                mean_action_time_a1 = sum_action_duration[0] / num_games
                mean_action_time_a2 = sum_action_duration[1] / num_games

                TensorboardInstrumentedRunner(eval(agent1), eval(agent2)).createStats(a1_name, a2_name,
                                                                                      score, battle_name, num_games,
                                                                                      mean_round_time,
                                                                                      mean_action_time_a1, mean_action_time_a2)
        i += 1
