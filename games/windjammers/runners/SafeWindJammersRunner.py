import os
from time import sleep

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from environments import Agent
from environments.GameRunner import GameRunner
from games.windjammers.WindJammersGameState import WindJammersGameState
import numpy as np


class SafeWindJammersRunner(GameRunner):

    def __init__(self, agent1: Agent, agent2: Agent):
        self.agents = (agent1, agent2)

    def run(self, max_rounds: int = -1,
            initial_game_state: WindJammersGameState = WindJammersGameState()) -> 'Tuple[float]':
        round_id = 0
        accumulated_scores = np.array([0.0, 0.0])
        while round_id < max_rounds or round_id == -1:
            gs = initial_game_state.copy_game_state()
            scores, terminal = gs.get_current_scores()
            while not terminal:
                current_player = gs.get_current_player_id()
                action = 0
                if current_player != -1:
                    action_ids = gs.get_available_actions_id_for_player(current_player)
                    info_state = gs.get_information_state_for_player(current_player)
                    action = self.agents[current_player].act(current_player,
                                                             info_state,
                                                             action_ids)

                (gs, score, terminal) = gs.step(current_player, action)

                self.agents[0].observe(score, terminal)
                self.agents[1].observe(-score, terminal)

                if not terminal:
                    scores, terminal = gs.get_current_scores()
                    accumulated_scores += scores
                round_id += 1

        return tuple(accumulated_scores)


if __name__ == "__main__":
    pass
