import numpy as np

from environments import InformationState
from environments.GameState import GameState
from games.quarto.QuartoInformationState import QuartoInformationState


class QuartoGameState(GameState):

    def __init__(self):
        self.current_player = 0
        self.board = np.array(
            (
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1)
            )
        )
        self.pieces = ([
            (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
            (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
            (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1),
            (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)
        ])

    def step(self, player_id: int, action_id: int) -> \
            ('GameState', float, bool):
        if self.current_player != player_id:
            raise Exception("This is not this player turn !")
        val = self.board[action_id // 4][action_id % 4]
        if (val != 0):
            raise Exception("Player can't play at specified position !")

        self.board[action_id // 4][action_id % 4] = \
            1 if player_id == 0 else -1

        (score, terminal) = self.compute_current_score_and_end_game_more_efficient()

        self.current_player = (self.current_player + 1) % 2
        return (self, score, terminal)

    def compute_current_score_and_end_game_more_efficient(self):
        board = self.board
        if self.board[0][0] + self.board[0][1] + self.board[0][2] + self.board[0][3] == 4 or \
                self.board[1][0] + self.board[1][1] + self.board[1][2] + self.board[1][3] == 4 or \
                self.board[2][0] + self.board[2][1] + self.board[2][2] + self.board[2][3] == 4 or \
                self.board[3][0] + self.board[3][1] + self.board[3][2] + self.board[3][3] == 4 or \
                self.board[0][0] + self.board[1][0] + self.board[2][0] + self.board[3][0] == 4 or \
                self.board[0][1] + self.board[1][1] + self.board[2][1] + self.board[3][1] == 4 or \
                self.board[0][2] + self.board[1][2] + self.board[2][2] + self.board[3][2] == 4 or \
                self.board[0][3] + self.board[1][3] + self.board[2][3] + self.board[3][3] == 4 or \
                self.board[0][0] + self.board[1][1] + self.board[2][2] + self.board[3][3] == 4 or \
                self.board[3][0] + self.board[2][1] + self.board[1][2] + self.board[0][3] == 4:
            return 1, True

        if self.board[0][0] + self.board[0][1] + self.board[0][2] + self.board[0][3] == -4 or \
                self.board[1][0] + self.board[1][1] + self.board[1][2] + self.board[1][3] == -4 or \
                self.board[2][0] + self.board[2][1] + self.board[2][2] + self.board[2][3] == -4 or \
                self.board[3][0] + self.board[3][1] + self.board[3][2] + self.board[3][3] == -4 or \
                self.board[0][0] + self.board[1][0] + self.board[2][0] + self.board[3][0] == -4 or \
                self.board[0][1] + self.board[1][1] + self.board[2][1] + self.board[3][1] == -4 or \
                self.board[0][2] + self.board[1][2] + self.board[2][2] + self.board[3][2] == -4 or \
                self.board[0][3] + self.board[1][3] + self.board[2][3] + self.board[3][3] == -4 or \
                self.board[0][0] + self.board[1][1] + self.board[2][2] + self.board[3][3] == -4 or \
                self.board[3][0] + self.board[2][1] + self.board[1][2] + self.board[0][3] == -4:
            return -1.0, True

        if 0 in self.board:
            return 0.0, False
        return 0.0, True

    def get_player_count(self) -> int:
        return 2

    def get_current_player_id(self) -> int:
        return self.current_player

    def get_not_current_player_id(self) -> int:
        return not self.current_player

    def get_information_state_for_player(self, player_id: int) -> 'InformationState':
        return QuartoInformationState(self.current_player,
                                      self.board.copy())

    def get_available_actions_id_for_player(self, player_id: int) -> 'Iterable(int)':
        if player_id != self.current_player:
            return []
        boards_tmp = list(filter(lambda i: self.board[i // 4][i % 4] == 1, range(0, 16)))
        pieces_tmp = np.array(self.pieces)
        available_action = boards_tmp * pieces_tmp
        return available_action

    def __str__(self):
        str = ""
        for i in range(0, 4):
            for j in range(0, 4):
                val = self.board[i][j]

                str += "_" if val == 0 else (
                    "0" if val == 1 else
                    "X"
                )
            str += "\n"
        return str

    def copy_game_state(self):
        gs = QuartoGameState()
        gs.board = self.board.copy()
        gs.current_player = self.current_player
        return gs

    def get_current_scores(self):
        winner, terminal = self.compute_current_score_and_end_game_more_efficient()
        return np.array([winner, -winner]), terminal


if __name__ == "__main__":
    gs = QuartoGameState()
