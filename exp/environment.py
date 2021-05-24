import numpy as np
from erlyx.environment import BaseEnvironment, Episode
from chess import Board, Move, STARTING_FEN
from erlyx.types import EpisodeStatus
import json


class TerminatedEpisodeStepException(BaseException):
    pass


class IlegalMoveException(BaseException):
    pass


with open('moves_dict.json', 'r') as file:
    MOVES_DICT = json.load(file)
MOVES_DICT = {True: MOVES_DICT['w'], False: MOVES_DICT['b']}
MOVES_DICT_INV = {side: {v: k for k, v in MOVES_DICT[side].items()} for side in [True, False]}
NUM_ACTIONS = len(MOVES_DICT[True])


class MinitChessEpisode(Episode):
    def __init__(self, fen):
        self._board = Board(fen)
        self._observation = None
        self._reward = None
        self._done = None
        self._board_array = None
        self._legal_moves = None
        self._legal_moves_board = None
        self._update_attributes()

    def _int_to_array(self, bitBoard):
        return np.unpackbits(np.array([bitBoard],
                                      dtype='>i8').view(np.uint8))[-30:]

    def _int_to_board(self, bitBoard, perspective):
        if perspective:
            return self._int_to_array(bitBoard).reshape(6, 5)
        else:
            return self._int_to_array(bitBoard).reshape(6, 5)[::-1, ::-1]

    def _update_attributes(self):
        # observation (fen)
        board, turn, no_progress_count, total_move_count = self._board.fen().split()
        board = board if self.turn else board[::-1].swapcase()
        self._observation = ' '.join([board, no_progress_count, total_move_count])

        # reward, done
        board_result = self._board.result()
        if board_result in ['1-0', '0-1']:
            self._reward, self._done = 1., True
        elif board_result == '1/2-1/2':
            self._reward, self._done = 0., True
        else: 
            self_reward, self._done = None, False

        # board_array
        channels = []
        for color in [self.turn, ~self.turn]:
            for i in range(1, 7):
                channels.append(
                    self._int_to_board(
                        self._board.pieces_mask(piece_type=i, color=color),
                        self.turn))
        channels.append(np.full((6,5), int(no_progress_count)))
        channels.append(np.full((6,5), int(total_move_count)))
        self._board_array = np.asarray(channels)

        # legal moves
        self._legal_moves_uci = list(self._board.legal_moves)
        legal_moves_codes = [MOVES_DICT[self.turn][move.uci()[:4]] for move in self._legal_moves_uci]
        self._legal_moves = sorted(legal_moves_codes)

    def get_observation(self):
        return self._observation

    def get_reward(self):
        return self._reward

    def is_done(self):
        return self._done
    
    def get_board_array(self):
        return self._board_array

    def get_legal_moves(self):
        return self._legal_moves

    @property
    def turn(self):
        return self._board.turn

    def step(self, action, return_status=True):
        if self.is_done():
            raise TerminatedEpisodeStepException
        uci = MOVES_DICT_INV[self.turn][action]
        move = Move.from_uci(uci)
        if move not in self._legal_moves_uci:
            move = Move.from_uci(uci+'q')
        if move in self._legal_moves_uci:
            self._board.push(move)
        else:
            print(f'action: {action}, uci: {uci}, move={move}, legal_moves_uci={self._legal_moves_uci}')
            raise IlegalMoveException
        self._update_attributes()
        if return_status:
            return self.get_status()

    def get_status(self):
        return EpisodeStatus(self.get_observation(), self.get_reward(), self.is_done())


class MinitChessEnvironment(BaseEnvironment):
    def new_episode(self, fen=None):
        episode = MinitChessEpisode(fen or STARTING_FEN)
        return episode, episode.get_observation()
