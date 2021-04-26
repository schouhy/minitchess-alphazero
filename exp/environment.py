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

    def is_done(self):
        return self._board.result() != '*'

    @property
    def turn(self):
        return self._board.turn


    def step(self, action, return_status=True):
        if self.is_done():
            raise TerminatedEpisodeStepException
        legal_moves = list(self._board.legal_moves)
        uci = MOVES_DICT_INV[self.turn][action]
        move = Move.from_uci(uci)
        if move not in legal_moves:
            move = Move.from_uci(uci+'q')
        if move in legal_moves:
            self._board.push(move)
        else:
            print(f'action: {action}, uci: {uci}, move={move}, legal_moves={list(legal_moves)}')
            raise IlegalMoveException
        if return_status:
            return self.get_status()

    def get_legal_moves(self):
        legal_moves = list(self._board.legal_moves)
        legal_moves_codes = [MOVES_DICT[self.turn][move.uci()[:4]] for move in legal_moves]
        return sorted(legal_moves_codes)

    def get_observation(self):
        board, turn, *rest = self._board.fen().split()
        board = board if self.turn else board[::-1].swapcase()
        return ' '.join([board, 'w', *rest])

    def _int_to_array(self, bitBoard):
        return np.unpackbits(np.array([bitBoard],
                                      dtype='>i8').view(np.uint8))[-30:]

    def _int_to_board(self, bitBoard, perspective):
        if perspective:
            return self._int_to_array(bitBoard).reshape(6, 5)
        else:
            return self._int_to_array(bitBoard).reshape(6, 5)[::-1, ::-1]

    def get_board_array(self):
        mask_pieces = []
        for color in [self.turn, ~self.turn]:
            for i in range(1, 7):
                mask_pieces.append(
                    self._int_to_board(
                        self._board.pieces_mask(piece_type=i, color=color),
                        self.turn))
        return np.asarray(mask_pieces).astype(bool)

    def get_status(self):
        result = self._board.result()
        reward = 0.
        # A player cannot loose right after a legal move
        if result in ['1-0', '0-1']:
            reward = 1.
        return EpisodeStatus(self.get_observation(), reward, self.is_done())


class MinitChessEnvironment(BaseEnvironment):
    def new_episode(self, fen=None):
        episode = MinitChessEpisode(fen or STARTING_FEN)
        return episode, episode.get_observation()
