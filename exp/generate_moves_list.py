import numpy as np
import json
import chess

def to_int(square):
    if square is None:
        return None
    a, b = square
    return 5 * a + b

qdirections = [[1, 1], [1, 0], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, 0],
               [-1, -1]]
movs_queen = []
for i in range(6):
    for j in range(5):
        for direction in qdirections:
            for dist in range(1, 6):
                from_square = np.array([i, j])
                to_square = from_square + dist * np.array(direction)
                if not ((0 <= to_square[0] < 6) and (0 <= to_square[1] < 5)):
                    to_square = None
                movs_queen.append((from_square, to_square))

kdirections = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1],
               [-2, -1]]
movs_knight = []
for i in range(6):
    for j in range(5):
        for direction in kdirections:
            from_square = np.array([i, j])
            to_square = from_square + np.array(direction)
            if not ((0 <= to_square[0] < 6) and (0 <= to_square[1] < 5)):
                to_square = None
            movs_knight.append((from_square, to_square))

movs = tuple(movs_queen + movs_knight)



moves_uci = []
for mov in movs:
    from_square = to_int(mov[0])
    to_square = to_int(mov[1])
    if from_square is not None and to_square is not None:
        moves_uci.append(chess.Move(from_square=from_square, to_square=to_square).uci())

moves_dict = dict()
moves_dict['w'] = {v: k for k,v in dict(enumerate(moves_uci)).items()}

moves_uci = []
for mov in movs:
    from_square = to_int(mov[0])
    to_square = to_int(mov[1])
    if from_square is not None and to_square is not None:
        moves_uci.append(chess.Move(from_square=29-from_square, to_square=29-to_square).uci())

moves_dict['b'] = {v: k for k,v in dict(enumerate(moves_uci)).items()}

with open('moves_dict.json', 'w') as file:
    json.dump(moves_dict, file)
