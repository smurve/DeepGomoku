import numpy as np
from .GomokuTools import GomokuTools as gt
from .GomokuBoard import GomokuBoard
from .HeuristicPolicy import HeuristicGomokuPolicy
from .QFunction import heuristic_QF

BLACK=0
WHITE=1
EDGES=2
STYLE_MIXED=2

def roll_out(board, policy, max_n = 40):
    """
    takes the board and subsequently applies the given policy to black and white
    until the game is decided or max_n moves have been played.
    """
    n = 0
    board.compute_all_scores()
    move = policy.suggest(board)
    while move.status == 0 and n < max_n:
        board.set(move.x,move.y)
        board.compute_all_scores()
        move = policy.suggest(board)
        n+=1
    return board


def variants_for(board):
    """
    Create a tensor 8x2xNxN to represent the 8 equivalent boards 
    that can be created from the stones by reflection and rotation.
    """
    stones = board.stones.copy()
    N = board.N
    array=np.zeros([8,2,N,N], dtype=float)
    color = np.arange(len(stones)) % 2
    for l, pos in list(zip(color, stones)):
        r, c = gt.b2m(pos, 15)
        array[0][l][r][c] = 1.0
        array[6][l][c][r] = 1.0

        array[1][l][c][N-r] = 1.0
        array[4][l][N-r][c] = 1.0

        array[3][l][N-c][r] = 1.0
        array[7][l][r][N-c] = 1.0

        array[2][l][N-r][N-c] = 1.0
        array[5][l][N-c][N-r] = 1.0

    return array


def transform(stones, N, quarters, reflect=False):
    """
    return stones' coordinates after rotation and reflection
    """
    coords = [gt.b2m(stone, N) for stone in stones]
    if quarters == 0:
        res = coords

    if quarters == 1:
        res = [(N-c-1, r) for r,c in coords]

    if quarters == 2:
        res = [(N-r-1, N-c-1) for r,c in coords]

    if quarters == 3:
        res = [(c, N-r-1) for r,c in coords]

    if reflect:
        res = [(r, N-c-1) for r,c in res]
        
    stones = [gt.m2b(coord, N) for coord in res]
    return stones


def create_sample(stones, N, viewpoint):
    
    sample = np.zeros([2, N, N], dtype=np.uint8)

    current = 0
    for move in stones:
        r,c=gt.b2m(move,N)
        sample[current][r][c]=1
        current = 1-current

        
    offensive = np.hstack([
        np.zeros([N+2,1], dtype=np.uint8), 
        np.vstack([np.zeros(N, dtype=np.uint8), 
                   sample[viewpoint], 
                   np.zeros(N, dtype=np.uint8)]),
        np.zeros([N+2,1], dtype=np.uint8)
    ])
    
    defensive = np.hstack([
        np.ones([N+2,1], dtype=np.uint8), 
        np.vstack([np.ones(N, dtype=np.uint8), 
                   sample[1-viewpoint], 
                   np.ones(N, dtype=np.uint8)]),
        np.ones([N+2,1], dtype=np.uint8)
    ])
    both = np.array([offensive, defensive])
    return np.rollaxis(both, 0, 3)


def wrap_sample(array, value):
    N = np.shape(array)[0]
    return np.hstack([
        np.zeros([N+2,1], dtype=np.float32) + value, 
        np.vstack([np.zeros(N, dtype=np.float32) + value, 
                   array, 
                   np.zeros(N, dtype=np.float32) + value]),
        np.zeros([N+2,1], dtype=np.float32) + value
    ])


def create_samples_and_qvalues(board, policy, heuristics):
    """
    create 8 equivalent samples and qvalues from the given board
    """
    from copy import deepcopy
    
    # A little tweak: If the last move saw an "immediate win", 
    # we won't need the board that has a finished line of 5.
    if True:
        dcp = deepcopy(board)
        dcp.undo()
        if policy.suggest(dcp).status == 2:
            board.undo().undo()

    all_stones_t = [transform(board.stones.copy(), board.N, rot, ref) 
        for rot in range(4)
        for ref in [False, True]]

    samples = []
    qvalues = []
    avalues = []
    distris = []
    for stones_t in all_stones_t:
        sample = create_sample(stones_t, board.N, 1-board.current_color)
        board = GomokuBoard(heuristics, board.N, stones=stones_t)
        qvalue, default_value = heuristic_QF(board, policy)
        qvalue = wrap_sample(qvalue, default_value)
        distri = policy.probas(board, 2)
        distri = wrap_sample(distri, 0)
        samples.append(sample)
        qvalues.append(qvalue)
        distris.append(distri)
        avalues.append((qvalue-default_value)/100.0)

    return (np.array(samples), 
            np.reshape(qvalues, [8, board.N+2, board.N+2, 1]), 
            np.reshape(avalues, [8, board.N+2, board.N+2, 1]),
            np.reshape(distris, [8, board.N+2, board.N+2, 1]))


def data_from_game(board, policy, heuristics):    
    """
    Careful: This function rolls back the board
    """
    # Don't want to see fours (my heuristics don't work well when the game is essentially done anyway.)
    board.undo(False).undo(False)

    s,q,a = create_samples_and_qvalues(board, policy, heuristics)
    while board.cursor > 6:
        board.undo()
        s1, q1, a1 = create_samples_and_qvalues(board, policy, heuristics)
        s = np.concatenate((s,s1))
        q = np.concatenate((q,q1))
        a = np.concatenate((a,a1))
    return s,q,a

def to_matrix12(sample, size=22):
    field = np.rollaxis(sample.reshape(size,size,2), 2, 0).astype(np.int)
    unwrapped = (field[0]+field[1]*2)[1:-1].T[1:-1].T
    return unwrapped

def to_matrix_xo(sample, size=22):
    if np.sum(to_matrix12(sample, size)>0) % 2 == 0:
        symbols = ['. ', 'x ', 'o ']
    else:
        symbols = ['. ', 'o ', 'x ']
    im12 = to_matrix12(sample, size)
    return "\n".join(["".join([symbols[c] for c in im12[r]]) for r in range(size-2) ])