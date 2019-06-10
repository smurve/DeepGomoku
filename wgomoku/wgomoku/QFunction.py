from operator import itemgetter
import numpy as np
from .GomokuTools import GomokuTools as gt

MAX_QVALUE=200

def enumerated_top(board, n):
    """
        return the enumerated, ranked lists of offensive and defensive positions
    """
    viewpoint=1-board.current_color
    clean_scores=board.get_clean_scores()
    enumerated_scores = [list(np.ndenumerate(clean_scores[v])) 
                         for v in [viewpoint, 1-viewpoint]]
    res = [sorted(score, key=itemgetter(1))[:-n-1:-1] 
           for score in enumerated_scores]
                         
    return res

def least_significant_move(board):
    scores = board.get_clean_scores(tag=1) # tag occupied positions non-zero
    least_score = scores[0] + scores[1]
    index = np.argmin(least_score)
    r, c = np.divmod(index,board.N)

    pos = gt.m2b((r,c), board.N)
    return pos


def value_after(board, move, policy):

    board.set(*move)
    counter = policy.suggest_naive(board, style=2, topn=1)

    # Some situations in recorded matches are still difficult for my heuristics at this point
    # So I simply ignore the value difference for these few
    if (counter.x, counter.y) == (0,0):
        
        if counter.status == -1: # Opponent gave up.
            return MAX_QVALUE
        else:
            print("IMPLAUSIBLE COUNTER MOVE!")
            print(board.stones)
            print(move)
            board.undo()
            return board.get_value()
    
    board.set(counter.x, counter.y)
    # those take the values after the best responses
    value = board.get_value() 
    
    board.undo(False).undo()
    return value

def heuristic_QF(board, policy):
    """
    returns a matrix for with all qvalues of the given board
    the policy is used for efficiency purposes and as an
    opponent model.
    """
    top_o, top_d = enumerated_top(board, 5)

    q = np.zeros([board.N, board.N], dtype=float)
    max_d = top_d[0][1]
    max_o = top_o[0][1]

    # Critical defensive situation: All but few moves are fatal
    if max_d > 6.9: 
        default_value = - MAX_QVALUE
        q = q - MAX_QVALUE # All options are deadly,...
        for move in top_d: # ...apart from those ranked as critical defenses
            if move[1] > 6.999:
                r,c=move[0]
                pos = gt.m2b(move[0], board.N)
                q[r][c] = value_after(board, pos, policy)

    # Winning situation: all but few moves are irrelevant
    elif max_o >= 6.9:
        default_value = 0.
        for move in top_o:
            if move[1] > 6.999:
                r,c = move[0]
                q[r][c] = MAX_QVALUE

    # for efficiency sake: all but the best 20 are considered as bad as the worst.
    else: 
        sampler = policy.suggest_from_best_value(board, 20, 2, .05)
        default_value = value_after(board, least_significant_move(board), policy)
        q = q + default_value
        for move in sampler.choices:
            r,c=move[1]
            pos = gt.m2b(move[1], board.N)
            q[r][c]=value_after(board, pos, policy)
        
    return q, default_value