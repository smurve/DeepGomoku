from copy import deepcopy
import numpy as np
from .GomokuTools import GomokuTools as gt
from .HeuristicPolicy import HeuristicGomokuPolicy


def least_significant_move(board):
    scores = board.get_clean_scores(tag=1) # tag occupied positions non-zero
    least_score = scores[0] + scores[1]
    index = np.argmin(least_score)
    r, c = np.divmod(index,board.N)

    pos = gt.m2b((r,c), board.N)
    return pos




class ThreatSearch():
    
    def __init__(self, max_depth, max_width):
        self.max_depth = max_depth
        self.max_width = max_width
    
    def is_threat(self, policy, x, y):
        policy.board.set(x,y)
        mcp = policy.most_critical_pos()
        policy.board.undo()
        return mcp    
    
    def is_tseq_won(self, board, max_depth=None, max_width=None):
        """if winnable by a threat sequence, returns that sequence as a list of moves.
        Otherwise returns an empty list"""
        
        max_depth = max_depth or self.max_depth
        max_width = max_width or self.max_width
        
        board = deepcopy(board)
        
        # Need a new policy on the copy.
        policy = HeuristicGomokuPolicy(board=board, style=0)

        return self._is_tseq_won(board, policy, max_depth, max_width, [])

    
    def _is_tseq_won(self, board, policy, max_depth, max_width, moves):

        if max_depth < 1:
            return moves, False

        #print(moves)

        crit = policy.most_critical_pos() 
        #print("critical:" + str(crit)) 
        if crit and crit.off_def == -1: # must defend, threat sequence is over
            #print(moves)
            #print("critical:" + str(crit)) 
            return moves, False

        sampler = policy.suggest_from_score(max_width, 0, 2.0)
        for c in sampler.choices:
            #print("checking move: " + str(c))
            x,y = gt.m2b((c[1][0],c[1][1]), 20)

            if self.is_threat(policy, x,y):
                board.set(x,y)
                moves.append((x,y))
                defense0 = policy.suggest()

                if defense0.status == -1: # The opponent gave up
                    return moves, True     
                else: 
                    #print(board.stones)
                    #print("defense:" + str(defense0))
                    #print(policy.defense_options(defense0.x, defense0.y))

                    branches = []
                    for defense in policy.defense_options(defense0.x, defense0.y):
                        # A single successful defense would make this branch useless

                        p = deepcopy(policy)
                        b = p.board
                        m = deepcopy(moves)
                        b.set(defense[0], defense[1])
                        m.append((defense[0], defense[1]))
                        branches.append(self._is_tseq_won(b, p, max_depth-1, max_width, m))

                    won = np.all([br[1] for br in branches])

                    if not won:
                        board.undo()
                        moves = moves[:-1]
                    else:
                        # all branches are successful. Return any.
                        return branches[0]

        return moves, False
    
    def is_tseq_threat(self, board, max_depth=None, max_width=None):
        
        max_depth = max_depth or self.max_depth
        max_width = max_width or self.max_width        

        board = deepcopy(board)
        x,y = least_significant_move(board)
        board.set(x,y)
        moves, won = self.is_tseq_won(board, max_depth, max_width)
        board.undo()
        if won:
            return moves
        else:
            return []