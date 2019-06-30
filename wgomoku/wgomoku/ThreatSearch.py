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



class Node:
    def __init__(self, move, value, children=None, parent=None):
        self.move = move
        self.value = value
        self.children = children or []
        self.parent = parent
       
    @staticmethod
    def from_list(l, parent=None):
        if len(l) == 1:
            root = Node(l[0], 0, children=[], parent = parent)
            return root

        else:
            node = Node(l[0], 0, children=[Node.from_list(l[1:])])
            node.children[0].parent = node
            return node
        
    def add_child(self, move, value, children = []):
        child = Node(move, value, children, self)
        self.children.append(child)
        return child
                
    def trajectory(self):
        parent = self
        t=[]
        while parent:
            t.append(parent.move)
            parent = parent.parent
        return t[::-1]
    
    def select(self, predicate):

        #print("before:", self, self.children)
        self.children = [child for child in self.children if child.select(predicate)]
        #print("after: ", self, self.children)
 
        return predicate(self) or len(self.children) > 0
            
 
    def all_trajectories(self):        
        if len(self.children) == 0:
            return [(self.trajectory(), self.value)]
        else:
            res = []
            for child in self.children:
                res += child.all_trajectories()
            return res
            
    
    def __repr__(self):
        return str(self.move)


    
    
class ThreatSearch():
    
    def __init__(self, max_depth, max_width):
        self.max_depth = max_depth
        self.max_width = max_width
    
    def is_threat(self, board, policy, x, y):
        board.set(x,y)
        mcp = policy.most_critical_pos(board, consider_threat_sequences=False)
        board.undo()
        return mcp    
    
    def is_tseq_won(self, board, max_depth=None, max_width=None):
        """if winnable by a threat sequence, returns that sequence as a list of moves.
        Otherwise returns an empty list"""
        
        max_depth = max_depth or self.max_depth
        max_width = max_width or self.max_width
        
        board = deepcopy(board)
        
        # Need a new policy on the copy.
        policy = HeuristicGomokuPolicy(style=0, bias=1.0, topn=5, threat_search=self)

        tree = Node.from_list(board.stones)
        while tree.children:
            tree = tree.children[0]
        return self._is_tseq_won(board, policy, max_depth, max_width, [], tree), tree

    
    def _is_tseq_won(self, board, policy, max_depth, max_width, moves, root):

        if max_depth < 1:
            return moves, False

        #print(moves)

        crit = policy.most_critical_pos(board, consider_threat_sequences=False) 
        #print("critical:" + str(crit)) 
        if crit and crit.off_def == -1: # must defend, threat sequence is over
            #print(moves)
            #print("critical:" + str(crit)) 
            return moves, False

        sampler = policy.suggest_from_score(board, max_width, 0, 2.0)
        for c in sampler.choices:
            #print("checking move: " + str(c))
            x,y = gt.m2b((c[1][0],c[1][1]), board.N)
            threat = root.add_child((x,y), 0)

            if self.is_threat(board, policy, x,y):
                board.set(x,y)
                moves.append((x,y))
                defense0 = policy.suggest(board)

                if defense0.status == -1: # The opponent gave up
                    return moves, True     
                else: 
                    branches = []
                    for defense in policy.defense_options(board, defense0.x, defense0.y):
                        # A single successful defense would make this branch useless

                        p = deepcopy(policy)
                        b = deepcopy(board)
                        m = deepcopy(moves)
                        b.set(defense[0], defense[1])
                        m.append((defense[0], defense[1]))
                        def_node = threat.add_child(defense, 0)
                        branches.append(self._is_tseq_won(b, p, max_depth-1, max_width, m, def_node))

                    won = np.all([br[1] for br in branches])

                    if not won:
                        board.undo()
                        moves = moves[:-1]
                    else:
                        threat.value = 1
                        for d in threat.children:
                            d.value = -1
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
