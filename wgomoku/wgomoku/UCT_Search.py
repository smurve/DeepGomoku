from copy import deepcopy
import numpy as np
from .Heuristics import won_or_lost
from .GomokuBoard import GomokuBoard

def uct_search(game_state, policy, num_reads, C, verbose=0, rollout_delay=1.0):
    """
    DEPRECATED!!!! See GomokuHeuristic_UCT and UCT_Search.py for current state of the art.
    """
    def _is_terminal(value):
        return value == 1 or value == -1
    
    def _choose_from(distr):
        action_distr = np.rollaxis(np.array([[i[0], i[1]] for i in distr.items()]), -1, 0)
        actions = action_distr[0]
        probas = list(action_distr[1])
        return np.random.choice(actions, 1, p=probas)[0]    

    count = [0,0,0]
    root = UCT_Node(game_state, C)
    for read_count in range(num_reads):

        # UCB-driven selection
        leaf = root.select_leaf()
        
        # policy-advised rollout until terminal state
        leaf_priors = policy.evaluate(leaf.game_state)
        value=won_or_lost(leaf.game_state)

        priors = leaf_priors
        game = leaf.game_state
        
        if read_count > rollout_delay * num_reads:
            while value == 0:
                move = _choose_from(priors)
                game, reward, terminal, info = env.step(move)
                priors = policy.evaluate(game)
                value = won_or_lost(board)
        
        count[value+1] += 1

        if _is_terminal(value_estimate):
            if verbose > 1:
                print(leaf, leaf.game_state)
            leaf.backup(value_estimate)
        else:
            # Only expand non-terminal states
            leaf.expand(leaf_priors)
            
    if verbose > 0:
        print("Counts: %s" % count)
    move, _ = max(root.children.items(), 
               key = lambda item: item[1].number_visits)
    return root, move


class UCT_Node:
    def __init__(self, game_state, C, pos=None, parent=None):
        self.game_state = game_state
        self.pos = pos
        self.is_expanded = False
        self.parent = parent
        self.children = []
        self.C = C
        self.penalty = 0
        self.child_moves=[]

        
    def any_argmax(self, aa):
        """
        stochastic function:
        returns any of the indices that have the maximum.
        """
        import random
        ind = np.argmax(aa)
        m = aa[ind]
        choices = np.where(aa == m)
        return random.choice(np.ndarray.tolist(choices[0]))        

    
    def child_Q(self):
        if self.number_visits == 0: 
            return np.zeros([len(self.children)], dtype=float)
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        if self.number_visits == 0: 
            return np.zeros([len(self.children)], dtype=float)
        else:
            return np.sqrt(np.log(self.number_visits) / 
                           (1 + self.child_number_visits))

    def best_child(self):
        """
        We are looking for the child that has the *worst* value, because
        the value is always computed as from the point of view of the one
        who's moving next. And the children of this node are all adversary
        moves. In short: Maximize the opponents pain! My move is your pain.
        """
        qus = - self.child_Q() + self.C * self.child_U()
        pos = self.any_argmax(qus) 
        if self.children:
            return self.children[pos]
        else:
            return None

    
    def select_leaf(self):
        current = self
        while current.is_expanded:
            current.number_visits += 1
            current.total_value -= self.penalty #make it less attractive until backup
            current = current.best_child()
        return current
    
    def expand(self, child_priors):
        self.is_expanded = True
        items = child_priors.items()
        self.child_moves=[i[0] for i in items]
        self.child_priors = [i[1] for i in items]
        self.child_total_value = np.zeros(
            [len(child_priors)], dtype=np.float32)
        self.child_number_visits =  np.zeros(
            [len(child_priors)], dtype=np.float32)
        self.children=[None]*len(child_priors)
        for pos in range(len(child_priors)):
            self.add_child(pos)
            
    def add_child(self, pos):
        new_game_state = deepcopy(self.game_state)
        move = self.child_moves[pos]
        new_game_state.stones.append(move)
        new_game_state.to_play = -new_game_state.to_play
        child = UCT_Node(new_game_state, C=self.C, pos=pos, parent=self)
        self.children[pos] = child
        return child
    
    def backup(self, value_estimate):
        current = self
        while current.parent is not None:
            upd = value_estimate * current.game_state.to_play + current.penalty
            #print("Updating: %s: %s, %s" % (self, current.game_state.to_play, upd))
            current.total_value += upd
            current = current.parent
            
    def pathid(self):
        if not self.parent:
            return [None]
        leaf = self
        name = [leaf.parent.child_moves[leaf.pos]]
        name = leaf.parent.pathid() + name
        return name    
            
    def __repr__(self):
        return "Path: " + str(self.pathid())
        
    @property
    def number_visits(self):
        if self.parent:
            return self.parent.child_number_visits[self.pos]
        else:
            return sum([c.number_visits for c in self.children])
    
    @number_visits.setter
    def number_visits(self, value):
        #print("Value: %s" % value)
        #raise ValueError()
        if self.parent:
            self.parent.child_number_visits[self.pos] = value
        
    @property
    def total_value(self):
        if self.parent:
            return self.parent.child_total_value[self.pos]
        else:
            return 0
    
    @total_value.setter
    def total_value(self, value):
        if self.parent:
            self.parent.child_total_value[self.pos] = value
            
            
            
class PolicyAdapter:
    
    def __init__(self, policy, heuristic):
        """
        policy: HeuristicGomokuPolicy
        """
        self.policy = policy
        self.topn = policy.topn
        self.bias = policy.bias
        self.heuristic = heuristic
    
    def evaluate(self, state):
        board = GomokuBoard(heuristics=self.heuristic, stones=state.stones, N=state.size)
        distr = self.policy.distr(board, self.topn, self.bias)
        value_estimate = self.policy.value(board) / 200.0
        return distr, value_estimate, won_or_lost(board)
    
    
from copy import deepcopy
from .Heuristics import won_or_lost, is_terminated

class GomokuEnvironment:
    
    def __init__(self, heuristic, N=20, disp_width=10, initial_stones=[]):
        """
        This environment will always reset to those initial_stones
        """
        self.initial_board = GomokuBoard(
            heuristic, N, disp_width=disp_width, stones=initial_stones)
        
    def state(self):
        # Note that the current_color of the board is the one that's just moved.
        to_play = -1 if self.board.current_color == 0 else 1
        return GomokuState(self.board.stones, size=self.board.N, to_play=to_play)
        
    def reset(self):
        self.board = deepcopy(self.initial_board)
        return self.state()
    
    def step(self, action):
        self.board.set(*action)
        reward = won_or_lost(self.board)
        return self.state(), reward, is_terminated(self.board), ""    
    
    
class GomokuState:
    def __init__(self, stones, size, to_play):
        """
        to_play: -1 for white, 1 for black
        """
        self.stones = stones
        self.size = size
        self.to_play = to_play