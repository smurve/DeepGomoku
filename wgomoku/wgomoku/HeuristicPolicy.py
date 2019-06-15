from operator import itemgetter
import numpy as np
from copy import deepcopy
from .GomokuTools import GomokuTools as gt

MUST_DEFEND=-1
CAN_ATTACK=1

class Move:
    def __init__(self, x, y, comment, status, off_def=0):
        self.status=status # 0: ongoing, -1: giving up, 1: declaring victory
        self.x = x
        self.y = y
        self.comment = comment
        self.off_def = off_def # +1 = can attack, -1 = must defend
        
    def __repr__(self):
        return self.comment+ ("" if self.status == -1 
                              else ": (%s, %s)" % (chr(self.x+64), self.y))
    

class StochasticMaxSampler:
    """
    This class allows to sample from the top n of an array of scores, 
    with higher probability for the larger scores. With bias > 1.0,
    the sampler has an even higher bias toward the larger scores.
    """
    def __init__(self, array, topn, bias=1.0):
        self.array = array
        self.bias = bias
        
        top = sorted(list(array), key=itemgetter(1))[-topn:]
        values = [v for _,v in top if v != 0]
        positions = [p for p,v in top if v != 0]

        biased = (values - min(values)) * self.bias
        probs = np.exp(np.asarray(biased))
        probs = probs / probs.sum(0)
        boundaries = [0.]+list(np.cumsum(probs))
        self.probs = probs
        self.choices = list(zip(boundaries[:-1], positions, probs, values))[::-1]

    def draw(self):        
        r = np.random.uniform(0,1)
        for i in self.choices:
            if r > i[0]:
                return i[1]
        
SURE_WIN = 1
IMMEDIATE_WIN = 2
SURE_LOSS = -1
ONGOING = 0
    
class HeuristicGomokuPolicy:
    def __init__(self, style, bias, topn, threat_search=None):
        """
        Params:
        style: 0=aggressive, 1=defensive, 2=mixed
        bias: bias towards the higher values when choosing from a distribution. Low bias tries more.
        topn: number of top moves to include in distribution
        threat_search: The ThreatSearch instance
        """
        self.style = style # 0=aggressive, 1=defensive, 2=mixed
        self.bias = bias
        self.topn = topn
        self.ts = threat_search
        
    
    def pos_and_scores(self, board, index, viewpoint):
        "index: the index of the scored position in a flattened array"
        mpos = np.divmod(index, board.N)
        bpos = gt.m2b(mpos, board.N)
        return (bpos[0], bpos[1], 
            board.scores[viewpoint][mpos[0]][mpos[1]])
      
    
    def most_critical_pos(self, board, consider_threat_sequences=True):
        "If this function returns not None, take the move or die."
    
        viewpoint = board.current_color
        clean_scores = board.get_clean_scores()
        o = np.argmax(clean_scores[1-viewpoint])
        d = np.argmax(clean_scores[viewpoint]) 
        xo, yo, vo = self.pos_and_scores(board, o, 1-board.current_color)
        xd, yd, vd = self.pos_and_scores(board, d, board.current_color)
        #print(xo, yo, vo)
        #print(xd, yd, vd)
        if vo > 7.0:
            return Move(xo, yo, "Immediate win", IMMEDIATE_WIN, CAN_ATTACK)
        elif vd > 7.0:
            if sorted(clean_scores[viewpoint].reshape(board.N*board.N))[-2] > 7.0:
                return Move(0,0,"Two or more immediate threats. Giving up.", 
                            SURE_LOSS, MUST_DEFEND)
            return Move (xd, yd, "Defending immediate threat", ONGOING, MUST_DEFEND)
        elif vo == 7.0:
            return Move(xo, yo, "Win-in-2", SURE_WIN, CAN_ATTACK)
        elif vd == 7.0:
            options = self.defense_options(board, xd, yd)
            l = list(zip(options, np.ones(len(options))))
            sampler = StochasticMaxSampler(l, len(options))
            xd, yd = sampler.draw()
            return Move(xd, yd, "Defending Win-in-2", ONGOING, MUST_DEFEND)

        elif vo == 6.9:
            return Move(xo, yo, "Soft-win-in-2", ONGOING, CAN_ATTACK)
        elif vd == 6.9:
            return Move(xd, yd, "Defending Soft-win-in-2", ONGOING, MUST_DEFEND)
        
        elif self.ts and consider_threat_sequences:
            # I might have a winning threat sequence...
            moves, won = self.ts.is_tseq_won(board)
            if won:
                x, y = moves[0]
                #print(moves)
                return Move(x, y, "Pursuing winning threat sequence", ONGOING, CAN_ATTACK)

            # I might need to defend a threat sequence...
            moves = self.ts.is_tseq_threat(board)
            if moves:
                #print(moves)
                x, y = moves[0]
                return Move(x, y, "Defending lurking threat sequence", ONGOING, MUST_DEFEND)
        else:    
            return None
        
    def defense_options(self, board, xd, yd):
        """
        return a list of all options that could remedy the critical state
        """
        color = board.current_color
        options=[]        
        rc = gt.b2m((xd, yd), board.N)
        for direction in ['e', 'ne', 'n', 'nw']:
            step = np.array(gt.dirs()[direction][1])
            for w in [-4,-3,-2,-1,1,2,3,4]:
                r, c = rc+w*step
                if r >= 0 and r < board.N and c >= 0 and c < board.N:
                    x,y = gt.m2b((r,c), board.N)
                    if (x,y) not in board.stones:
                            board.set(x,y)
                            board.compute_scores(color)
                            clean_scores = board.get_clean_scores()
                            s=clean_scores[color][r][c]
                            board.undo()
                            board.compute_scores(color)
                            clean_scores = board.get_clean_scores()
                            s1=clean_scores[color][r][c]
                            if s < 7.0 and s1 == 7.0:
                                options.append((x,y))
        options.append((xd,yd))
        return options

                                
    def distr(self, board, topn=None, bias=None, style=None):
        """
        computes a dict of moves (x,y) with their priors based on the current board
        """
        style = style or self.style
        bias = bias or self.bias
        topn = topn or self.topn

        critical = self.most_critical_pos(board)
        if critical:
            return {(critical.x, critical.y): 1.0}
        else:
            choices = self.suggest_from_best_value(board, topn, style, bias).choices
            return {(gt.m2b(c[1], board.N)[0], gt.m2b(c[1], board.N)[1]): c[2] for c in choices}
       
    def probas (self, board, style):
        """    
        Returns the (suggested) probabilities for a move to be taken.
        In case, masks all but the positions belonging to the most critical category. 
        Params:
        board: a gomoku board.
        style: 0=aggressive, 1=defensive, 2=mixed
        In non-critical situations, style = 0 looks at offensive options only,
        style = 1 looks at defensive options only and style = 2 considers both aspects
        equally.
        """
        scores = board.get_clean_scores()
        to_move = 1 - board.current_color
        o = scores[to_move]
        d = scores[1 - to_move]

        criticals = [lambda: o > 7.0, lambda: d > 7.0,
                     lambda: o == 7.0, lambda: d == 7.0,
                     lambda: o == 6.9, lambda: d == 6.9]

        for crit in criticals:
            field = crit()
            if field.any():
                return field / field.sum()

        if style == 0:
            noncrit = scores[to_move]
        elif style == 1:
            noncrit = scores[1-to_move]
        elif style == 2: 
            noncrit = scores[to_move] + scores[1-to_move] 

        exp = np.exp(noncrit)
        return np.floor( exp / exp.sum() * 1000 ) / 1000
                   
        
    def value(self, board):
        """
        The value of the board as from the scores.
        """
        return board.get_value(compute_scores=True)
        
                                
    def suggest(self, board, style=None, bias=None, topn=None):
        style = style or self.style
        bias = bias or self.bias
        topn = topn or self.topn
            
        critical = self.most_critical_pos(board)
        if critical is not None:
            return critical
        else:
            sampler = self.suggest_from_best_value(board, topn, style, bias)
            r_c = sampler.draw()
            x, y = gt.m2b(r_c, board.N)
            return Move(x, y, "Style: %s" % style, 0)


    def suggest_naive(self, board, style=None, bias=1.0, topn=10):
        """
        Non-deterministic! Samples from the highest naive scores
        """
        if style == None:
            style = self.style
        critical = self.most_critical_pos(board, consider_threat_sequences=False)
        if critical is not None:
            return critical
        else:
            sampler = self.suggest_from_score(board, topn, style, bias)
            r_c = sampler.draw()
            x, y = gt.m2b(r_c, board.N)
            return Move(x, y, "Style: %s" % style, 0)

        
    def suggest_from_score(self, board, n, style, bias):
        """
        return a sampler for the top n choices of the given style with a bias > 1.0
        towards the larger scores
        """
        from operator import itemgetter

        if style == None:
            style = self.style

        clean_scores = board.get_clean_scores()

        viewpoint = board.current_color
        w_o, w_d = 0.5, 0.5 # relative weights

        if style == 0:  # offensive scores
            scores = clean_scores[1-viewpoint]
            
        elif style == 1: # defensive scores
            scores = clean_scores[viewpoint] 

        elif style == 2: # weighted sum of both scores
            scores = (w_o * clean_scores[1-viewpoint] + 
                      w_d * clean_scores[viewpoint])

        scores = np.ndenumerate(scores)
        return StochasticMaxSampler(scores, n, bias)
            

    def suggest_from_best_value(self, board, n, style, bias, nscores=10):

        sampler = self.suggest_from_score(board, max(n, nscores), style, bias)

        scores = []
        for choice in sampler.choices:
            move = gt.m2b(choice[1], board.N)
            board.set(*move)
            for color in [0,1]:
                board.compute_scores(color)

            # naive-strongest defense assumed
            counter = self.suggest_naive(board, style=2, bias=1.0, topn=3)

            if counter.status == -1:
                #print("Opponent gave up")
                #print(str(counter))
                scores = [(choice[1], np.float32(6.95))] # 6.95 is a 'marker'. 
                board.undo()
                break
            
            board.set(counter.x, counter.y)
            
            for color in [0,1]:
                board.compute_scores(color)
            value = board.get_value()
            
            board.undo().undo()
            for color in [0,1]:
                board.compute_scores(color)
            
            scores.append((choice[1], value))
            
        return StochasticMaxSampler(scores, n, bias)
            
        
        

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

        return self._is_tseq_won(board, policy, max_depth, max_width, [])

    
    def _is_tseq_won(self, board, policy, max_depth, max_width, moves):

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

            if self.is_threat(board, policy, x,y):
                board.set(x,y)
                moves.append((x,y))
                defense0 = policy.suggest(board)

                if defense0.status == -1: # The opponent gave up
                    return moves, True     
                else: 
                    #print(board.stones)
                    #print("defense:" + str(defense0))
                    #print(policy.defense_options(defense0.x, defense0.y))

                    branches = []
                    for defense in policy.defense_options(board, defense0.x, defense0.y):
                        # A single successful defense would make this branch useless

                        p = deepcopy(policy)
                        b = deepcopy(board)
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