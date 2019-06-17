import numpy as np
from wgomoku import (
    GomokuBoard, Move, StochasticMaxSampler, HeuristicGomokuPolicy, 
    ThreatSearch, Heuristics, GomokuTools as gt)
h = Heuristics(kappa=3.0)
ts = ThreatSearch(max_depth=6, max_width=5)
p = HeuristicGomokuPolicy(style = 2, bias=.5, topn=5, threat_search=ts)

winner = -1
def play(board, p, N, ts):
    colors = ['Black', 'White']
    for i in range(N):
        moves, won = ts.is_tseq_won(board)
        if won:
            x, y = moves[0]
            #print ("Thread sequence: %s: " % moves)
        else:
            probas = p.probas(board, 2)
            sampler = StochasticMaxSampler(np.ndenumerate(probas), bias=0.5, topn=5)
            move = sampler.draw()
            x, y = gt.m2b(move, size=19)
        board.set(x,y)

        status = board.game_state()
        if status == 1: 
            winner = 1 - board.current_color
            break
        elif status == -1: 
            winner = board.current_color
            break
      
    
for i in range(1000):
    board = GomokuBoard(h, N=19, disp_width=10)
    board.set(10,10).set(9,9)
    play(board, p, 40, ts)
    status = board.game_state()
    print(gt.stones_to_string(board.stones)+ ", %s" % winner)