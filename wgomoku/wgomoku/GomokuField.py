import numpy as np
from .GomokuTools import GomokuTools as gt
from .NH9x9 import NH9x9
from .Heuristics import Heuristics

BLACK=0
WHITE=1
EDGES=2

__IMPACT9x9__=[
    [ 
        0x1 << c if r == 4 and c<4 
        else 0x1 << (c-1) if c>4 and r==4

        else 0x100 << c if c == 8-r and c<4
        else 0x100 << (c-1) if c == 8-r and c>4

        else 0x10000 << 8-r-1 if r<4 and c == 4  
        else 0x10000 << 8-r if r>4 and c==4

        else 0x1000000 << 8-c-1 if c == r and c<4
        else 0x1000000 << 8-c if c == r and c>4

        else 0
         for c in range(9) 
    ] for r in range(9) 
]


def impact_from(N, r,c):
    """
    Construct a complete nxn impact representation of a stone at row=r, col=c
    """
    src=np.hstack([
        np.zeros((N+10,c+1),dtype=np.int),
        np.vstack([
            np.zeros((r+1,9), dtype=np.int32), 
            __IMPACT9x9__, 
            np.zeros((N-1-r+1,9), dtype=np.int32)
        ]),
        np.zeros((N+10,N-1-c+1),dtype=np.int)
    ])
    return (src[5:-5].T[5:-5].T).copy()


def as_bytes(l):
    """
    returns an array of 4 bytes representing l, with the LSB at pos 0
    """
    l0=(l & 0xFF)               # east
    l1=(l & 0xFF00) >> 8        # north east
    l2=(l & 0xFF0000) >> 16     # north
    l3=(l & 0xFF000000) >> 24   # north west
    return np.array([l0, l1, l2, l3])




class GomokuField:
    """
    A GomokuField represents the empty positions and what they "see" in each direction
    """
    
    def __init__(self, N, heuristics):
        self.N = N
        self.lines = np.zeros([3, N, N, 4], dtype=int)

        impacts_int32 = np.array([[impact_from(N, r,c) 
                         for c in range(N)] 
                        for r in range(N)])        
        self.impacts = np.rollaxis(as_bytes(impacts_int32), 0, 5)
        
        self.compute_edges()
        
        self.heuristics = heuristics if heuristics is not None else Heuristics()
        
        self.scores = [[],[]]
        
        
    def getnh(self, x,y):
        row, col = gt.b2m((x,y), self.N)
        black = self.lines[BLACK][row][col]
        white = self.lines[WHITE][row][col]
        edges = self.lines[EDGES][row][col]
        return NH9x9(black, white, edges)
        
        
    def compute_neighbourhoods(self, x, y, action):
        r, c = gt.b2m((x,y),self.N)
        if action == 'r':
            self.lines[self.current_color] |= self.impacts[r][c]
        elif action == 'u':
            self.lines[self.current_color] &= (0xFF ^ self.impacts[r][c])
        

    def compute_edges(self):
        edges_sn=[impact_from(self.N, r, c) 
                    for r in [-1, self.N] 
                    for c in range(-1, self.N)]
        edges_ew=[impact_from(self.N, r, c) 
                    for c in [-1, self.N] 
                    for r in range(-1, self.N)]

        for edges in edges_sn + edges_ew:
            edges_as_bytes = np.rollaxis(as_bytes(edges), 0, 3)
            self.lines[EDGES] |= edges_as_bytes


    def compute_all_scores(self):
        for viewpoint in [BLACK, WHITE]:
            self.compute_scores(viewpoint)
            
    def compute_scores(self, viewpoint, upto=None):
        o = self.lines[viewpoint]
        d = self.lines[1-viewpoint] | self.lines[2]
        lines = self.heuristics.lookup_line_score(o, d)
        self.scores[viewpoint] = self.heuristics.lookup_total_scores(lines)

        return
        # Don't correct for positions occupied by stones        
        for stone in stones:
            r, c = gt.b2m(stone,self.N)
            self.scores[viewpoint][r][c]=0
            
        
    def get_score(self, viewpoint, x,y):
        r, c = gt.b2m((x,y),self.N)
        return self.scores[viewpoint][r][c]
    
    
    def _get_value(self, viewpoint):
        o = self.scores[1-viewpoint]
        d = self.scores[viewpoint]
        return np.sum(o) - np.sum(d)
    

