import numpy as np
import matplotlib.pyplot as plt
from .GomokuField import GomokuField
from .GomokuTools import GomokuTools
import pandas as pd

BLACK=0
WHITE=1

def maybe_convert(x):
    if type(x) != str:
        return x
    return ord(x)-64        
            
            


class GomokuBoard(GomokuField):
    
    def __init__(self, heuristics, N, disp_width=6, stones=[]):
        GomokuField.__init__(self, N, heuristics=heuristics)
        self.disp_width=disp_width
        self.stones = []
        self.current_color=WHITE
        self.cursor = -1
        for stone in stones:
            self.set(*stone, compute_scores=False)
        self.compute_all_scores()
        self.color_scheme = [ # visualize the offensive/defensive score
            ['#F0F0F0', '#FFC0C0', '#FF9090', '#FF6060', '#FF0000'],
            ['#A0FFA0', '#E8D088', '#FFA080', '#F86040', '#F01808'],
            ['#00FF00', '#B0D818', '#EFB060', '#F07040', '#E03010'],
            ['#00CF00', '#80B014', '#C0A048', '#E08050', '#D04820'],
            ['#00A000', '#307810', '#607020', '#907828', '#C06030']
        ]

      
    def color_for(self, offensive, defensive):
        def _color_for(c):
            if c < 1.5:
                return 0
            elif c <= 4:
                return 1
            elif c < 6:
                return 2
            elif c < 11:
                return 3
            else:
                return 4
        return self.color_scheme[
            _color_for(defensive)][ 
            _color_for(offensive)]
    
        
    def _is_valid(self, index):
        """
        checks the array indexes (not the board coordinates!)
        """
        return index[0] >= 0 and index[0] < self.N and index[1] >= 0 and index[1] < self.N
    
 
    def ctoggle(self):
        """
        toggle current color
        """
        self.current_color = 1 - self.current_color
        return self.current_color
        
        
    def set(self, x,y, compute_scores=True):
        """
        x,y: 1-based indices of the board, x may be an uppercase letter
        """
        x = maybe_convert(x)

        if self.cursor != len(self.stones)-1:
            raise(ValueError("Cursor not at end position."))
        if (x,y) in self.stones:
            print(self.stones)
            raise(ValueError("Position (%s, %s) is occupied." % (x,y)))
        if not self._is_valid((self.N-y, x-1)):
            raise(ValueError("Not a valid move: (%s, %s). Beyond board boundary."  % (x,y)))
        
        self.stones.append((x,y))        
        c = self.current_color
        c_next = self.ctoggle()
        self.cursor = len(self.stones)-1
        
        self.compute_neighbourhoods(x, y, 'r')
        
        if compute_scores:
            self.compute_all_scores()
        
        return self


    def undo(self, compute_scores=True):
        if self.cursor != len(self.stones)-1:
            raise(ValueError("Cursor not at end position."))
        x,y = self.stones[-1]
        self.stones = self.stones[:-1]
        self.compute_neighbourhoods(x, y, 'u')
        self.ctoggle()
        self.cursor = len(self.stones)-1
        if compute_scores:
            self.compute_all_scores()
        
        return self
    
    
    def bwd(self, n=1):
        if ( n > 1 ):
            self.bwd()
            self.bwd(n-1)
            return self
            
        if self.cursor >= 0:
            stone = self.stones[self.cursor]
            self.cursor -= 1
            self.compute_neighbourhoods(*stone, action='u')
            self.ctoggle()
        return self

    
    def fwd(self, n=1):
        if ( n > 1 ):
            self.fwd()
            self.fwd(n-1)
            return self
        if self.cursor < len(self.stones)-1:
            self.cursor += 1
            c = self.current_color
            self.ctoggle()
            self.compute_neighbourhoods(*self.stones[self.cursor], action='r')
        return self
            
    
        
    def display(self, viewpoint=None):
        size=self.N
        if viewpoint=='current':
            viewpoint=self.current_color
        
        fig, axis = plt.subplots(figsize=(self.disp_width, self.disp_width))
        axis.set_xlim([0, size+1])
        axis.set_ylim([0, size+1])
        plt.xticks(range(1,size+1),[chr(asci) for asci in range(65, 65+size)])
        plt.yticks(range(1,size+1))
        axis.set_facecolor('#8080FF')
        xlines = [[ [1, size], [y,y], '#E0E0E0'] for y in range(1, size+1)]
        ylines = [[ [x,x], [1, size], '#E0E0E0'] for x in range(1, size+1)]
        ylines = np.reshape(xlines + ylines, [-1])
        axis.plot(*ylines)
        self.display_helpers(axis)

        if self.cursor >= 0:
            self.display_stones(self.stones, axis)

        if viewpoint is not None:
            self.display_scores(axis, viewpoint)

        
    def display_helpers(self, axis):
        if self.N == 15:
            axis.scatter([4, 4, 12, 12, 8], [4, 12, 12, 4, 8], 
                         s=self.disp_width**2, c='#E0E0E0')
        elif self.N == 19:
            axis.scatter([4, 4, 4, 10, 16, 16, 16, 10, 10], 
                         [4, 10, 16, 16, 16, 10, 4, 4, 10], 
                         s=self.disp_width**2, c='#E0E0E0')
        elif self.N==20:
            axis.scatter([6, 6, 15, 15], [6, 15, 15, 6], 
                         s=self.disp_width**2, c='#E0E0E0')

            
    def display_cursor(self):
        x,y = self.stones[self.cursor]
        box = np.array(
            [[-0.6,  0.6,  0.6, -0.6, -0.6],
            [-0.6, -0.6,  0.6,  0.6, -0.6]])
        box = box + [[x], [y]]
        plt.plot(*box, color='w', zorder=30)
            
            
    def display_stones(self, stones, axis):
        colors=['white', 'black']
        for i in range(1, self.cursor + 2):
            x,y = self.stones[i-1][0:2]
            stc = colors[i % 2]
            fgc = colors[1 - i % 2]
            axis.scatter([x],[y], c=stc, s=self.stones_size(), zorder=10);
            self.display_cursor()
            plt.text(x, y, i, color=fgc, fontsize=12, zorder=20,
                     horizontalalignment='center', verticalalignment='center');

    
    def display_scores(self, axis, viewpoint):
        
        for v in [0,1]:
            self.compute_scores(v)

        for c in range(self.N):
            for r in range(self.N):
                x,y=GomokuTools.m2b((r,c), self.N)
                if (x,y) not in self.stones[:self.cursor+1]:
                    offensive = self.scores[viewpoint][r][c]
                    defensive = self.scores[1-viewpoint][r][c]
                    color = self.color_for(offensive, defensive)
                    if offensive >= 1.5 or defensive >= 1.5:
                        axis.scatter([x],[y], color=color, s=self.stones_size()/4.0, zorder=10)
            
                
    def stones_size(self):
        return 120 / self.N * self.disp_width**2
        

    def save(self, filename):
        df = pd.DataFrame(self.stones)
        df.to_csv(filename, header=None, index=None)

        
    def get_value(self, compute_scores=False):
        scores = self.get_clean_scores(compute_scores)
        o = scores[1-self.current_color]
        d = scores[self.current_color]
        return np.sum(o) - np.sum(d)
        
        #return self._get_value(self.current_color)
  

    def get_clean_scores(self, compute_scores=False, tag=0):
        """ 
        get the scores with the occupied positions zeroed out
        """
        if compute_scores:
            self.compute_all_scores()
            
        cp = self.scores.copy()
        for pos in self.stones:
            r, c = GomokuTools.b2m(pos,self.N)
            for color in [0,1]:
                cp[color][r][c]=tag
        return cp
        
        
    @staticmethod        
    def from_csv(filename, heuristics, size=19, disp_width=10):
        stones = pd.read_csv(filename, header=None).values.tolist()
        return GomokuBoard(heuristics, size, disp_width, stones=stones)

