import numpy as np
from .GomokuTools import GomokuTools

class NH9x9:
    
    """
    9-by-9 neighbourhood of an empty Gomoku position. Provides 12 8 bit integers representing 
    what is visibile from that field in a particular direction as input for a valuation function.
    Example: the six stones seen from '*' in south-east/north-west:

    - - - - - - - - x
    - - - - - - - x - 
    - - - - - - - - - 
    - - - - - x - - -
    - - - - * - - - - 
    - - - - - - - - - 
    - - o - - - - - - 
    - o - - - - - - -
    o - - - - - - - - 
    
    will be represented by the following bytes encoded as two int32

    Black:
     e : 0 0 0 0   0 0 0 0
    ne : 0 0 0 0   1 0 1 1
     n : 0 0 0 0   0 0 0 0
    nw : 0 0 0 0   0 0 0 0

    White:
     e : 0 0 0 0   0 0 0 0
    ne : 1 1 1 0   0 0 0 0
     n : 0 0 0 0   0 0 0 0
    nw : 0 0 0 0   0 0 0 0
    
    """
    def __init__(self, black=[0,0,0,0], white=[0,0,0,0], edges=[0,0,0,0]):
        for n in [black, white, edges]:
            for i in range(4):
                assert(n[i] >= 0 and n[i] < 256)
        self.b = black
        self.w = white
        self.e = edges
        
        
    def register(self, color, direction, distance):
        assert(color==0 or color==1)
        assert(direction>=0 and direction<8)
        assert(distance>=1 and distance <=4)
        
        if color==0:
            self.b[direction%4] |= GomokuTools.as_bit(direction, distance)
        else:
            self.w[direction%4] |= GomokuTools.as_bit(direction, distance)
        return self
        
    def unregister(self, color, direction, distance):
        assert(color==0 or color==1)
        assert(direction>=0 and direction<8)
        assert(distance>=1 and distance <=4)
        
        bit = GomokuTools.as_bit(direction, distance)
        if color==0:
            self.b[direction] &= (0xFF ^ bit)
        else:
            self.w[direction] &= (0xFF ^ bit)
        return self

    
    def get_line(self, direction):
        """
        Return two arrays of 8 integers representing black and white stones on a line
        of length 9. The middle position is not represented in the array
        Args:
            direction: either one of 'e', 'ne', 'n', 'nw' or their integer representations
        """
        assert(direction>=0 and direction <= 3)
        
        return [
            GomokuTools.as_bit_array(self.b[direction]),
            GomokuTools.as_bit_array(self.w[direction]),
            GomokuTools.as_bit_array(self.e[direction])]
    

    def as_bits(self):
        return [self.get_line(d) for d in range(4)]

    
    def __repr__(self):
        dirs=[[0, 1], [-1, 1], [-1, 0], [-1, -1]]
        field = [[' ' for i in range(9)] for j in range(9)]
        field[4][4]='*'
        for h in range(4):               
            step = dirs[h]
            pos0 = np.array([4,4]) - 4 * np.array(step)
            bits = self.get_line(h)
            for x in range(8):
                row, col = pos0 + (x + x//4) * np.array(step)
                field[row][col]='x' if bits[0][x] == 1 \
                    else 'o' if bits[1][x] == 1 \
                    else '+' if bits[2][x] == 1 \
                    else ' '                
        return "\n".join([('|' + ' '.join(field[r]) + '|') for r in range(9)])

