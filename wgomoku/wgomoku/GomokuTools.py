import numpy as np

class GomokuTools:

    @staticmethod    
    def string_to_stones(encoded):
        """
        returns an array of pairs for a string-encoded sequence
        e.g. [('A',1), ('M',14)] for 'a1m14'
        """
        x, y = encoded[0].upper(), 0
        stones = []
        for c in encoded[1:]: 
            if c.isdigit():
                y = 10 * y + int(c)
            else:
                stones.append((x,y))
                x = c.upper()
                y = 0
        stones.append((x,y)) 
        return stones

    
    @staticmethod    
    def stones_to_string(stones):
        """
        returns a string-encoded sequence for an array of pairs.
        e.g. 'a1m14' for [('A',1), ('M',14)]  
        """
        return "".join( [(s[0].lower() if type(s[0])==str else chr(96+s[0]))+str(s[1]) for s in stones])

    
    @staticmethod    
    def str_base(number, base, width=8):
        def _str_base(number,base):
            (d, m) = divmod(number, base)
            if d > 0:
                return _str_base(d, base) + str(m)
            return str(m)
        s = _str_base(number, base)
        return '0'*(width-len(s))+s

    
    @staticmethod    
    def base2_to_xo(number):
        return GomokuTools.str_base(number, 3).replace('2', 'o').replace('1', 'x').replace('0', '.')    

    
    @staticmethod    
    def mask(offensive, defensive):
        n = defensive
        l = n & 0xF0
        l = (l | l<<1 | l<<2 | l<<3) & 0xF0

        r = n & 0x0F
        r = (r | r>>1 | r>>2 | r>>3) & 0x0F

        mask=(~(l | r)) & 0xFF
        free_stones=mask & offensive

        return free_stones, mask


    @staticmethod    
    def num_offensive(o, d):
        s, l, offset = GomokuTools.mask2(o, d)
        m2o_bits = GomokuTools.as_bit_array(s)[:l]
        max_count = 0
        for w in [2,1,0]:
            i = 0
            while i <= len(m2o_bits) - 2 - w:
                count = sum(m2o_bits[i:i+w+2])
                count = 3*count - (w+2)
                if count > max_count:
                    max_count = count
                i+=1
        if m2o_bits[0] == 0:
            max_count += 1
        if m2o_bits[-1] == 0:
            max_count += 1

        # Criticality correction for the fatal double-open 3
        if max_count == 8:
            max_count=13
        return max_count        
    
    
    @staticmethod    
    def mask2(offensive, defensive):
        n = defensive
        l = n & 0xF0
        l = (l | l<<1 | l<<2 | l<<3) & 0xF0

        r = n & 0x0F
        r = (r | r>>1 | r>>2 | r>>3) & 0x0F

        mask=(~(l | r))
        free_stones=mask & offensive

        free_length=np.sum([(mask>>i)&1 for i in range(8)], axis=0)
        l_offset = np.sum([(l>>i)&1 for i in range(8)], axis=0)
        #free_length = (free_length > 5) * 5 + (free_length <= 5) * free_length
        return free_stones << l_offset, free_length, l_offset    

    
    @staticmethod
    def dirs():
        return {
            'e' : (0, [0, 1]),
            'ne': (1, [-1, 1]),
            'n' : (2, [-1, 0]),
            'nw': (3, [-1, -1]),
            'w' : (4, [0, -1]),
            'sw': (5, [1, -1]),
            's' : (6, [1, 0]),
            'se': (7, [1, 1])}

    @staticmethod
    def int_for_dir(dirstr):
        return GomokuTools.dirs()[dirstr][0]
    
    @staticmethod
    def as_bit(direction, distance):
        return (1<<(distance+3) if direction//4 else 1<<(4-distance))# << (8*(direction%4))
        
    @staticmethod
    def m2b(m, size):
        """matrix index to board position"""
        r, c = m
        return np.array([c+1, size-r])

    @staticmethod
    def b2m(p, size):
        """board position to matrix index"""
        x, y = p
        return np.array([size-y, x-1])
        
    
        
    @staticmethod
    def as_bit_array(n):
        """
        Returns an array of int 0 or 1 
        """
        assert(n >= 0 and n <= 255)
        return [np.sign(n & (1<<i)) for i in range(7, -1, -1)]

    @staticmethod
    def line_for_xo(xo_string):
        """
        return a 2x8 int array representing the 'x..o..' xo_string 
        """
        powers=np.array([2**i for i in range(7, -1, -1)])
        return [sum([1 if (ch=='x' and c==0) 
                 or (ch=='o' and c==1) 
                 else 0 for ch in xo_string] * powers) for c in [0,1]]


