import numpy as np
from .GomokuTools import GomokuTools as gt

class SampleDataHelper:
    def __init__(self, N):
        """
        N: board size
        """
        self.N = N
        
        
    def from_string_with_bellmann(self, board, terminal_value=None, gamma=1.0):
        """
        Creates an symmetry-augmented dataset of board positions and their values.
        Values are calculated as Bellmann rollouts backwards from terminal_value with alternating 
        signs using discount factor gamma.
        The samples will have shape [N_moves*8, N+2, N+2, 2]

        If terminal_value is 1, expect the values to be [..., gamma^2, -gamma, gamma, -1, 1]

        board: The string rep of the stones, like e.g. "j10k11i9"
        N: the size of the board
        """
        
        # stones in matrix coordinates respecting the border padding
        padded_stones = self.padded_coords(board)

        # all equivalent representations: 4xrot, 2xflip = 8 variants 
        all_traj = self.all_symmetries(padded_stones)
        
        all_samples = []
        all_values = []
        for traj in all_traj:
            samples, values = self.traj_to_samples(traj, 
                                              terminal_value=terminal_value, 
                                              gamma=gamma)
            all_samples = all_samples + samples
            all_values = all_values + values        
            
        return all_samples, all_values

    
    def padded_coords(self, board):
        """
        return stones in matrix coordinates respecting the border padding
        """
        stones = gt.string_to_stones(board)
        matrix_coords = [gt.b2m([ord(s[0])-64, s[1]], self.N) for s in stones]
        return np.add(matrix_coords, [1,1])

    
    def all_symmetries(self, coords):
        """
        All 8 equivalent game plays
        coords: border-padded coordinates
        """
        return [
            self.rot_flip(coords, quarters=quarters, flip=flip) 
            for quarters in range(4) for flip in [False, True]
        ]

    
    def traj_to_samples(self, traj, terminal_value, gamma):
        """
        creates samples for a given trajectory, together with the bellmann-values

        traj: trajectory, represented by their padded coordinates
        terminal_value: the given value of the last state in the trajectory
        gamma: discount factor. 
        """
        samples = []
        values = []
        value = terminal_value
        to_move_first = 1
        for t in range(len(traj)):      
            to_move = to_move_first
            moves = traj[:t+1]
            sample = self.template()
            for move in moves:
                sample[move[0], move[1], to_move] = 1
                to_move = 1 - to_move
            samples.append(sample)
            values.append(value)
            # Actually, this is the wrong order, we'll invert before returning
            if to_move_first == 0:
                value = - value
            else:
                value = - gamma * value
            to_move_first = 1 - to_move_first

        return samples, values[::-1]
    
    
    def template(self):
        """
        create a fresh empty board representation
        """
        s = np.zeros([self.N,self.N], dtype=np.int16)
        defensive = np.pad(s, pad_width=1, constant_values=1)
        offensive = np.zeros([self.N+2, self.N+2], dtype=np.int16)
        template = np.stack([offensive, defensive], axis=-1)
        return template
    
    
    def rot_flip (self, coords, quarters, flip):
        """
        coords: list of tuples of matrix coordinates
        quarters: multiples of 90 degrees to rotate
        flip: boolean: Flip or not
        """
        N = self.N+2
        r,c = np.transpose(coords)
        quarters = quarters % 4
        if quarters == 0:
            res = (r,c)
        elif quarters == 1:
            res = (N-c-1, r)
        elif quarters == 2:
            res = (N-r-1,N-c-1)
        elif quarters == 3:
            res = (c, N-r-1)
        if flip:
            res = res[::-1]
        return np.transpose(res)