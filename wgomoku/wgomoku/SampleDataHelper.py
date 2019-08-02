import numpy as np
import tensorflow as tf
from .GomokuTools import GomokuTools as gt

class ValueDataHelper:
    def __init__(self, N, representation, edges=True, cut_off=None):
        """
        Helper supporting various representations of a particular gomoku position
        params:
        N: board size
        representation: either of "NxNx2", "NxNx1"
            "NxNx2" uses two matrices alternatingly, the player to move next always on the top matrix
            "NxNx1" uses a single matrix, the player to move always +1, the other -1
            "NxNx1B" uses one matrix for the players (like NxNx1) and a second for the border. 
        edges: boolean, whether to model edges as defensive stones or not. Not relevant for NxNx1B
        """
        self.N = N
        self.rep = representation
        self.edges = edges
        self.cut_off = cut_off
        
        
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
        maybe_padded_stones = self.maybe_padded_coords(board)

        # all equivalent representations: 4xrot, 2xflip = 8 variants 
        all_traj = self.all_symmetries(maybe_padded_stones)
        
        all_samples = []
        all_values = []
        for traj in all_traj:
            samples, values = self.traj_to_samples(traj, 
                                              terminal_value=terminal_value, 
                                              gamma=gamma) 

            # Take only the last, most meaningful values
            if self.cut_off: 
                samples = samples[-self.cut_off:]            
                values = values[-self.cut_off:]

            all_samples = all_samples + samples
            all_values = all_values + values
            
        return all_samples, all_values

    
    def maybe_padded_coords(self, board):
        """
        return stones in matrix coordinates respecting the border padding
        """
        stones = gt.string_to_stones(board)
        matrix_coords = [gt.b2m([ord(s[0])-64, s[1]], self.N) for s in stones]
        if self.edges:
            return np.add(matrix_coords, [1,1])
        else: 
            return matrix_coords

    
    def all_symmetries(self, coords):
        """
        All 8 equivalent game plays
        coords: border-padded coordinates
        """
            
        return [
            self.rot_flip(coords, quarters=quarters, flip=flip) 
            for quarters in range(4) for flip in [False, True]
        ]

    
    def create_sample(self, moves, to_move):
        sample = self.template()
        for move in moves:
            if self.rep == 'NxNx2':
                sample[move[0], move[1], to_move] = 1
            elif self.rep == 'NxNx1B':
                sample[move[0], move[1], 0] = 1 - 2 * to_move
            else:
                sample[move[0], move[1]] = 1 - 2 * to_move
            to_move = 1 - to_move
        return sample
    
    
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
            moves = traj[:t+1]            
            sample = self.create_sample(moves, to_move_first)

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

        if self.rep == 'NxNx1B':
            border = 1
        elif self.edges:
            if self.rep == 'NxNx1':
                border = -1
            else:
                border = 1
        if self.edges or self.rep == 'NxNx1B':
            defensive = np.pad(s, pad_width=1, constant_values=border)
        else: 
            defensive = s
        
        if self.rep == "NxNx2":
            if self.edges:
                N=self.N+2
            else:
                N=self.N
            offensive = np.zeros([N, N], dtype=np.int16)
            template = np.stack([offensive, defensive], axis=-1)
        elif self.rep == 'NxNx1B':
            offensive = np.zeros([self.N+2, self.N+2], dtype=np.int16)
            template = np.stack([offensive, defensive], axis=-1)            
        else: 
            template = defensive

        return template
    
    
    def rot_flip (self, coords, quarters, flip):
        """
        coords: list of tuples of matrix coordinates
        quarters: multiples of 90 degrees to rotate
        flip: boolean: Flip or not
        """
        if self.edges:
            N = self.N+2
        else:
            N = self.N

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
    
    
class SampleDataHelper (ValueDataHelper):
    """
    Adds policy-related utilities to the ValueDataHelper
    """
    def __init__(self, N, representation, edges=True, cut_off=None):
        """
        Helper supporting various representations of a particular gomoku position
        params:
        N: board size
        representation: either of "NxNx2", "NxNx1"
            "NxNx2" uses two matrices alternatingly, the player to move next always on the top matrix
            "NxNx1" uses a single matrix, the player to move always +1, the other -1
            "NxNx1B" uses one matrix for the players (like NxNx1) and a second for the border. 
        edges: boolean, whether to model edges as defensive stones or not. Not relevant for NxNx1B
        cut_off: the number of positions per trajectory, counted from the end backwards
        Allows to focus on the later stages of the game
        """
        super().__init__(N, representation, edges, cut_off)

        
    def from_string_with_actions(self, board):
        """
        Creates an symmetry-augmented dataset of board positions and their actions.
        The samples will have shape [N_moves*8, N+2, N+2, 2]

        board: The string rep of the stones, like e.g. "j10k11i9"
        """
        
        # stones in matrix coordinates respecting the border padding
        maybe_padded_stones = self.maybe_padded_coords(board)

        # all equivalent representations: 4xrot, 2xflip = 8 variants 
        all_traj = self.all_symmetries(maybe_padded_stones)
        
        all_samples = []
        all_values = []
        for traj in all_traj:
            samples, values = self.traj_to_samples_with_actions(traj)
                        
            if self.cut_off: 
                samples = samples[-self.cut_off:]            
                values = values[-self.cut_off:]

            all_samples = all_samples + samples
            all_values = all_values + values        

            
        return all_samples, all_values

    def traj_to_samples_with_actions(self, traj):
        """
        creates samples from a given trajectory, together with their actions

        traj: trajectory, represented by their padded coordinates
        terminal_value: the given value of the last state in the trajectory
        gamma: discount factor. 
        """
        samples = []
        actions = []
        to_move_first = 1
        for t in range(len(traj)-1):      
            moves = traj[:t+1]
            next_move = traj[t+1]
            if self.rep == 'NxNx1B' or self.edges:
                next_move -= [1, 1]
            action = np.zeros([self.N, self.N], dtype=np.float32)
            action[next_move[0]][next_move[1]] = 1.0
            sample = self.create_sample(moves, to_move_first)
            
            # Hopefully, we won't need this
            #occupied = np.abs(np.rollaxis(sample, 2, 0)[0].T[1:-1].T[1:-1])
            #action -= occupied * .1
            
            samples.append(sample)
            actions.append(action)

            to_move_first = 1 - to_move_first

            
        return samples, actions
    
    
    def sample_from_string(self, game):
        coords = self.maybe_padded_coords(game)
        to_move = 1 if len(coords) % 2 == 1 else 0
        return self.create_sample(coords, to_move)
 

    def stones_from_NxNx1B (self, smp):
        """
        reconstructs stones from a NxNx1B sample representation
        Note that the moves will not (likely) be in their original order 
        """
        board = np.rollaxis(smp, 2, 0)[0].T[1:-1].T[1:-1]
        to_move = np.sum(board > 0)
        other = np.sum(board < 0)

        if other > to_move:
            BLACK, WHITE = -1, 1
        else:
            BLACK, WHITE = 1, -1

        r,c = np.where(board == WHITE)
        x,y = gt.m2b((r,c), self.N)
        white_moves = list(zip(x,y))

        r,c = np.where(board == BLACK)
        x,y = gt.m2b((r,c), self.N)
        black_moves = list(zip(x,y))

        moves = np.zeros([len(black_moves) + len(white_moves),2])
        moves[::2,] = np.array(black_moves)
        moves[1::2,] = np.array(white_moves)
        stones = [(int(move[0]), int(move[1])) for move in moves]

        return stones

    
def new_policy_dataset(file_pattern, sdh,
                batch_size=256, num_epochs=1, buffer_size=500):
    
    games = tf.data.experimental.make_csv_dataset(
        file_pattern = file_pattern,
        column_names=["game", "winner"],
        batch_size=1,
        num_epochs=num_epochs
    ) 
    def _generator(): 
        for batch in iter(games):
            game = tf.squeeze(batch['game']).numpy().decode('ascii')
            smp, lbl = sdh.from_string_with_actions(game)
            lbl = np.reshape(lbl, [-1, 19*19])
            zipped = zip(smp, lbl)
            for s_and_v in zipped:
                yield s_and_v
    
    inputs = tf.data.Dataset.from_generator(
        _generator, output_types=(tf.int32, tf.float32))
    
    inputs = inputs.shuffle(buffer_size).batch(batch_size)
    return inputs
    
    
    
    
def analyse_and_recommend(smp, pi, n_best, N=19):
    """
    calculates the n_best best positions in board coordinates of smp, 
    looking at the sample and its policy result pi(smp)
    Params:
    smp: np.ndarray: A sample in NxNx1B representation (using 1,-1 for the stones)
    pi:  np.ndarray: the NxN policy pi(a|s=smp)
    n_best: the n_best positions to return. Note that there may be further positions
    having pi match the worst of the best positions. We make a hard cut so that the
    number of returned positions is exactly n_best.
    """

    # mask occupied positions
    occupied = np.abs(np.rollaxis(smp, 2, 0)[0].T[1:-1].T[1:-1])
    
    ##
    ##  Super-evil hack! Will come back and solve, I promise!
    ##  Background: pi seems to be shifted down-right. Can't find out, why.
    ##  So I shift it back up-left here before continuing.
    ##
    pi = np.vstack([pi, np.zeros(19)])[1:,] # up
    pi = np.hstack([pi, np.zeros([19,1], dtype=np.float32)]).T[1:,].T #left
    
    distr = (1 - occupied) * pi

    max_likely = sorted(np.reshape(distr, [N*N]))[::-1][:n_best]
    yn = distr == max_likely[0]
    for i in range(1,n_best):
        yn = np.add(yn, distr == max_likely[i]) 
    r, c = np.where(yn)
    greedy_positions = [
        (gt.m2b((rr,cc), N), distr[rr][cc])
        for rr, cc in zip(r,c)    
    ]
    
    return distr, greedy_positions[:n_best]


