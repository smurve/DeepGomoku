import tensorflow as tf
import numpy as np
from .GomokuTools import GomokuTools as gt

def detect5(stones):    
    """
    returns a vector (B,W) with B=1.0 iff black has a line of five, same for white.
    Uses a hand-crafted convnet for that
    stones in board coordinates
    """
    bw = np.zeros([2, 19, 19, 1], dtype=np.float16)
    player = 0
    for (x,y) in stones:
        r,c = gt.b2m((x,y), 19)
        bw[player][r][c] = 1
        player = 1 - player
    
    hor=np.zeros([5,5], dtype=np.float16)
    hor[2]=1.
    diag=np.eye(5, dtype=np.float16)
    filters = np.array([hor, hor.T, diag, diag[::-1]])
    kernel_init = tf.constant_initializer(np.rollaxis(filters, 0, 3))
    bias_init = tf.constant_initializer(-4.0)

    detect = tf.keras.layers.Conv2D(bias_initializer=bias_init, kernel_size=5,
        activation='relu', filters=4, padding='same', kernel_initializer=kernel_init)
    pool = tf.keras.layers.MaxPool2D(pool_size=19, strides=1)
    res = pool(detect(bw))
    return np.squeeze(tf.reduce_max(res, axis=-1).numpy())
    
    
def display_sample(sample):
    """
    displays any NxNx2 sample as overlay of 2 NxN matrices with the second channel multiplied by 2
    """
    if tf.is_tensor(sample):
        sample = sample.numpy()
    s = np.rollaxis(sample, -1, 0)
    print(str(s[0]+2*s[1]))
    
    
class TerminalDetector():
    """
    Convnet to detect lines of five. Works for NxNx1B representation only.
    """
    def __init__(self, N):
        self.N = N
        hor=np.zeros([5,5], dtype=np.float16)
        hor[2]=1.
        diag=np.eye(5, dtype=np.float16)
        filters = np.array([hor, hor.T, diag, diag[::-1]])
        kernel_init = tf.constant_initializer(np.rollaxis(filters, 0, 3))
        bias_init = tf.constant_initializer(-4.0)

        self.conv = tf.keras.layers.Conv2D(
            bias_initializer=bias_init, kernel_size=5,
            activation='relu', filters=4, padding='same', 
            kernel_initializer=kernel_init)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=self.N, strides=1)
        
    def call(self, game):
        both = np.rollaxis(game, 2, 0)[0]
        to_move = (both+1) // 2
        other = (1-both) // 2
        both_again = np.stack([to_move, other], axis=0)
        both_again = np.reshape(both_again, [2, 21, 21, 1])        
        both_again = tf.cast(both_again, dtype=tf.float32)
        res = self.pool(self.conv(both_again))
        return np.squeeze(tf.reduce_max(res, axis=-1).numpy())
